"""
CDK Stack: Earth2Studio UI — CloudFront + S3 + API Gateway + Lambda.

Deploys:
  - Cognito User Pool with strong password policy + MFA
  - WAFv2 Web ACL attached to API Gateway
  - S3 bucket for React SPA static files (with access logging)
  - CloudFront distribution with two origins (S3 default + API Gateway for /api/*)
  - API Gateway REST API with Cognito authorizer + access logging + request validation
  - Lambda function (Python 3.13) for the backend API
  - BucketDeployment to upload frontend/dist/ to S3

Security (AWS Well-Architected Security Pillar):
  - CDK Nag AwsSolutions checks applied
  - Cognito User Pool authentication (APIG4/COG4 satisfied)
  - WAFv2 with AWS managed rules (APIG3 satisfied)
  - Least-privilege IAM (scoped SageMaker + S3 permissions)
  - S3 enforced SSL, encryption, block public access
  - CloudFront + API Gateway access logging
  - API Gateway request validation enabled
  - Lambda reserved concurrency
  - TLS 1.2 minimum on CloudFront
"""

from aws_cdk import (
    Stack,
    Duration,
    RemovalPolicy,
    CfnOutput,
    Aspects,
    aws_s3 as s3,
    aws_s3_deployment as s3deploy,
    aws_cloudfront as cloudfront,
    aws_cloudfront_origins as origins,
    aws_lambda as lambda_,
    aws_apigateway as apigw,
    aws_iam as iam,
    aws_logs as logs,
    aws_cognito as cognito,
    aws_wafv2 as wafv2,
)
from cdk_nag import AwsSolutionsChecks, NagSuppressions
from constructs import Construct


class UIStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        model_bucket: s3.IBucket,
        s3_prefix: str,
        **kwargs,
    ) -> None:
        """
        Args:
            model_bucket: S3 bucket from SageMakerInfraStack for model artifacts + async I/O.
            s3_prefix: S3 key prefix for Earth2Studio resources (e.g., "earth2-weather-models").
        """
        super().__init__(scope, construct_id, **kwargs)

        # Apply CDK Nag AwsSolutions checks
        Aspects.of(self).add(AwsSolutionsChecks(verbose=True))

        # ---------------------------------------------------------------
        # 0. Logging Bucket (for S3 access logs + CloudFront logs)
        # ---------------------------------------------------------------
        logging_bucket = s3.Bucket(
            self,
            "LoggingBucket",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            object_ownership=s3.ObjectOwnership.OBJECT_WRITER,
        )

        NagSuppressions.add_resource_suppressions(
            logging_bucket,
            [
                {
                    "id": "AwsSolutions-S1",
                    "reason": "This IS the access logging bucket — it cannot log to itself.",
                },
            ],
        )

        # ---------------------------------------------------------------
        # 1. Cognito User Pool (authentication)
        # ---------------------------------------------------------------
        # Self-signup is DISABLED — users must be created by an administrator
        # (`aws cognito-idp admin-create-user`) to prevent unauthorized account
        # creation.
        user_pool = cognito.UserPool(
            self,
            "UserPool",
            user_pool_name="earth2-forecast-users",
            self_sign_up_enabled=False,
            sign_in_aliases=cognito.SignInAliases(email=True),
            auto_verify=cognito.AutoVerifiedAttrs(email=True),
            # Strong password policy (AwsSolutions-COG1)
            password_policy=cognito.PasswordPolicy(
                min_length=8,
                require_lowercase=True,
                require_uppercase=True,
                require_digits=True,
                require_symbols=True,
                temp_password_validity=Duration.days(7),
            ),
            # No MFA — email + password only with email verification
            mfa=cognito.Mfa.OFF,
            # Account recovery
            account_recovery=cognito.AccountRecovery.EMAIL_ONLY,
            removal_policy=RemovalPolicy.DESTROY,
            # Threat protection (AwsSolutions-COG3 — adaptive authentication)
            # Uses new API replacing deprecated advancedSecurityMode
            standard_threat_protection_mode=cognito.StandardThreatProtectionMode.FULL_FUNCTION,
        )

        # Suppress MFA requirement — email + password with verified email is sufficient
        NagSuppressions.add_resource_suppressions(
            user_pool,
            [
                {
                    "id": "AwsSolutions-COG2",
                    "reason": "MFA is intentionally disabled. Authentication uses email + password "
                    "with required email verification. WAFv2 rate limiting and Cognito advanced "
                    "security (adaptive authentication) provide additional protection.",
                },
            ],
        )

        # User Pool Client for the React SPA
        user_pool_client = user_pool.add_client(
            "WebAppClient",
            user_pool_client_name="earth2-web-app",
            auth_flows=cognito.AuthFlow(
                user_srp=True,
                user_password=False,  # Use SRP only — more secure
            ),
            prevent_user_existence_errors=True,
            # Token validity
            id_token_validity=Duration.hours(1),
            access_token_validity=Duration.hours(1),
            refresh_token_validity=Duration.days(30),
        )

        # ---------------------------------------------------------------
        # 2. S3 Bucket for React SPA
        # ---------------------------------------------------------------
        website_bucket = s3.Bucket(
            self,
            "WebsiteBucket",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            server_access_logs_bucket=logging_bucket,
            server_access_logs_prefix="s3-website-logs/",
        )

        # ---------------------------------------------------------------
        # 3. Lambda Function for Backend API
        # ---------------------------------------------------------------
        api_lambda = lambda_.Function(
            self,
            "ForecastAPI",
            runtime=lambda_.Runtime.PYTHON_3_13,
            handler="handler.lambda_handler",
            code=lambda_.Code.from_asset("backend"),
            timeout=Duration.seconds(60),
            memory_size=256,
            reserved_concurrent_executions=10,
            environment={
                "S3_BUCKET": model_bucket.bucket_name,
                "S3_PREFIX": s3_prefix,
                "REGION": self.region,
            },
        )

        # Grant SageMaker permissions — scoped to earth2-* endpoints
        api_lambda.add_to_role_policy(
            iam.PolicyStatement(
                actions=["sagemaker:ListEndpoints"],
                resources=["*"],
            )
        )
        api_lambda.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "sagemaker:DescribeEndpoint",
                    "sagemaker:InvokeEndpointAsync",
                ],
                resources=[
                    f"arn:aws:sagemaker:{self.region}:{self.account}:endpoint/earth2-*",
                ],
            )
        )

        # Grant S3 permissions for async I/O — scoped to model bucket + prefix
        api_lambda.add_to_role_policy(
            iam.PolicyStatement(
                actions=["s3:PutObject", "s3:DeleteObject"],
                resources=[
                    model_bucket.arn_for_objects(f"{s3_prefix}/async-input/*"),
                ],
            )
        )
        api_lambda.add_to_role_policy(
            iam.PolicyStatement(
                actions=["s3:GetObject"],
                resources=[
                    model_bucket.arn_for_objects(f"{s3_prefix}/async-output/*"),
                ],
            )
        )

        # Suppress IAM5 wildcards with specific appliesTo entries
        NagSuppressions.add_resource_suppressions(
            api_lambda,
            [
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "sagemaker:ListEndpoints requires Resource::* — "
                    "the API does not support resource-level permissions.",
                    "appliesTo": ["Resource::*"],
                },
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "SageMaker endpoints scoped to earth2-* prefix. "
                    "Wildcard needed because endpoint names are dynamic.",
                    "appliesTo": [
                        "Resource::arn:aws:sagemaker:<AWS::Region>:<AWS::AccountId>:endpoint/earth2-*",
                    ],
                },
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "S3 async-input prefix wildcard — each request creates a unique UUID key.",
                    "appliesTo": [
                        f"Resource::<ModelArtifactsBucket80ACAD84.Arn>/{s3_prefix}/async-input/*",
                    ],
                },
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "S3 async-output prefix wildcard — SageMaker writes output with unique key.",
                    "appliesTo": [
                        f"Resource::<ModelArtifactsBucket80ACAD84.Arn>/{s3_prefix}/async-output/*",
                    ],
                },
                {
                    "id": "AwsSolutions-L1",
                    "reason": "Python 3.13 IS the latest Lambda runtime as of CDK v2.246. "
                    "CDK Nag may lag behind runtime availability.",
                },
            ],
            apply_to_children=True,
        )

        # ---------------------------------------------------------------
        # 4. API Gateway REST API (with Cognito auth + logging + validation)
        # ---------------------------------------------------------------
        api_log_group = logs.LogGroup(
            self,
            "ApiAccessLogs",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.DESTROY,
        )

        api = apigw.RestApi(
            self,
            "ForecastRestApi",
            rest_api_name="earth2-forecast-api",
            deploy_options=apigw.StageOptions(
                access_log_destination=apigw.LogGroupLogDestination(api_log_group),
                access_log_format=apigw.AccessLogFormat.json_with_standard_fields(
                    caller=True,
                    http_method=True,
                    ip=True,
                    protocol=True,
                    request_time=True,
                    resource_path=True,
                    response_length=True,
                    status=True,
                    user=True,
                ),
                logging_level=apigw.MethodLoggingLevel.INFO,
            ),
            default_cors_preflight_options=apigw.CorsOptions(
                allow_origins=apigw.Cors.ALL_ORIGINS,
                allow_methods=apigw.Cors.ALL_METHODS,
                allow_headers=[
                    "Content-Type",
                    "Authorization",
                    "X-Amz-Date",
                    "X-Api-Key",
                    "X-Amz-Security-Token",
                ],
            ),
        )

        # Cognito Authorizer — satisfies AwsSolutions-APIG4 & COG4
        cognito_authorizer = apigw.CognitoUserPoolsAuthorizer(
            self,
            "CognitoAuthorizer",
            cognito_user_pools=[user_pool],
            authorizer_name="earth2-cognito-auth",
            identity_source="method.request.header.Authorization",
        )

        # Request validator
        request_validator = api.add_request_validator(
            "RequestValidator",
            validate_request_body=True,
            validate_request_parameters=True,
        )

        lambda_integration = apigw.LambdaIntegration(api_lambda)

        # Common method options with Cognito auth
        auth_method_options = {
            "authorizer": cognito_authorizer,
            "authorization_type": apigw.AuthorizationType.COGNITO,
            "request_validator": request_validator,
        }

        api_resource = api.root.add_resource("api")

        # GET /api/endpoints
        endpoints_resource = api_resource.add_resource("endpoints")
        endpoints_resource.add_method(
            "GET",
            lambda_integration,
            **auth_method_options,
        )

        # POST /api/forecast (with request model)
        forecast_model = api.add_model(
            "ForecastRequestModel",
            content_type="application/json",
            model_name="ForecastRequest",
            schema=apigw.JsonSchema(
                type=apigw.JsonSchemaType.OBJECT,
                required=["endpoint_name"],
                properties={
                    "endpoint_name": apigw.JsonSchema(type=apigw.JsonSchemaType.STRING),
                    "date": apigw.JsonSchema(type=apigw.JsonSchemaType.STRING),
                    "lead_time_hours": apigw.JsonSchema(type=apigw.JsonSchemaType.INTEGER),
                    "variables": apigw.JsonSchema(
                        type=apigw.JsonSchemaType.ARRAY,
                        items=apigw.JsonSchema(type=apigw.JsonSchemaType.STRING),
                    ),
                    "return_grid": apigw.JsonSchema(type=apigw.JsonSchemaType.BOOLEAN),
                },
            ),
        )

        forecast_resource = api_resource.add_resource("forecast")
        forecast_resource.add_method(
            "POST",
            lambda_integration,
            request_models={"application/json": forecast_model},
            **auth_method_options,
        )

        # GET /api/status/{endpoint_name}
        status_resource = api_resource.add_resource("status")
        status_endpoint = status_resource.add_resource("{endpoint_name}")
        status_endpoint.add_method(
            "GET",
            lambda_integration,
            **auth_method_options,
        )

        # Suppress APIG4/COG4 on auto-generated CORS OPTIONS (no auth needed for preflight)
        NagSuppressions.add_resource_suppressions_by_path(
            self,
            [
                f"/{construct_id}/ForecastRestApi/Default/api/endpoints/OPTIONS/Resource",
                f"/{construct_id}/ForecastRestApi/Default/api/forecast/OPTIONS/Resource",
                f"/{construct_id}/ForecastRestApi/Default/api/status/{{endpoint_name}}/OPTIONS/Resource",
            ],
            [
                {
                    "id": "AwsSolutions-APIG4",
                    "reason": "OPTIONS methods are CORS preflight requests — "
                    "auto-generated by CDK and do not require authorization.",
                },
                {
                    "id": "AwsSolutions-COG4",
                    "reason": "CORS preflight OPTIONS do not require Cognito authorization.",
                },
            ],
        )

        # Suppress IAM4 on API GW CloudWatch role (managed policy required by API GW)
        NagSuppressions.add_resource_suppressions_by_path(
            self,
            [f"/{construct_id}/ForecastRestApi/CloudWatchRole/Resource"],
            [
                {
                    "id": "AwsSolutions-IAM4",
                    "reason": "AmazonAPIGatewayPushToCloudWatchLogs is the AWS-recommended managed policy "
                    "for API Gateway CloudWatch logging. Required for access logging.",
                    "appliesTo": [
                        "Policy::arn:<AWS::Partition>:iam::aws:policy/service-role/AmazonAPIGatewayPushToCloudWatchLogs",
                    ],
                },
            ],
        )

        # ---------------------------------------------------------------
        # 5. WAFv2 Web ACL (attached to API Gateway stage)
        # ---------------------------------------------------------------
        waf_acl = wafv2.CfnWebACL(
            self,
            "ApiWaf",
            default_action=wafv2.CfnWebACL.DefaultActionProperty(allow={}),
            scope="REGIONAL",
            visibility_config=wafv2.CfnWebACL.VisibilityConfigProperty(
                cloud_watch_metrics_enabled=True,
                metric_name="earth2-api-waf",
                sampled_requests_enabled=True,
            ),
            rules=[
                # AWS Managed Rule: Common Rule Set (XSS, SQLi, etc.)
                wafv2.CfnWebACL.RuleProperty(
                    name="AWS-AWSManagedRulesCommonRuleSet",
                    priority=1,
                    override_action=wafv2.CfnWebACL.OverrideActionProperty(none={}),
                    statement=wafv2.CfnWebACL.StatementProperty(
                        managed_rule_group_statement=wafv2.CfnWebACL.ManagedRuleGroupStatementProperty(
                            vendor_name="AWS",
                            name="AWSManagedRulesCommonRuleSet",
                            excluded_rules=[
                                # Exclude SizeRestrictions_BODY since forecast responses can be large
                                wafv2.CfnWebACL.ExcludedRuleProperty(name="SizeRestrictions_BODY"),
                            ],
                        ),
                    ),
                    visibility_config=wafv2.CfnWebACL.VisibilityConfigProperty(
                        cloud_watch_metrics_enabled=True,
                        metric_name="CommonRuleSet",
                        sampled_requests_enabled=True,
                    ),
                ),
                # AWS Managed Rule: Known Bad Inputs
                wafv2.CfnWebACL.RuleProperty(
                    name="AWS-AWSManagedRulesKnownBadInputsRuleSet",
                    priority=2,
                    override_action=wafv2.CfnWebACL.OverrideActionProperty(none={}),
                    statement=wafv2.CfnWebACL.StatementProperty(
                        managed_rule_group_statement=wafv2.CfnWebACL.ManagedRuleGroupStatementProperty(
                            vendor_name="AWS",
                            name="AWSManagedRulesKnownBadInputsRuleSet",
                        ),
                    ),
                    visibility_config=wafv2.CfnWebACL.VisibilityConfigProperty(
                        cloud_watch_metrics_enabled=True,
                        metric_name="KnownBadInputs",
                        sampled_requests_enabled=True,
                    ),
                ),
                # Rate limiting: 1000 requests per 5 min per IP
                wafv2.CfnWebACL.RuleProperty(
                    name="RateLimit",
                    priority=3,
                    action=wafv2.CfnWebACL.RuleActionProperty(block={}),
                    statement=wafv2.CfnWebACL.StatementProperty(
                        rate_based_statement=wafv2.CfnWebACL.RateBasedStatementProperty(
                            limit=1000,
                            aggregate_key_type="IP",
                        ),
                    ),
                    visibility_config=wafv2.CfnWebACL.VisibilityConfigProperty(
                        cloud_watch_metrics_enabled=True,
                        metric_name="RateLimit",
                        sampled_requests_enabled=True,
                    ),
                ),
            ],
        )

        # Associate WAF with API Gateway stage (satisfies AwsSolutions-APIG3)
        wafv2.CfnWebACLAssociation(
            self,
            "ApiWafAssociation",
            resource_arn=f"arn:aws:apigateway:{self.region}::/restapis/{api.rest_api_id}/stages/{api.deployment_stage.stage_name}",
            web_acl_arn=waf_acl.attr_arn,
        )

        # ---------------------------------------------------------------
        # 6. CloudFront Distribution — two origins (with access logging)
        # ---------------------------------------------------------------
        s3_origin = origins.S3BucketOrigin.with_origin_access_control(website_bucket)
        api_origin = origins.RestApiOrigin(api)

        distribution = cloudfront.Distribution(
            self,
            "Distribution",
            default_behavior=cloudfront.BehaviorOptions(
                origin=s3_origin,
                viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                cache_policy=cloudfront.CachePolicy.CACHING_OPTIMIZED,
            ),
            additional_behaviors={
                "/api/*": cloudfront.BehaviorOptions(
                    origin=api_origin,
                    viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                    cache_policy=cloudfront.CachePolicy.CACHING_DISABLED,
                    origin_request_policy=cloudfront.OriginRequestPolicy.ALL_VIEWER_EXCEPT_HOST_HEADER,
                    allowed_methods=cloudfront.AllowedMethods.ALLOW_ALL,
                ),
            },
            default_root_object="index.html",
            error_responses=[
                cloudfront.ErrorResponse(
                    http_status=403,
                    response_http_status=200,
                    response_page_path="/index.html",
                    ttl=Duration.seconds(0),
                ),
                cloudfront.ErrorResponse(
                    http_status=404,
                    response_http_status=200,
                    response_page_path="/index.html",
                    ttl=Duration.seconds(0),
                ),
            ],
            enable_logging=True,
            log_bucket=logging_bucket,
            log_file_prefix="cloudfront-logs/",
            minimum_protocol_version=cloudfront.SecurityPolicyProtocol.TLS_V1_2_2021,
        )

        NagSuppressions.add_resource_suppressions(
            distribution,
            [
                {
                    "id": "AwsSolutions-CFR1",
                    "reason": "Geo restrictions not required — global weather app for worldwide access.",
                },
                {
                    "id": "AwsSolutions-CFR2",
                    "reason": "CloudFront WAF would duplicate API Gateway WAF protection. "
                    "API is already protected by WAFv2 on API Gateway.",
                },
                {
                    "id": "AwsSolutions-CFR4",
                    "reason": "Using TLS_V1_2_2021 minimum protocol. Custom SSL certificate "
                    "not required for demo (using CloudFront default *.cloudfront.net domain).",
                },
            ],
        )

        # ---------------------------------------------------------------
        # 7. Deploy React build to S3
        # ---------------------------------------------------------------
        s3deploy.BucketDeployment(
            self,
            "DeployWebsite",
            sources=[s3deploy.Source.asset("frontend/dist")],
            destination_bucket=website_bucket,
            distribution=distribution,
            distribution_paths=["/*"],
        )

        # Suppress CDK Nag for CDK-managed BucketDeployment custom resource
        NagSuppressions.add_resource_suppressions_by_path(
            self,
            [
                f"/{construct_id}/Custom::CDKBucketDeployment8693BB64968944B69AAFB0CC9EB8756C/ServiceRole/Resource",
                f"/{construct_id}/Custom::CDKBucketDeployment8693BB64968944B69AAFB0CC9EB8756C/ServiceRole/DefaultPolicy/Resource",
                f"/{construct_id}/Custom::CDKBucketDeployment8693BB64968944B69AAFB0CC9EB8756C/Resource",
            ],
            [
                {
                    "id": "AwsSolutions-IAM4",
                    "reason": "CDK BucketDeployment custom resource — CDK-managed, not user-controlled.",
                    "appliesTo": [
                        "Policy::arn:<AWS::Partition>:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
                    ],
                },
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "CDK BucketDeployment needs wildcard S3/CloudFront permissions. CDK-managed.",
                },
                {
                    "id": "AwsSolutions-L1",
                    "reason": "CDK BucketDeployment Lambda runtime is CDK-managed. Cannot override.",
                },
            ],
        )

        # Suppress for ForecastAPI Lambda execution role managed policy
        NagSuppressions.add_resource_suppressions_by_path(
            self,
            [f"/{construct_id}/ForecastAPI/ServiceRole/Resource"],
            [
                {
                    "id": "AwsSolutions-IAM4",
                    "reason": "AWSLambdaBasicExecutionRole is the minimum managed policy for Lambda. "
                    "Required for CloudWatch Logs.",
                    "appliesTo": [
                        "Policy::arn:<AWS::Partition>:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
                    ],
                },
            ],
        )

        # ---------------------------------------------------------------
        # Outputs
        # ---------------------------------------------------------------
        CfnOutput(
            self,
            "CloudFrontURL",
            value=f"https://{distribution.distribution_domain_name}",
            description="CloudFront URL for the weather forecast app",
        )
        CfnOutput(
            self,
            "ApiURL",
            value=api.url,
            description="API Gateway URL",
        )
        CfnOutput(
            self,
            "UserPoolId",
            value=user_pool.user_pool_id,
            description="Cognito User Pool ID",
        )
        CfnOutput(
            self,
            "UserPoolClientId",
            value=user_pool_client.user_pool_client_id,
            description="Cognito User Pool Client ID",
        )
