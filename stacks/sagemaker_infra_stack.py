"""
CDK Stack: SageMaker Infrastructure — IAM Role + S3 Bucket + SSM Parameters.

This stack provisions the long-lived infrastructure that SageMaker endpoints
need, WITHOUT creating the actual endpoints (which are ephemeral and managed
by deploy-all.sh using the SageMaker Python SDK).

Deploys:
  - S3 Bucket for model artifacts + async inference I/O
  - IAM Role for SageMaker execution (least-privilege)
  - SSM Parameters so deploy scripts and Lambda can discover resources

Why this is a separate stack:
  - SageMaker endpoints cost ~$5.90/hr and are created/deleted on-demand
  - The IAM role and S3 bucket are long-lived and should be managed as IaC
  - The deploy-all.sh script references these resources via SSM Parameter Store
  - Independent lifecycle: update infra without touching endpoints, and vice versa

Security (AWS Well-Architected):
  - CDK Nag AwsSolutions checks applied
  - Least-privilege IAM (SageMaker scoped to specific S3 paths + ECR pull)
  - S3 enforced SSL, encryption, block public access
  - S3 lifecycle rules for async I/O cleanup (cost optimization)
"""

from aws_cdk import (
    Stack,
    Duration,
    RemovalPolicy,
    CfnOutput,
    Aspects,
    aws_s3 as s3,
    aws_iam as iam,
    aws_ssm as ssm,
    aws_ecr as ecr,
    aws_codebuild as codebuild,
    aws_secretsmanager as secretsmanager,
)
from cdk_nag import AwsSolutionsChecks, NagSuppressions
from constructs import Construct


class SageMakerInfraStack(Stack):
    """Long-lived SageMaker supporting infrastructure."""

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Apply CDK Nag AwsSolutions checks
        Aspects.of(self).add(AwsSolutionsChecks(verbose=True))

        # ---------------------------------------------------------------
        # Configuration
        # ---------------------------------------------------------------
        s3_prefix = "earth2-weather-models"

        # ---------------------------------------------------------------
        # 1. Logging Bucket (for S3 access logs)
        # ---------------------------------------------------------------
        logging_bucket = s3.Bucket(
            self,
            "SageMakerLogsBucket",
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
        # 2. S3 Bucket for model artifacts + async inference I/O
        # ---------------------------------------------------------------
        model_bucket = s3.Bucket(
            self,
            "ModelArtifactsBucket",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            versioned=True,
            server_access_logs_bucket=logging_bucket,
            server_access_logs_prefix="model-bucket-logs/",
            lifecycle_rules=[
                # Auto-cleanup async inference I/O after 1 day (cost optimization)
                s3.LifecycleRule(
                    id="CleanupAsyncInput",
                    prefix=f"{s3_prefix}/async-input/",
                    expiration=Duration.days(1),
                    enabled=True,
                ),
                s3.LifecycleRule(
                    id="CleanupAsyncOutput",
                    prefix=f"{s3_prefix}/async-output/",
                    expiration=Duration.days(1),
                    enabled=True,
                ),
                # Cleanup non-current versions of model artifacts after 7 days
                s3.LifecycleRule(
                    id="CleanupOldModelVersions",
                    prefix=f"{s3_prefix}/",
                    noncurrent_version_expiration=Duration.days(7),
                    enabled=True,
                ),
            ],
        )

        # ---------------------------------------------------------------
        # 3. IAM Role for SageMaker Execution
        # ---------------------------------------------------------------
        sagemaker_role = iam.Role(
            self,
            "SageMakerExecutionRole",
            role_name="earth2-sagemaker-execution-role",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            description="Execution role for Earth2Studio SageMaker endpoints. "
            "Scoped to specific S3 paths for model artifacts and async I/O.",
        )

        # S3 permissions — scoped to our model bucket + prefixes
        sagemaker_role.add_to_policy(
            iam.PolicyStatement(
                sid="S3ModelArtifactsRead",
                actions=["s3:GetObject", "s3:HeadObject"],
                resources=[
                    model_bucket.arn_for_objects(f"{s3_prefix}/*/model.tar.gz"),
                ],
            )
        )

        sagemaker_role.add_to_policy(
            iam.PolicyStatement(
                sid="S3AsyncInputRead",
                actions=["s3:GetObject", "s3:HeadObject"],
                resources=[
                    model_bucket.arn_for_objects(f"{s3_prefix}/async-input/*"),
                ],
            )
        )

        sagemaker_role.add_to_policy(
            iam.PolicyStatement(
                sid="S3AsyncOutputWrite",
                actions=["s3:PutObject"],
                resources=[
                    model_bucket.arn_for_objects(f"{s3_prefix}/async-output/*"),
                ],
            )
        )

        # SageMaker async inference validation requires s3:ListBucket
        # without prefix conditions, and s3:PutObject on the bucket.
        # These are checked at endpoint creation time.
        sagemaker_role.add_to_policy(
            iam.PolicyStatement(
                sid="S3ListBucket",
                actions=["s3:ListBucket"],
                resources=[model_bucket.bucket_arn],
            )
        )

        # SageMaker endpoint creation needs s3:PutObject for model artifacts
        # (repacking) and s3:GetObject for the model bucket root
        sagemaker_role.add_to_policy(
            iam.PolicyStatement(
                sid="S3ModelBucketWrite",
                actions=["s3:PutObject", "s3:GetObject", "s3:HeadObject"],
                resources=[
                    model_bucket.arn_for_objects(f"{s3_prefix}/*"),
                ],
            )
        )

        # S3 permissions — default SageMaker session bucket
        # The SageMaker Python SDK automatically repacks model.tar.gz into
        # s3://sagemaker-{region}-{account}/pytorch-inference-*/model.tar.gz
        # The execution role needs read access to this repacked artifact.
        default_sm_bucket_arn = f"arn:aws:s3:::sagemaker-{self.region}-{self.account}"

        sagemaker_role.add_to_policy(
            iam.PolicyStatement(
                sid="S3DefaultBucketReadWrite",
                actions=["s3:GetObject", "s3:HeadObject", "s3:PutObject"],
                resources=[
                    f"{default_sm_bucket_arn}/pytorch-inference-*",
                ],
            )
        )

        sagemaker_role.add_to_policy(
            iam.PolicyStatement(
                sid="S3DefaultBucketList",
                actions=["s3:ListBucket"],
                resources=[default_sm_bucket_arn],
            )
        )

        # ECR permissions — SageMaker needs to pull the PyTorch DLC container
        sagemaker_role.add_to_policy(
            iam.PolicyStatement(
                sid="ECRPullContainer",
                actions=[
                    "ecr:GetDownloadUrlForLayer",
                    "ecr:BatchGetImage",
                    "ecr:BatchCheckLayerAvailability",
                ],
                resources=[
                    f"arn:aws:ecr:{self.region}:763104351884:repository/*",
                ],
            )
        )

        sagemaker_role.add_to_policy(
            iam.PolicyStatement(
                sid="ECRAuth",
                actions=["ecr:GetAuthorizationToken"],
                resources=["*"],
            )
        )

        # CloudWatch Logs — for endpoint container logs
        sagemaker_role.add_to_policy(
            iam.PolicyStatement(
                sid="CloudWatchLogs",
                actions=[
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                    "logs:DescribeLogStreams",
                ],
                resources=[
                    f"arn:aws:logs:{self.region}:{self.account}:log-group:/aws/sagemaker/Endpoints/earth2-*",
                    f"arn:aws:logs:{self.region}:{self.account}:log-group:/aws/sagemaker/Endpoints/earth2-*:*",
                ],
            )
        )

        # CloudWatch Metrics — for endpoint monitoring
        sagemaker_role.add_to_policy(
            iam.PolicyStatement(
                sid="CloudWatchMetrics",
                actions=["cloudwatch:PutMetricData"],
                resources=["*"],
                conditions={
                    "StringEquals": {
                        "cloudwatch:namespace": [
                            "/aws/sagemaker/Endpoints",
                            "aws/sagemaker",
                        ],
                    },
                },
            )
        )

        # Suppress CDK Nag for necessary wildcards
        NagSuppressions.add_resource_suppressions(
            sagemaker_role,
            [
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "ecr:GetAuthorizationToken requires Resource::* — "
                    "it does not support resource-level permissions.",
                    "appliesTo": ["Resource::*"],
                },
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "SageMaker needs to pull PyTorch DLC images from AWS-managed ECR repo "
                    "(763104351884). Wildcard on repo name needed as image tags vary by version.",
                    "appliesTo": [
                        "Resource::arn:aws:ecr:<AWS::Region>:763104351884:repository/*",
                    ],
                },
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "Default SageMaker session bucket: the SageMaker Python SDK repacks "
                    "model.tar.gz into s3://sagemaker-{region}-{account}/pytorch-inference-*/. "
                    "The execution role needs read access to the repacked artifact.",
                    "appliesTo": [
                        "Resource::arn:aws:s3:::sagemaker-<AWS::Region>-<AWS::AccountId>/pytorch-inference-*",
                    ],
                },
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "S3 model artifacts wildcard: each model stores model.tar.gz under "
                    "a different model name prefix (e.g., dlwp/, fcn3/).",
                    "appliesTo": [
                        f"Resource::<{self.get_logical_id(model_bucket.node.default_child)}.Arn>/{s3_prefix}/*/model.tar.gz",
                    ],
                },
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "S3 async-input wildcard: each request creates a unique UUID key.",
                    "appliesTo": [
                        f"Resource::<{self.get_logical_id(model_bucket.node.default_child)}.Arn>/{s3_prefix}/async-input/*",
                    ],
                },
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "S3 async-output wildcard: SageMaker writes output with unique key.",
                    "appliesTo": [
                        f"Resource::<{self.get_logical_id(model_bucket.node.default_child)}.Arn>/{s3_prefix}/async-output/*",
                    ],
                },
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "S3 model bucket read/write: SageMaker needs PutObject/GetObject on the "
                    "model artifacts prefix for async inference I/O and model repacking.",
                    "appliesTo": [
                        f"Resource::<{self.get_logical_id(model_bucket.node.default_child)}.Arn>/{s3_prefix}/*",
                    ],
                },
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "CloudWatch Logs wildcard: endpoint log streams are dynamically named.",
                    "appliesTo": [
                        "Resource::arn:aws:logs:<AWS::Region>:<AWS::AccountId>:log-group:/aws/sagemaker/Endpoints/earth2-*",
                        "Resource::arn:aws:logs:<AWS::Region>:<AWS::AccountId>:log-group:/aws/sagemaker/Endpoints/earth2-*:*",
                    ],
                },
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "cloudwatch:PutMetricData requires Resource::* but is scoped by "
                    "condition to SageMaker namespaces only.",
                    "appliesTo": ["Resource::*"],
                },
            ],
            apply_to_children=True,
        )

        # ---------------------------------------------------------------
        # 4. ECR Repository for FCN3 BYOC container
        # ---------------------------------------------------------------
        fcn3_ecr_repo = ecr.Repository(
            self,
            "FCN3ContainerRepo",
            repository_name="earth2-fcn3",
            removal_policy=RemovalPolicy.DESTROY,
            empty_on_delete=True,
            lifecycle_rules=[
                ecr.LifecycleRule(
                    description="Keep only last 3 images",
                    max_image_count=3,
                ),
            ],
        )

        # Grant SageMaker role to pull from our ECR repo
        fcn3_ecr_repo.grant_pull(sagemaker_role)

        # ---------------------------------------------------------------
        # 5. CodeBuild project to build FCN3 container (GPU required)
        # ---------------------------------------------------------------
        # Source: S3 bucket (same bucket used for model artifacts).
        # Upload source with:
        #   cd earth2studio-on-aws && zip -r /tmp/source.zip . -x '.git/*'
        #   aws s3 cp /tmp/source.zip s3://<bucket>/codebuild/fcn3-source.zip
        fcn3_build = codebuild.Project(
            self,
            "FCN3ContainerBuild",
            project_name="earth2-fcn3-container-build",
            description="Builds the FCN3 BYOC container with CUDA-compiled torch-harmonics",
            source=codebuild.Source.s3(
                bucket=model_bucket,
                path="codebuild/fcn3-source.zip",
            ),
            build_spec=codebuild.BuildSpec.from_source_filename(
                "sagemaker_deploy/container_fcn3/buildspec.yml"
            ),
            environment=codebuild.BuildEnvironment(
                build_image=codebuild.LinuxBuildImage.STANDARD_7_0,
                compute_type=codebuild.ComputeType.LARGE,
                privileged=True,  # Required for docker build
            ),
            timeout=Duration.minutes(60),
        )

        # Grant CodeBuild permissions to push to ECR
        fcn3_ecr_repo.grant_pull_push(fcn3_build)

        # Grant CodeBuild permission to pull SageMaker DLC base image
        fcn3_build.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "ecr:GetDownloadUrlForLayer",
                    "ecr:BatchGetImage",
                    "ecr:BatchCheckLayerAvailability",
                    "ecr:GetAuthorizationToken",
                ],
                resources=["*"],
            )
        )

        # Grant CodeBuild permission to read NGC API key from Secrets Manager
        # The buildspec.yml references this secret for `docker login nvcr.io`
        ngc_secret = secretsmanager.Secret.from_secret_name_v2(
            self, "NgcApiKeySecret", "/earth2/ngc-api-key"
        )
        ngc_secret.grant_read(fcn3_build)

        # Suppress CDK Nag for CodeBuild
        NagSuppressions.add_resource_suppressions(
            fcn3_build,
            [
                {
                    "id": "AwsSolutions-CB3",
                    "reason": "Privileged mode is required for docker-in-docker builds.",
                },
                {
                    "id": "AwsSolutions-CB4",
                    "reason": "KMS encryption not needed for container build artifacts.",
                },
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "ecr:GetAuthorizationToken requires Resource::* — "
                    "cannot be scoped to specific repositories.",
                    "appliesTo": ["Resource::*"],
                },
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "S3 source: CDK auto-generates s3:GetObject*, s3:GetBucket*, "
                    "s3:List* for the CodeBuild source bucket.",
                    "appliesTo": [
                        "Action::s3:GetObject*",
                        "Action::s3:GetBucket*",
                        "Action::s3:List*",
                    ],
                },
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "CodeBuild log group wildcard — auto-generated by CDK.",
                    "appliesTo": [
                        "Resource::arn:<AWS::Partition>:logs:<AWS::Region>:<AWS::AccountId>:log-group:/aws/codebuild/<FCN3ContainerBuild8C64AA98>:*",
                    ],
                },
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "CodeBuild report group wildcard — auto-generated by CDK.",
                    "appliesTo": [
                        "Resource::arn:<AWS::Partition>:codebuild:<AWS::Region>:<AWS::AccountId>:report-group/<FCN3ContainerBuild8C64AA98>-*",
                    ],
                },
            ],
            apply_to_children=True,
        )

        # ---------------------------------------------------------------
        # 6. SSM Parameters — discoverable by deploy scripts + Lambda
        # ---------------------------------------------------------------
        ssm_prefix = "/earth2/sagemaker"

        ssm.StringParameter(
            self,
            "ParamBucketName",
            parameter_name=f"{ssm_prefix}/bucket-name",
            string_value=model_bucket.bucket_name,
            description="S3 bucket for Earth2Studio model artifacts and async I/O",
        )

        ssm.StringParameter(
            self,
            "ParamS3Prefix",
            parameter_name=f"{ssm_prefix}/s3-prefix",
            string_value=s3_prefix,
            description="S3 key prefix for Earth2Studio resources",
        )

        ssm.StringParameter(
            self,
            "ParamRoleArn",
            parameter_name=f"{ssm_prefix}/role-arn",
            string_value=sagemaker_role.role_arn,
            description="SageMaker execution role ARN for Earth2Studio endpoints",
        )

        ssm.StringParameter(
            self,
            "ParamRegion",
            parameter_name=f"{ssm_prefix}/region",
            string_value=self.region,
            description="AWS region for Earth2Studio SageMaker resources",
        )

        # ---------------------------------------------------------------
        # Expose as stack properties for cross-stack references
        # ---------------------------------------------------------------
        self.model_bucket = model_bucket
        self.sagemaker_role = sagemaker_role
        self.s3_prefix = s3_prefix

        # ---------------------------------------------------------------
        # Outputs
        # ---------------------------------------------------------------
        CfnOutput(
            self,
            "ModelBucketName",
            value=model_bucket.bucket_name,
            description="S3 bucket name for model artifacts",
        )
        CfnOutput(
            self,
            "SageMakerRoleArn",
            value=sagemaker_role.role_arn,
            description="SageMaker execution role ARN",
        )
        CfnOutput(
            self,
            "SSMPrefix",
            value=ssm_prefix,
            description="SSM Parameter Store prefix for all Earth2 SageMaker config",
        )
