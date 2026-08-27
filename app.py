#!/usr/bin/env python3
"""CDK app entry point for Earth2Studio.

Stacks:
  1. Earth2SageMaker — Long-lived SageMaker infrastructure (IAM role, S3 bucket, SSM)
  2. Earth2UI         — Frontend + Backend (CloudFront, S3, API GW, Lambda, Cognito, WAF)

The UI stack receives the model bucket from the SageMaker stack so Lambda
can read/write async inference I/O without hardcoded bucket names.

Both stacks carry an AWS Solution identity in their CloudFormation
Description, which is how deployments are counted. Earth2UI consumes the
SageMaker stack, so it sits at the top of the dependency graph and is the one
stack that carries the bare solution id; Earth2SageMaker carries the
`-sagemaker` suffixed form, or one install would be counted twice. See
solution.py and README.md ('Operational metrics').
"""

import aws_cdk as cdk
from solution import stack_description
from stacks.sagemaker_infra_stack import SageMakerInfraStack
from stacks.ui_stack import UIStack

app = cdk.App()

# Apply common tags to all stacks/resources synthesized by this app.
cdk.Tags.of(app).add("auto-delete", "no")

# Stack 1: Long-lived SageMaker supporting infrastructure.
# Supporting stack: id_suffix keeps it out of the install count, since this
# foundation can outlive an abandoned deploy of the UI above it.
sagemaker_infra = SageMakerInfraStack(
    app,
    "Earth2SageMaker",
    description=stack_description(id_suffix="sagemaker"),
)

# Stack 2: UI + Backend (depends on SageMaker infra for bucket name).
# Top of the dependency graph: the one stack carrying the bare solution id.
UIStack(
    app,
    "Earth2UI",
    model_bucket=sagemaker_infra.model_bucket,
    s3_prefix=sagemaker_infra.s3_prefix,
    description=stack_description(),
)

app.synth()
