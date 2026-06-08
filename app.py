#!/usr/bin/env python3
"""CDK app entry point for Earth2Studio.

Stacks:
  1. Earth2SageMaker — Long-lived SageMaker infrastructure (IAM role, S3 bucket, SSM)
  2. Earth2UI         — Frontend + Backend (CloudFront, S3, API GW, Lambda, Cognito, WAF)

The UI stack receives the model bucket from the SageMaker stack so Lambda
can read/write async inference I/O without hardcoded bucket names.
"""

import aws_cdk as cdk
from stacks.sagemaker_infra_stack import SageMakerInfraStack
from stacks.ui_stack import UIStack

app = cdk.App()

# Apply common tags to all stacks/resources synthesized by this app.
cdk.Tags.of(app).add("auto-delete", "no")

# Stack 1: Long-lived SageMaker supporting infrastructure
sagemaker_infra = SageMakerInfraStack(
    app,
    "Earth2SageMaker",
)

# Stack 2: UI + Backend (depends on SageMaker infra for bucket name)
UIStack(
    app,
    "Earth2UI",
    model_bucket=sagemaker_infra.model_bucket,
    s3_prefix=sagemaker_infra.s3_prefix,
)

app.synth()
