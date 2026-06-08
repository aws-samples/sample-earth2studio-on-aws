"""Shared configuration for the Lambda backend.

These values are injected as environment variables by the Earth2UI CDK
stack (see stacks/ui_stack.py). For local development they must be set
explicitly (see backend/local_server.py).
"""

import os

AWS_REGION = os.environ.get("REGION") or os.environ.get("AWS_REGION")
S3_BUCKET = os.environ.get("S3_BUCKET")
S3_PREFIX = os.environ.get("S3_PREFIX", "earth2-weather-models")

if not AWS_REGION:
    raise RuntimeError("REGION (or AWS_REGION) environment variable is required.")
if not S3_BUCKET:
    raise RuntimeError("S3_BUCKET environment variable is required.")
