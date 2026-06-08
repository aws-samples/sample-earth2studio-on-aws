"""
Configuration constants for NVIDIA Earth-2 SageMaker deployment.

Resource discovery:
  The S3 bucket, IAM role, and region are provisioned by CDK (Earth2SageMaker
  stack) and published to SSM Parameter Store under /earth2/sagemaker/*.
  This config auto-discovers them via SSM. Environment variables take
  precedence for local override / CI.

Priority order for each setting:
  1. Environment variable (e.g., EARTH2_S3_BUCKET)
  2. SSM Parameter Store (populated by CDK Earth2SageMaker stack)

If neither is available the module raises at import time, since deploy
operations cannot proceed without these values.
"""

import logging
import os

logger = logging.getLogger(__name__)

# ============================================================
# SSM Parameter Store Discovery
# ============================================================
SSM_PREFIX = "/earth2/sagemaker"


def _get_ssm_param(name: str) -> str | None:
    """Read an SSM parameter; return None on any failure."""
    try:
        import boto3

        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        ssm = boto3.client("ssm", region_name=region)
        resp = ssm.get_parameter(Name=f"{SSM_PREFIX}/{name}")
        value = resp["Parameter"]["Value"]
        logger.debug(f"SSM {SSM_PREFIX}/{name} = {value}")
        return value
    except Exception:
        return None


def _require(env_var: str, ssm_name: str, description: str) -> str:
    """Resolve a required value from env var or SSM; raise if neither is set."""
    value = os.environ.get(env_var) or _get_ssm_param(ssm_name)
    if not value:
        raise RuntimeError(
            f"{description} is not configured. Set the {env_var} environment "
            f"variable or deploy the Earth2SageMaker CDK stack to populate "
            f"{SSM_PREFIX}/{ssm_name}."
        )
    return value


# ============================================================
# AWS Configuration
# ============================================================
AWS_REGION = (
    os.environ.get("EARTH2_REGION")
    or os.environ.get("AWS_REGION")
    or os.environ.get("AWS_DEFAULT_REGION")
    or _get_ssm_param("region")
)
if not AWS_REGION:
    raise RuntimeError(
        "AWS region is not configured. Set AWS_REGION, AWS_DEFAULT_REGION, "
        f"or EARTH2_REGION, or deploy the Earth2SageMaker CDK stack to populate {SSM_PREFIX}/region."
    )

# S3 bucket where model artifacts will be stored
# CDK creates this bucket in SageMakerInfraStack and publishes the name to SSM
S3_BUCKET = _require("EARTH2_S3_BUCKET", "bucket-name", "S3 bucket for model artifacts")
S3_PREFIX = (
    os.environ.get("EARTH2_S3_PREFIX")
    or _get_ssm_param("s3-prefix")
    or "earth2-weather-models"
)

# SageMaker execution role ARN
# CDK creates this role in SageMakerInfraStack and publishes the ARN to SSM
SAGEMAKER_ROLE = _require(
    "EARTH2_SAGEMAKER_ROLE", "role-arn", "SageMaker execution role ARN"
)


# AWS account ID — derived from caller identity for ECR URIs.
# Lazily resolved so importing this module doesn't require a live STS call.
_AWS_ACCOUNT_ID: str | None = None


def _get_account_id() -> str:
    """Return the current AWS account ID (cached after first call)."""
    global _AWS_ACCOUNT_ID
    if _AWS_ACCOUNT_ID:
        return _AWS_ACCOUNT_ID
    explicit = os.environ.get("AWS_ACCOUNT_ID") or os.environ.get("CDK_DEFAULT_ACCOUNT")
    if explicit:
        _AWS_ACCOUNT_ID = explicit
        return _AWS_ACCOUNT_ID
    try:
        import boto3

        _AWS_ACCOUNT_ID = boto3.client("sts").get_caller_identity()["Account"]
        return _AWS_ACCOUNT_ID
    except Exception as e:
        raise RuntimeError(
            "Could not determine AWS account ID. Set AWS_ACCOUNT_ID, "
            "CDK_DEFAULT_ACCOUNT, or configure AWS credentials."
        ) from e


def _ecr_uri(repo: str, tag: str = "latest") -> str:
    """Build the ECR URI for a repository in the current account/region."""
    return f"{_get_account_id()}.dkr.ecr.{AWS_REGION}.amazonaws.com/{repo}:{tag}"

# ============================================================
# Model Configuration
# ============================================================
# This sample only ships permissively licensed models (Apache-2.0). Other
# weather models exposed by earth2studio (Pangu, GraphCast, FuXi, etc.) carry
# non-commercial / research-only licenses and are intentionally omitted to
# keep this repository safe to publish under aws-samples.
#
# Class names and variables are verified against the earth2studio source:
# https://github.com/NVIDIA/earth2studio/blob/main/earth2studio/models/px/__init__.py

SUPPORTED_MODELS = {
    "dlwp": {
        "name": "DLWP (Deep Learning Weather Prediction)",
        # License: Apache-2.0 (University of Washington, via NVIDIA Earth2Studio)
        "description": "UW cubed-sphere CNN weather model",
        "model_class": "DLWP",
        "import_path": "earth2studio.models.px",
        "default_instance": "ml.g5.2xlarge",
        "endpoint_prefix": "earth2-dlwp",
        "model_code_dir": "model_code_dlwp",
        "gpu_memory_gb": 4,
        "variables": [],  # Populated dynamically
        "resolution_deg": 1.0,
        "grid_size": (180, 360),
        "time_step_hours": 6,
    },
    "fcn3": {
        "name": "FourCastNet v3 (FCN3)",
        # License: Apache-2.0 (NVIDIA, via Earth2Studio)
        "description": "NVIDIA probabilistic ensemble weather model using spherical harmonics, "
        "72 variables, 0.25° resolution, requires 80GB GPU (gpu:80gb badge)",
        "model_class": "FCN3",
        "import_path": "earth2studio.models.px",
        "default_instance": "ml.g7e.2xlarge",  # RTX PRO Server 6000, 96GB VRAM
        "endpoint_prefix": "earth2-fcn3",
        "model_code_dir": "model_code_fcn3",
        # Two BYOC container variants — both verified working on Blackwell (g7e):
        #   ngc: NVIDIA NGC base image (compiled CUDA torch-harmonics, needs NGC API key)
        #   dlc: AWS Training DLC base image (PyPI torch-harmonics, no NGC auth needed)
        # ECR repos live in the caller's account; URIs are resolved at deploy time.
        "container_repos": {
            "ngc": "earth2-fcn3",
            "dlc": "earth2-fcn3-dlc",
        },
        "container_repo_default": "earth2-fcn3",  # default variant: NGC
        "gpu_memory_gb": 80,
        # Verified 72 variables from earth2studio/models/px/fcn3.py
        "variables": [
            "u10m", "v10m", "u100m", "v100m", "t2m", "msl", "tcwv",
            "u50", "u100", "u150", "u200", "u250", "u300", "u400",
            "u500", "u600", "u700", "u850", "u925", "u1000",
            "v50", "v100", "v150", "v200", "v250", "v300", "v400",
            "v500", "v600", "v700", "v850", "v925", "v1000",
            "z50", "z100", "z150", "z200", "z250", "z300", "z400",
            "z500", "z600", "z700", "z850", "z925", "z1000",
            "t50", "t100", "t150", "t200", "t250", "t300", "t400",
            "t500", "t600", "t700", "t850", "t925", "t1000",
            "q50", "q100", "q150", "q200", "q250", "q300", "q400",
            "q500", "q600", "q700", "q850", "q925", "q1000",
        ],
        "resolution_deg": 0.25,
        "grid_size": (721, 1440),  # lat x lon at 0.25 degree (south-pole including)
        "time_step_hours": 6,
    },
}

# ============================================================
# Inference Configuration
# ============================================================
DEFAULT_LEAD_TIME_HOURS = 24  # Default forecast length
MAX_LEAD_TIME_HOURS = 240     # 10 days maximum

# SageMaker endpoint timeout (seconds)
# For long ensemble forecasts, use Async inference instead
ENDPOINT_TIMEOUT_SECONDS = 120

# ============================================================
# PyTorch Deep Learning Container (DLC) Configuration
# ============================================================
# SageMaker pre-built PyTorch containers
# See: https://github.com/aws/deep-learning-containers/blob/master/available_images.md
# Must be >= 2.5.0 to match earth2studio's torch requirement.
# Using 2.3.0 causes earth2studio to upgrade torch, breaking the container's
# NCCL library (undefined symbol: ncclCommShrink).
PYTORCH_FRAMEWORK_VERSION = "2.6.0"
PYTHON_VERSION = "py312"
