#!/usr/bin/env python3
"""
Deploy NVIDIA Earth-2 Weather Models to Amazon SageMaker.

This script:
1. Packages the inference code and model config into model.tar.gz
2. Uploads the artifact to S3
3. Creates a SageMaker Model using the PyTorch DLC container
4. Deploys the model to a SageMaker Endpoint

Usage:
    # Deploy DLWP (default)
    python deploy.py --model dlwp --instance-type ml.g5.2xlarge

    # Deploy FourCastNet v3 (FCN3) — picks the BYOC NGC variant by default
    python deploy.py --model fcn3
    python deploy.py --model fcn3 --container-variant dlc

    # Deploy with custom endpoint name
    python deploy.py --model dlwp --endpoint-name my-weather-endpoint

    # Delete an endpoint
    python deploy.py --delete --endpoint-name earth2-dlwp-endpoint

Only Apache-2.0-licensed models are shipped (DLWP, FCN3); other earth2studio
classes are intentionally not registered in config.py to keep the sample
publishable under aws-samples.

Requirements:
    pip install boto3 sagemaker
    aws configure  (must have SageMaker permissions)
"""

import argparse
import json
import logging
import os
import shutil
import sys
import tarfile
import tempfile
from datetime import datetime

import boto3
import sagemaker
from sagemaker.model import Model
from sagemaker.pytorch import PyTorchModel
from sagemaker.async_inference import AsyncInferenceConfig
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

# Add parent directory to path for config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    AWS_REGION,
    S3_BUCKET,
    S3_PREFIX,
    SAGEMAKER_ROLE,
    SUPPORTED_MODELS,
    PYTORCH_FRAMEWORK_VERSION,
    PYTHON_VERSION,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ACTIVE_ENDPOINT_FILE = os.path.join(SCRIPT_DIR, "active_endpoint.json")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Deploy NVIDIA Earth-2 weather models on Amazon SageMaker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python deploy.py --model dlwp
  python deploy.py --model fcn3 --container-variant dlc
  python deploy.py --model fcn3 --endpoint-name my-fcn3-endpoint
  python deploy.py --delete --endpoint-name earth2-dlwp-endpoint
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="dlwp",
        choices=list(SUPPORTED_MODELS.keys()),
        help="Weather model to deploy (default: dlwp)",
    )
    parser.add_argument(
        "--instance-type",
        type=str,
        default=None,
        help="SageMaker instance type (default: model-specific, e.g., ml.g5.2xlarge)",
    )
    parser.add_argument(
        "--instance-count",
        type=int,
        default=1,
        help="Number of instances (default: 1)",
    )
    parser.add_argument(
        "--endpoint-name",
        type=str,
        default=None,
        help="Custom endpoint name (default: auto-generated from model name)",
    )
    parser.add_argument(
        "--s3-bucket",
        type=str,
        default=None,
        help=f"S3 bucket for model artifacts (default: {S3_BUCKET})",
    )
    parser.add_argument(
        "--role",
        type=str,
        default=None,
        help="SageMaker execution role ARN (default: from config.py or auto-detect)",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=AWS_REGION,
        help=f"AWS region (default: {AWS_REGION})",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete an existing endpoint instead of creating one",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Don't wait for endpoint — return immediately after creation",
    )
    parser.add_argument(
        "--container-variant",
        type=str,
        default=None,
        choices=["ngc", "dlc"],
        help="For BYOC models (e.g., fcn3): choose 'ngc' (NVIDIA NGC, default) or "
             "'dlc' (AWS Training DLC, no NGC auth needed). Ignored for non-BYOC models.",
    )

    return parser.parse_args()


def get_sagemaker_role(args):
    """Get the SageMaker execution role."""
    if args.role:
        return args.role

    if SAGEMAKER_ROLE and "YOUR_ACCOUNT_ID" not in SAGEMAKER_ROLE:
        return SAGEMAKER_ROLE

    # Try to auto-detect from SageMaker SDK (works inside SageMaker notebooks).
    # Outside a SageMaker notebook this is expected to fail; we then fall through
    # to the IAM-based discovery below.
    try:
        role = sagemaker.get_execution_role()
        logger.info(f"Auto-detected SageMaker role: {role}")
        return role
    except Exception as e:
        logger.debug(f"sagemaker.get_execution_role() unavailable: {e}")

    # Try to find a SageMaker role via IAM
    try:
        iam = boto3.client("iam", region_name=args.region)
        paginator = iam.get_paginator("list_roles")
        for page in paginator.paginate():
            for r in page["Roles"]:
                if "sagemaker" in r["RoleName"].lower():
                    logger.info(f"Found SageMaker role: {r['Arn']}")
                    return r["Arn"]
    except Exception as e:
        logger.warning(f"Could not list IAM roles: {e}")

    logger.error(
        "Could not determine SageMaker execution role. Please either:\n"
        "  1. Set SAGEMAKER_ROLE in config.py\n"
        "  2. Pass --role arn:aws:iam::ACCOUNT:role/YourRole\n"
        "  3. Run from a SageMaker notebook (auto-detects role)\n"
    )
    sys.exit(1)


def get_s3_bucket(args):
    """Get the S3 bucket to use."""
    bucket = args.s3_bucket or S3_BUCKET
    if bucket == "your-sagemaker-bucket":
        # Try to use the default SageMaker bucket. If this fails (no default
        # bucket configured for this account/region), we fall through to the
        # explicit error message below.
        try:
            sess = sagemaker.Session(boto_session=boto3.Session(region_name=args.region))
            bucket = sess.default_bucket()
            logger.info(f"Using default SageMaker bucket: {bucket}")
            return bucket
        except Exception as e:
            logger.debug(f"Default SageMaker bucket unavailable: {e}")

        logger.error(
            "No S3 bucket configured. Please either:\n"
            "  1. Set S3_BUCKET in config.py\n"
            "  2. Pass --s3-bucket your-bucket-name\n"
        )
        sys.exit(1)
    return bucket


def package_model_artifacts(model_name):
    """
    Package the inference code and model configuration into model.tar.gz.

    SageMaker PyTorch DLC expects model.tar.gz with this structure when
    using the `model_data` parameter (without source_dir):

        model.tar.gz/
        ├── code/                     # Required: SageMaker looks for code/ dir
        │   ├── inference.py          # Entry point (model_fn, input_fn, etc.)
        │   └── requirements.txt      # Extra pip dependencies for the container
        └── model_config.json         # Our custom config for the weather model

    When SageMaker extracts model.tar.gz into model_dir, it:
    1. Finds code/ directory
    2. Installs code/requirements.txt via pip
    3. Loads code/inference.py as the inference handler
    4. Calls model_fn(model_dir) where model_dir is the extraction root

    Reference:
    https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#serve-a-pytorch-model
    """
    logger.info(f"Packaging model artifacts for: {model_name}")

    model_config = SUPPORTED_MODELS[model_name]
    # Use the model-specific code directory (model_code_dlwp or model_code_fcn3).
    code_dir_name = model_config["model_code_dir"]
    model_code_dir = os.path.join(SCRIPT_DIR, code_dir_name)

    # Create temp directory for packaging.
    #
    # SECURITY NOTE (B108 SAST finding — false positive, justification):
    #   `tempfile.mkdtemp()` is the secure CPython API for creating a
    #   private temp directory. It:
    #     - creates the directory atomically (no TOCTOU race),
    #     - sets mode 0o700 so only the current user can read/write,
    #     - uses the OS-default temp location (TMPDIR / /tmp / etc.) —
    #       it does NOT hardcode "/tmp".
    #   Bandit's B108 rule pattern-matches any string mentioning a temp
    #   path and cannot tell that mkdtemp() already mitigates the risk.
    #   The `# nosec B108` marker below silences that single line.
    #   Reference: https://docs.python.org/3/library/tempfile.html#tempfile.mkdtemp
    tmp_dir = tempfile.mkdtemp(prefix="earth2_sagemaker_")  # nosec B108

    try:
        # Create code/ directory (SageMaker PyTorch DLC convention)
        code_dir = os.path.join(tmp_dir, "code")
        os.makedirs(code_dir, exist_ok=True)

        # Copy inference handler
        shutil.copy2(
            os.path.join(model_code_dir, "inference.py"),
            os.path.join(code_dir, "inference.py"),
        )

        # Copy container requirements
        shutil.copy2(
            os.path.join(model_code_dir, "requirements.txt"),
            os.path.join(code_dir, "requirements.txt"),
        )

        # Write model configuration at the root of the tarball
        config_data = {
            "model_class": model_config["model_class"],
            "import_path": model_config["import_path"],
            "variables": model_config["variables"],
            "resolution_deg": model_config["resolution_deg"],
            "grid_size": list(model_config["grid_size"]),
            "time_step_hours": model_config["time_step_hours"],
            "deployed_at": datetime.utcnow().isoformat() + "Z",
        }
        config_json_path = os.path.join(tmp_dir, "model_config.json")
        with open(config_json_path, "w") as f:
            json.dump(config_data, f, indent=2)

        # Create model.tar.gz
        tar_path = os.path.join(tmp_dir, "model.tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            # Add code/ directory (preserving directory structure)
            tar.add(code_dir, arcname="code")
            # Add model_config.json at root
            tar.add(config_json_path, arcname="model_config.json")

        tar_size = os.path.getsize(tar_path) / 1024
        logger.info(f"Created model.tar.gz ({tar_size:.1f} KB)")

        # Log contents for verification
        with tarfile.open(tar_path, "r:gz") as tar:
            logger.info("model.tar.gz contents:")
            for member in tar.getmembers():
                logger.info(f"  {member.name} ({member.size} bytes)")

        return tar_path, tmp_dir

    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise e


def upload_to_s3(tar_path, bucket, model_name, region):
    """Upload model.tar.gz to S3."""
    s3_key = f"{S3_PREFIX}/{model_name}/model.tar.gz"
    s3_uri = f"s3://{bucket}/{s3_key}"

    logger.info(f"Uploading model artifacts to {s3_uri}")

    s3_client = boto3.client("s3", region_name=region)
    s3_client.upload_file(tar_path, bucket, s3_key)

    logger.info("Upload complete")
    return s3_uri


def deploy_model(args):
    """Deploy the weather model to a SageMaker endpoint."""

    model_name = args.model
    model_config = SUPPORTED_MODELS[model_name]
    instance_type = args.instance_type or model_config["default_instance"]
    endpoint_name = args.endpoint_name or f"{model_config['endpoint_prefix']}-endpoint"

    logger.info("=" * 60)
    logger.info(f"  Deploying: {model_config['name']}")
    logger.info(f"  Model class: {model_config['import_path']}.{model_config['model_class']}")
    logger.info(f"  Instance:  {instance_type} × {args.instance_count}")
    logger.info(f"  Endpoint:  {endpoint_name}")
    logger.info(f"  Region:    {args.region}")
    logger.info("=" * 60)

    # Step 1: Get role and bucket
    role = get_sagemaker_role(args)
    bucket = get_s3_bucket(args)
    logger.info(f"Role: {role}")
    logger.info(f"S3 Bucket: {bucket}")

    # Step 2: Package model artifacts
    tar_path, tmp_dir = package_model_artifacts(model_name)

    try:
        # Step 3: Upload to S3
        model_data_uri = upload_to_s3(tar_path, bucket, model_name, args.region)

        # Step 4: Create SageMaker PyTorch model
        #
        # Important: We use model_data (with code/ inside the tarball) and
        # do NOT set source_dir. This avoids double-packaging the code.
        # SageMaker auto-detects the code/ directory inside model.tar.gz.
        #
        # Reference: https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html
        boto_session = boto3.Session(region_name=args.region)
        sagemaker_session = sagemaker.Session(boto_session=boto_session)

        # Common environment variables for all models. The inference handler
        # uses S3_BUCKET / AWS_REGION to upload full-resolution forecast data.
        from config import S3_BUCKET, AWS_REGION  # type: ignore[attr-defined]

        model_env = {
            "EARTH2_MODEL_CLASS": model_config["model_class"],
            "EARTH2_CACHE": "/tmp/earth2_cache",
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
            "SAGEMAKER_MODEL_SERVER_TIMEOUT": "600",
            "SAGEMAKER_TS_RESPONSE_TIMEOUT": "600",
            "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
            "S3_BUCKET": S3_BUCKET,
            "AWS_REGION": AWS_REGION,
        }

        # Resolve container URI — support variant selection for BYOC models.
        # Config stores ECR repo names only; we build the full URI from
        # the caller's account + region at deploy time.
        from config import _ecr_uri  # type: ignore[attr-defined]

        container_repos = model_config.get("container_repos", {})
        default_repo = model_config.get("container_repo_default")
        variant = getattr(args, "container_variant", None)

        repo: str | None = None
        if variant and container_repos:
            repo = container_repos.get(variant)
            if not repo:
                raise ValueError(
                    f"Unknown container variant '{variant}'. "
                    f"Available: {list(container_repos)}"
                )
            logger.info(f"Using BYOC container variant '{variant}' (repo: {repo})")
        elif default_repo:
            repo = default_repo
            logger.info(f"Using default BYOC container (repo: {repo})")

        container_uri = _ecr_uri(repo) if repo else None

        if container_uri:
            # BYOC: Use custom container (e.g., FCN3 with CUDA-compiled torch-harmonics)
            logger.info(f"Container image: {container_uri}")
            sagemaker_model = Model(
                model_data=model_data_uri,
                image_uri=container_uri,
                role=role,
                sagemaker_session=sagemaker_session,
                env=model_env,
            )
        else:
            # Standard: Use SageMaker PyTorch DLC
            logger.info("Using SageMaker PyTorch DLC container...")
            sagemaker_model = PyTorchModel(
                model_data=model_data_uri,
                role=role,
                framework_version=PYTORCH_FRAMEWORK_VERSION,
                py_version=PYTHON_VERSION,
                entry_point="inference.py",
                sagemaker_session=sagemaker_session,
                env=model_env,
            )

        # Step 5: Deploy to endpoint
        logger.info(f"Deploying to endpoint: {endpoint_name}")
        logger.info("This may take 10-20 minutes:")
        logger.info("  - Container image pull (~2-3 min)")
        logger.info("  - pip install earth2studio + deps (~3-5 min)")
        logger.info("  - Model weight download from NVIDIA registry (~2-5 min)")
        logger.info("  - GPU initialization (~1-2 min)")

        wait = not args.no_wait

        # Use Async Inference — real-time endpoints have a hard 60-second
        # gateway timeout which is too short for weather models that need
        # to download GFS data + run GPU inference (typically 2-5 minutes).
        # Async inference supports up to 1 hour per request.
        async_output_s3 = f"s3://{bucket}/{S3_PREFIX}/async-output/"
        logger.info(f"Using Async Inference (output → {async_output_s3})")

        async_config = AsyncInferenceConfig(
            output_path=async_output_s3,
            max_concurrent_invocations_per_instance=1,  # Weather models use full GPU
        )

        predictor = sagemaker_model.deploy(
            initial_instance_count=args.instance_count,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
            async_inference_config=async_config,
            serializer=JSONSerializer(),
            deserializer=JSONDeserializer(),
            wait=wait,
        )

        # Save active endpoint info to JSON for other tools (invoke)
        active_info = {
            "endpoint_name": endpoint_name,
            "model": model_name,
            "instance_type": instance_type,
            "deployed_at": datetime.utcnow().isoformat() + "Z",
            "region": args.region,
        }
        try:
            with open(ACTIVE_ENDPOINT_FILE, "w") as f:
                json.dump(active_info, f, indent=2)
            logger.info(f"Saved active endpoint info to {ACTIVE_ENDPOINT_FILE}")
        except Exception as e:
            logger.warning(f"Could not save active_endpoint.json: {e}")

        if wait:
            logger.info("")
            logger.info("=" * 60)
            logger.info("  ✅ DEPLOYMENT SUCCESSFUL!")
            logger.info(f"  Endpoint Name: {endpoint_name}")
            logger.info("")
            logger.info("  To run inference:")
            logger.info(f"    python invoke_endpoint.py --endpoint-name {endpoint_name}")
            logger.info("")
            logger.info("  To delete when done (saves costs!):")
            logger.info(f"    python deploy.py --delete --endpoint-name {endpoint_name}")
            logger.info("=" * 60)
        else:
            logger.info(f"Endpoint creation initiated: {endpoint_name}")
            logger.info("Check status with:")
            logger.info(f"  aws sagemaker describe-endpoint --endpoint-name {endpoint_name}")

        return predictor

    finally:
        # Clean up temp directory
        shutil.rmtree(tmp_dir, ignore_errors=True)


def delete_endpoint(args):
    """Delete an existing SageMaker endpoint, its config, and its model."""
    endpoint_name = args.endpoint_name
    if not endpoint_name:
        logger.error("--endpoint-name is required with --delete")
        sys.exit(1)

    logger.info(f"Deleting endpoint: {endpoint_name}")
    sm_client = boto3.client("sagemaker", region_name=args.region)

    try:
        # Get endpoint config and model names for cleanup
        endpoint_info = sm_client.describe_endpoint(EndpointName=endpoint_name)
        config_name = endpoint_info["EndpointConfigName"]

        # Get model name from endpoint config
        config_info = sm_client.describe_endpoint_config(EndpointConfigName=config_name)
        model_names = [
            variant["ModelName"]
            for variant in config_info.get("ProductionVariants", [])
        ]

        # Delete endpoint
        logger.info("Deleting endpoint...")
        sm_client.delete_endpoint(EndpointName=endpoint_name)

        # Wait for deletion
        logger.info("Waiting for endpoint deletion (this takes ~2-5 min)...")
        waiter = sm_client.get_waiter("endpoint_deleted")
        waiter.wait(
            EndpointName=endpoint_name,
            WaiterConfig={"Delay": 15, "MaxAttempts": 40},
        )

        # Delete endpoint config
        try:
            logger.info(f"Deleting endpoint config: {config_name}")
            sm_client.delete_endpoint_config(EndpointConfigName=config_name)
        except Exception as e:
            logger.warning(f"Could not delete endpoint config: {e}")

        # Delete model(s)
        for model_name in model_names:
            try:
                logger.info(f"Deleting model: {model_name}")
                sm_client.delete_model(ModelName=model_name)
            except Exception as e:
                logger.warning(f"Could not delete model {model_name}: {e}")

        logger.info(f"✅ Endpoint '{endpoint_name}' and associated resources deleted.")

        # Clear active_endpoint.json if it matches the deleted endpoint
        try:
            if os.path.exists(ACTIVE_ENDPOINT_FILE):
                with open(ACTIVE_ENDPOINT_FILE, "r") as f:
                    active_info = json.load(f)
                if active_info.get("endpoint_name") == endpoint_name:
                    os.remove(ACTIVE_ENDPOINT_FILE)
                    logger.info(f"Removed {ACTIVE_ENDPOINT_FILE} (endpoint deleted)")
        except Exception as e:
            logger.warning(f"Could not update active_endpoint.json: {e}")

    except sm_client.exceptions.ClientError as e:
        if "Could not find endpoint" in str(e):
            logger.error(f"Endpoint '{endpoint_name}' not found.")
        else:
            raise


def main():
    args = parse_args()

    if args.delete:
        delete_endpoint(args)
    else:
        deploy_model(args)


if __name__ == "__main__":
    main()
