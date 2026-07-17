# SageMaker Endpoint Deployment

Deploy NVIDIA Earth2Studio weather models to Amazon SageMaker async inference endpoints.

## Overview

This module manages the **ephemeral** SageMaker GPU endpoints that run weather model inference. The supporting infrastructure (S3 bucket, IAM role) is provisioned by CDK in the `Earth2SageMaker` stack and auto-discovered via SSM Parameter Store.

## Models

This sample only ships **Apache-2.0-licensed** weather models. Other models exposed by `earth2studio` (Pangu, GraphCast, FuXi, etc.) carry research / non-commercial licenses and are intentionally not registered here.

| Model | Class | Instance | GPU | VRAM | Cost/hr | License |
|---|---|---|---|---|---|---|
| DLWP | `DLWP` | `ml.g5.2xlarge` | A10G | 24 GB | ~$1.52 | Apache-2.0 |
| **FCN3** | `FCN3` | `ml.g7e.2xlarge` | RTX PRO Server 6000 | 96 GB | ~$4.20 | Apache-2.0 |

**Total: ~$5.72/hr (~$137/day) when both endpoints are running.**

## Quick Start

```bash
# Deploy all models
./deploy-all.sh

# Check status
./deploy-all.sh --status

# Delete all endpoints (stop billing!)
./deploy-all.sh --delete
```

## Deploy a Single Model

```bash
# Deploy DLWP (standard PyTorch DLC)
python deploy.py --model dlwp --instance-type ml.g5.2xlarge

# Deploy FCN3 — NGC container (default, compiled CUDA torch-harmonics)
python deploy.py --model fcn3 --no-wait

# Deploy FCN3 — Training DLC container (no NGC API key needed)
python deploy.py --model fcn3 --container-variant dlc --no-wait

# Deploy FCN3 — explicitly choose NGC
python deploy.py --model fcn3 --container-variant ngc --no-wait

# Deploy with --no-wait (don't block for 10-20 min)
python deploy.py --model dlwp --no-wait

# Delete a specific endpoint
python deploy.py --delete --endpoint-name earth2-dlwp-endpoint
```

## FCN3 BYOC Container

FCN3 requires a **custom (BYOC) container** because:
1. PyTorch 2.6.0 (latest SageMaker inference DLC) doesn't support Blackwell GPU (sm_120)
2. FCN3 needs `torch-harmonics` + `makani` dependencies not in standard DLCs
3. AWS stopped releasing PyTorch inference DLCs ≥ 2.7 (TorchServe in maintenance mode)

### Two Container Variants

| | **NGC** (`container_fcn3/`) | **Training DLC** (`container_fcn3_dlc/`) |
|---|---|---|
| Base image | `nvcr.io/nvidia/pytorch:25.12-py3` | `pytorch-training:2.7.1-gpu-py312-cu128` |
| torch-harmonics | CUDA C++ compiled | PyPI wheel (float32 fallback) |
| NGC API key | ⚠️ Required (Secrets Manager) | ✅ Not needed |
| Image size | ~15-20 GB | ~10-12 GB |
| Deploy flag | `--container-variant ngc` (default) | `--container-variant dlc` |

Both use Flask/gunicorn on port 8080 (`/ping` + `/invocations`).

### Build Pipeline (NGC — via CodeBuild)

```bash
zip -rq /tmp/source.zip . -x '.git/*' 'node_modules/*' '.venv/*'
aws s3 cp /tmp/source.zip s3://<bucket>/codebuild/fcn3-source.zip
aws codebuild start-build --project-name earth2-fcn3-container-build --region us-west-2
```

### Deploy All with Variant

```bash
./deploy-all.sh                  # FCN3 uses NGC (default)
./deploy-all.sh --variant dlc    # FCN3 uses Training DLC
```

## How It Works

### Deployment Flow

```
deploy.py
  ├─ 1. Package inference code → model.tar.gz
  │     ├── code/inference.py
  │     ├── code/requirements.txt
  │     └── model_config.json
  ├─ 2. Upload to S3 (CDK-managed bucket, auto-discovered via SSM)
  ├─ 3. SageMaker SDK creates PyTorch model (DLC container 2.5.1)
  └─ 4. Deploy to async inference endpoint
```

### Inference Flow (inside SageMaker container)

```
Request (JSON) → S3 input
  → model_fn()    — Load Earth2Studio model onto GPU
  → input_fn()    — Parse JSON payload
  → predict_fn()  — Run weather forecast (2-5 min)
  → output_fn()   — Serialize results to JSON
S3 output ← Response (JSON with forecast data)
```

### Resource Discovery

The deploy script auto-discovers infrastructure from SSM Parameter Store:

| SSM Parameter | Description |
|---------------|-------------|
| `/earth2/sagemaker/bucket-name` | S3 bucket for model artifacts |
| `/earth2/sagemaker/s3-prefix` | S3 key prefix (`earth2-weather-models`) |
| `/earth2/sagemaker/role-arn` | SageMaker execution role ARN |
| `/earth2/sagemaker/region` | AWS region |

Override with environment variables: `EARTH2_S3_BUCKET`, `EARTH2_SAGEMAKER_ROLE`, etc.

## Files

| File | Description |
|------|-------------|
| `deploy-all.sh` | Batch deploy/delete/status for all models (`--variant ngc\|dlc`) |
| `deploy.py` | Single model deploy script (`--container-variant ngc\|dlc`) |
| `config.py` | Model definitions + SSM auto-discovery |
| `invoke_endpoint.py` | Test endpoint invocation |
| `model_code_dlwp/` | Inference handler for DLWP |
| `model_code_fcn3/` | Inference handler for FCN3 (needs torch-harmonics + makani) |
| `container_fcn3/` | BYOC Dockerfile — NGC base (production, compiled CUDA) |
| `container_fcn3_dlc/` | BYOC Dockerfile — Training DLC base (no NGC auth) |

## Troubleshooting

### Check endpoint logs
```bash
aws logs tail /aws/sagemaker/Endpoints/earth2-dlwp-endpoint --follow --region us-west-2
```

### Check endpoint status
```bash
aws sagemaker describe-endpoint --endpoint-name earth2-dlwp-endpoint --region us-west-2
```

### Common issues

| Error | Cause | Fix |
|-------|-------|-----|
| `Could not access model data` | IAM role missing S3 permissions | Re-deploy `Earth2SageMaker` CDK stack |
| `endpoint-config already exists` | Stale resource from failed deploy | Delete manually: `aws sagemaker delete-endpoint-config --endpoint-config-name <name>` |
| Container timeout | Model download too slow | Increase `SAGEMAKER_MODEL_SERVER_TIMEOUT` in deploy.py |
| OOM (GPU) | Instance too small for model | Check `gpu_memory_gb` in config.py and use larger instance |

## Adding a New Model

1. Add entry to `SUPPORTED_MODELS` in `config.py`:
   ```python
   "my_model": {
       "name": "My Model",
       "model_class": "MyModel",          # Must match earth2studio class name
       "import_path": "earth2studio.models.px",
       "default_instance": "ml.g5.2xlarge",
       "endpoint_prefix": "earth2-mymodel",
       "gpu_memory_gb": 4,                # Check model's gpu badge
       "variables": [],                    # Or list from source code
       "resolution_deg": 0.25,
       "grid_size": (721, 1440),
       "time_step_hours": 6,
   }
   ```

2. Add to `MODELS` array in `deploy-all.sh`:
   ```bash
   MODELS=(
       ...
       "my_model:ml.g5.2xlarge"
   )
   ```

3. Deploy: `python deploy.py --model my_model`
