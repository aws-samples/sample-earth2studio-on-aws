# Earth2Studio FCN3 container for SageMaker inference (BYOC)
# ============================================================
# ALTERNATIVE: Uses AWS SageMaker Training DLC as base image
# instead of NGC (see container_fcn3/ for the NGC version).
#
# Advantage: No NGC API key needed — the Training DLC is in
# AWS's own ECR (763104351884), which CodeBuild can access
# with standard ECR auth.
#
# Trade-off: Uses PyPI torch-harmonics wheel (no CUDA C++
# extensions compiled) — FCN3 falls back to float32 SHT path.
# Performance difference is minimal for inference.
# ============================================================
#
# Build:
#   docker build -t earth2-fcn3-dlc -f Dockerfile .
#
# Region note:
#   The us-west-2 region in the FROM line below and in the push commands
#   here is the example region used throughout this repo. If you deploy in
#   a different region, replace BOTH occurrences with your region — for
#   example us-east-1, eu-west-1, ap-northeast-1. The 763104351884 account
#   is AWS's public Deep Learning Containers account and is the same in
#   every region; only the regional ECR endpoint changes. The full list of
#   supported regions is at:
#     https://github.com/aws/deep-learning-containers/blob/master/available_images.md
#   The 123456789012 below is the canonical AWS-docs placeholder for YOUR
#   account ID — replace it with your actual 12-digit AWS account ID.
#
# Push to ECR:
#   aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-west-2.amazonaws.com
#   docker tag earth2-fcn3-dlc:latest 123456789012.dkr.ecr.us-west-2.amazonaws.com/earth2-fcn3-dlc:latest
#   docker push 123456789012.dkr.ecr.us-west-2.amazonaws.com/earth2-fcn3-dlc:latest

# ---------------------------------------------------------------------------
# Base: SageMaker PyTorch Training DLC with Blackwell support
# PyTorch 2.7.1 + CUDA 12.8 — supports sm_120 (Blackwell)
# No NGC auth needed — this is in AWS's ECR (763104351884)
# Region: us-west-2 is the example region used in this repo. To deploy in
# a different region, swap "us-west-2" in the FROM line for your region
# (e.g. us-east-1.dkr.ecr.us-east-1.amazonaws.com/...). The 763104351884
# account is the same in every region; only the regional ECR endpoint
# changes. See the AWS DLC available-images list for supported regions.
# ---------------------------------------------------------------------------
FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.7.1-gpu-py312-cu128-ubuntu22.04-sagemaker

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    git make curl cmake \
    libeccodes-tools libeccodes-dev \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Earth2Studio + FCN3 dependencies
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir \
    "earth2studio>=0.3.0,<1.0.0" \
    "nvidia-physicsnemo>=1.0.0" \
    "zarr>=3.0.0" \
    "xarray>=2024.1.0" \
    "netCDF4>=1.6.0" \
    "ruamel.yaml>=0.18.14" \
    "scipy>=1.10.0"

# torch-harmonics: PyPI wheel (no CUDA C++ extensions)
# FCN3 auto-detects and uses the float32 SHT path
RUN pip install --no-cache-dir torch-harmonics>=0.8.0

# makani: NVIDIA model loader for FCN3/SFNO
# moviepy + wandb: required transitive deps of makani
RUN pip install --no-cache-dir \
    moviepy wandb \
    "makani @ git+https://github.com/NVIDIA/makani.git@b38fcb2799d7dbc146fa60459f3f9823394a8bf1"

# ---------------------------------------------------------------------------
# SageMaker Serving Stack
# ---------------------------------------------------------------------------
# Flask + gunicorn implement the SageMaker BYOC inference contract:
#   GET  /ping        → health check
#   POST /invocations → inference
#   Port 8080
RUN pip install --no-cache-dir flask gunicorn boto3

# ---------------------------------------------------------------------------
# SageMaker Serving Files
# ---------------------------------------------------------------------------
COPY serve /usr/local/bin/serve
COPY sagemaker_serve.py /opt/program/sagemaker_serve.py
RUN chmod +x /usr/local/bin/serve

# SageMaker expects model artifacts at /opt/ml/model
RUN mkdir -p /opt/ml/model

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
ENV EARTH2STUDIO_CACHE=/tmp/earth2_cache
ENV SAGEMAKER_PROGRAM=inference.py

# Verify FCN3 can be imported
RUN python -c "from earth2studio.models.px import FCN3; print('✅ FCN3 import successful (Training DLC)')"

WORKDIR /opt/program

# Container health check — uses the same /ping endpoint that SageMaker probes.
# `curl` is already installed above as a system dependency. Long start period
# (1500 s) accommodates the first-call model weight download from NVIDIA's
# registry, which can take several minutes on a cold container.
HEALTHCHECK --interval=30s --timeout=10s --start-period=1500s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/ping || exit 1

# SageMaker runs "serve" to start the inference server
ENTRYPOINT ["serve"]
