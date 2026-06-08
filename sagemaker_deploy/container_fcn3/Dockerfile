# Earth2Studio FCN3 container for SageMaker inference (BYOC)
#
# Based on: local_files/earth2-eks-inference/container/Dockerfile
# Same NGC base image + dependency versions, with SageMaker serving stack added.
#
# The EKS Dockerfile is proven working on Blackwell (g7e instances).
# We mirror it exactly and add a lightweight Flask/gunicorn server
# to implement the SageMaker BYOC inference contract:
#   GET  /ping          → 200 OK (health check)
#   POST /invocations   → inference (calls model_fn/input_fn/predict_fn/output_fn)
#   Listening on port 8080
#
# NGC auth required for base image:
#   docker login nvcr.io -u '$oauthtoken' -p <NGC_API_KEY>
#
# Build:
#   docker build -t earth2-fcn3-sagemaker -f Dockerfile .
#
# Push to ECR (replace ACCOUNT_ID and REGION with your own):
#   aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com
#   docker tag earth2-fcn3-sagemaker:latest $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/earth2-fcn3:latest
#   docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/earth2-fcn3:latest

# ---------------------------------------------------------------------------
# Base: NVIDIA NGC PyTorch container — proven Blackwell support (PyTorch 2.7+, CUDA 12.8+)
# Same image as the working EKS deployment
# ---------------------------------------------------------------------------
FROM nvcr.io/nvidia/pytorch:25.12-py3

# ---------------------------------------------------------------------------
# System dependencies (mirrors EKS Dockerfile)
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    git make curl cmake python3-dev \
    libeccodes-tools libeccodes-dev \
    && rm -rf /var/lib/apt/lists/*

# Remove pip constraints from NGC container (mirrors EKS Dockerfile)
ENV PIP_CONSTRAINT=

# Install uv (fast package manager, same as EKS)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# ---------------------------------------------------------------------------
# Earth2Studio with model extras (same versions as EKS)
# ---------------------------------------------------------------------------
RUN uv pip install --system --break-system-packages \
    "earth2studio[dlwp,fcn,sfno,data]@git+https://github.com/NVIDIA/earth2studio.git@0.12.1"

# ---------------------------------------------------------------------------
# FCN3 Dependencies & Multi-Arch CUDA Compilation (mirrors EKS exactly)
# 8.9 = L40S/Ada (g6e), 12.0 = RTX PRO 6000/Blackwell (g7e)
# ---------------------------------------------------------------------------
ENV TORCH_CUDA_ARCH_LIST="8.9 12.0"
ENV FORCE_CUDA=1
ENV FORCE_CUDA_EXTENSION=1
ENV MAX_JOBS=32

# torch-harmonics: build from source with CUDA C++ extensions (same as EKS)
RUN git clone --branch v0.8.0 --depth 1 https://github.com/NVIDIA/torch-harmonics.git /tmp/torch-harmonics && \
    cd /tmp/torch-harmonics && \
    pip install --no-build-isolation --no-deps . && \
    rm -rf /tmp/torch-harmonics

# makani: NVIDIA model loader for FCN3/SFNO (same commit as EKS)
RUN uv pip install --system --break-system-packages \
    "makani @ git+https://github.com/NVIDIA/makani.git@fd1a62b9f27aebbc793f0cbafd838a3811de43fd"

# Finalize Earth2Studio FCN3 extras (same as EKS)
RUN uv pip install --system --break-system-packages \
    "earth2studio[fcn3]@git+https://github.com/NVIDIA/earth2studio.git@0.12.1"

# moviepy + wandb: transitive deps of makani (visualize.py imports them)
RUN uv pip install --system --break-system-packages moviepy wandb

# ---------------------------------------------------------------------------
# SageMaker Serving Stack (NOT in EKS — this is the SageMaker-specific addition)
# ---------------------------------------------------------------------------
# Flask + gunicorn implement the SageMaker BYOC inference contract:
#   GET  /ping        → health check
#   POST /invocations → inference
#   Port 8080
RUN uv pip install --system --break-system-packages \
    flask gunicorn boto3

# ---------------------------------------------------------------------------
# SageMaker Serving Files
# ---------------------------------------------------------------------------
# Copy the serve entrypoint and Flask app into the container
COPY serve /usr/local/bin/serve
COPY sagemaker_serve.py /opt/program/sagemaker_serve.py
RUN chmod +x /usr/local/bin/serve

# SageMaker expects model artifacts at /opt/ml/model
# model.tar.gz is extracted here at runtime (contains code/inference.py + model_config.json)
RUN mkdir -p /opt/ml/model

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
ENV EARTH2STUDIO_CACHE=/tmp/earth2_cache
ENV SAGEMAKER_PROGRAM=inference.py

# Verify FCN3 can be imported
RUN python -c "from earth2studio.models.px import FCN3; print('✅ FCN3 import successful')"

WORKDIR /opt/program

# Container health check — uses the same /ping endpoint that SageMaker probes.
# `curl` is already installed above as a system dependency. Long start period
# (1500 s) accommodates the first-call model weight download from NVIDIA's
# registry, which can take several minutes on a cold container.
HEALTHCHECK --interval=30s --timeout=10s --start-period=1500s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/ping || exit 1

# SageMaker runs "serve" to start the inference server
ENTRYPOINT ["serve"]
