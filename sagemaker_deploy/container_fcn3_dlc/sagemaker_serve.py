"""
SageMaker BYOC Inference Server — Flask/Gunicorn implementation.

Implements the SageMaker inference container contract:
    GET  /ping          → 200 OK when model is loaded and ready
    POST /invocations   → run inference using the handler from inference.py

This bridges the existing inference.py handler (model_fn/input_fn/predict_fn/output_fn)
to the SageMaker BYOC HTTP contract, replacing the torchserve/MMS stack that
was pre-installed in the now-deprecated PyTorch inference DLC.

Model loading:
    The model is loaded eagerly at import time (gunicorn --preload).
    SageMaker extracts model.tar.gz into /opt/ml/model/ which contains:
        code/inference.py       — the handler functions
        code/requirements.txt   — extra pip deps
        model_config.json       — model class/import path config
"""

import importlib.util
import json
import logging
import os
import sys
import traceback

from flask import Flask, Response, request

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("sagemaker_serve")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_DIR = os.environ.get("SAGEMAKER_MODEL_DIR", "/opt/ml/model")
INFERENCE_CODE_PATH = os.path.join(MODEL_DIR, "code", "inference.py")

# ---------------------------------------------------------------------------
# Load inference.py dynamically from the model artifact
# ---------------------------------------------------------------------------
_inference_module = None


def _load_inference_module():
    """
    Dynamically import inference.py from /opt/ml/model/code/inference.py.

    SageMaker extracts model.tar.gz into MODEL_DIR at container startup,
    so inference.py is available at INFERENCE_CODE_PATH.
    """
    global _inference_module

    if not os.path.exists(INFERENCE_CODE_PATH):
        raise FileNotFoundError(
            f"inference.py not found at {INFERENCE_CODE_PATH}. "
            f"MODEL_DIR contents: {os.listdir(MODEL_DIR) if os.path.exists(MODEL_DIR) else 'DIR NOT FOUND'}"
        )

    # Add model code directory to sys.path so inference.py can do relative imports
    code_dir = os.path.dirname(INFERENCE_CODE_PATH)
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)

    spec = importlib.util.spec_from_file_location("inference", INFERENCE_CODE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Verify required functions exist
    for fn_name in ["model_fn", "input_fn", "predict_fn", "output_fn"]:
        if not hasattr(module, fn_name):
            raise AttributeError(
                f"inference.py is missing required function: {fn_name}"
            )

    _inference_module = module
    logger.info(f"Loaded inference.py from {INFERENCE_CODE_PATH}")
    return module


# ---------------------------------------------------------------------------
# Model Loading (eager — called once at startup via gunicorn --preload)
# ---------------------------------------------------------------------------
_model = None
_model_load_error = None


def _load_model():
    """Load the model using model_fn from inference.py."""
    global _model, _model_load_error

    try:
        logger.info("=" * 60)
        logger.info("  Loading model...")
        logger.info(f"  MODEL_DIR: {MODEL_DIR}")
        logger.info("=" * 60)

        module = _load_inference_module()
        _model = module.model_fn(MODEL_DIR)

        logger.info("=" * 60)
        logger.info("  ✅ Model loaded successfully!")
        logger.info("  Server is ready for inference requests.")
        logger.info("=" * 60)

    except Exception as e:
        _model_load_error = traceback.format_exc()
        logger.error(f"❌ Model loading failed: {e}")
        logger.error(_model_load_error)


# Model loading strategy:
# We do NOT load at import time or with --preload because CUDA cannot be
# re-initialized in a forked subprocess (gunicorn forks before importing).
# Instead, the model is loaded lazily on the first /ping request.
# SageMaker sends /ping health checks repeatedly during startup — the first
# one triggers model loading, subsequent ones return 503 while loading,
# and once loaded, /ping returns 200.
_model_loading = False


def _ensure_model_loaded():
    """Load the model if not already loaded. Thread-safe for single worker."""
    global _model_loading
    if _model is None and _model_load_error is None and not _model_loading:
        _model_loading = True
        _load_model()
        _model_loading = False


# ---------------------------------------------------------------------------
# Flask App
# ---------------------------------------------------------------------------
app = Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    """
    SageMaker health check endpoint.

    Returns 200 if the model is loaded and ready.
    Returns 503 if model is still loading or failed.
    First /ping triggers model loading.
    """
    # Trigger lazy model loading on first ping
    _ensure_model_loaded()

    if _model is not None:
        return Response(status=200, response="OK\n", mimetype="text/plain")
    else:
        # Return 503 so SageMaker knows the container isn't ready yet
        error_msg = _model_load_error or "Model still loading..."
        logger.warning(f"/ping returning 503: {error_msg[:200]}")
        return Response(
            status=503,
            response=json.dumps({"status": "unhealthy", "error": error_msg[:500]}),
            mimetype="application/json",
        )


@app.route("/invocations", methods=["POST"])
def invocations():
    """
    SageMaker inference endpoint.

    Receives POST requests with JSON payloads, runs through the
    inference.py handler chain: input_fn → predict_fn → output_fn.
    """
    if _model is None:
        error_msg = _model_load_error or "Model not loaded"
        return Response(
            status=503,
            response=json.dumps({"status": "error", "error": error_msg[:500]}),
            mimetype="application/json",
        )

    try:
        # Get request details
        content_type = request.content_type or "application/json"
        accept = request.accept_mimetypes.best or "application/json"
        request_body = request.get_data()

        logger.info(f"Inference request: content_type={content_type}, "
                     f"accept={accept}, body_size={len(request_body)} bytes")

        # Run the handler chain from inference.py
        # 1. Deserialize input
        parsed_input = _inference_module.input_fn(request_body, content_type)

        # 2. Run prediction
        prediction = _inference_module.predict_fn(parsed_input, _model)

        # 3. Serialize output
        response_body, response_content_type = _inference_module.output_fn(
            prediction, accept
        )

        logger.info(f"Inference complete: response_size={len(response_body)} bytes")

        return Response(
            status=200,
            response=response_body,
            mimetype=response_content_type,
        )

    except Exception:
        # Log the full traceback server-side, but do NOT leak it to the
        # client — tracebacks expose container paths, library versions,
        # and occasionally request data parsed during input deserialization.
        logger.exception("Inference error")
        return Response(
            status=500,
            response=json.dumps({
                "status": "error",
                "error": "Internal inference error. See container logs.",
            }),
            mimetype="application/json",
        )


@app.route("/execution-parameters", methods=["GET"])
def execution_parameters():
    """
    Optional SageMaker endpoint for batch transform parameters.
    Returns default execution parameters.
    """
    return Response(
        status=200,
        response=json.dumps({
            "MaxConcurrentTransforms": 1,
            "BatchStrategy": "SINGLE_RECORD",
            "MaxPayloadInMB": 6,
        }),
        mimetype="application/json",
    )


# ---------------------------------------------------------------------------
# Direct execution (for local testing only).
#
# The production path is `serve` → gunicorn, which binds 0.0.0.0:8080 because
# the SageMaker BYOC contract requires the container to listen on all
# interfaces (the SageMaker runtime forwards /ping and /invocations from a
# private VPC). This __main__ block is a developer convenience for running
# the Flask app outside SageMaker; it binds to loopback only by default.
#
# Override with FLASK_HOST=0.0.0.0 if you knowingly need to test from
# another machine on a trusted network.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", "8080"))
    logger.info(f"Starting Flask dev server on {host}:{port} (LOCAL TESTING ONLY)...")
    app.run(host=host, port=port, debug=False)
