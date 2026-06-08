#!/usr/bin/env python3
"""
Local development server that wraps the Lambda handler.

Usage:
    pip install flask flask-cors
    cd backend
    python local_server.py

This serves the same API as the Lambda function at http://localhost:3001/api/*
using your existing AWS credentials (same as invoke_endpoint.py uses).
The Vite dev server (port 5173) proxies /api/* to this server.
"""

import json
import sys
import os

# Ensure handler can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify
from flask_cors import CORS

# Validate required environment variables before importing handler. The easiest
# way to set these is to source them from the SSM parameters published by the
# Earth2SageMaker CDK stack:
#
#   export AWS_REGION=us-east-1
#   export S3_BUCKET=$(aws ssm get-parameter --name /earth2/sagemaker/bucket-name --query Parameter.Value --output text)
#   python local_server.py
if not os.environ.get("S3_BUCKET"):
    raise RuntimeError(
        "S3_BUCKET is not set. Run:\n"
        "  export S3_BUCKET=$(aws ssm get-parameter --name /earth2/sagemaker/bucket-name --query Parameter.Value --output text)"
    )
os.environ.setdefault("REGION", os.environ.get("AWS_REGION", ""))
os.environ.setdefault("S3_PREFIX", "earth2-weather-models")

from handler import lambda_handler

app = Flask(__name__)
CORS(app)


def invoke_handler(http_method, path, body=None, path_params=None):
    """Simulate an API Gateway event and invoke the Lambda handler."""
    event = {
        "httpMethod": http_method,
        "path": path,
        "pathParameters": path_params or {},
        "queryStringParameters": dict(request.args) if request.args else {},
        "headers": dict(request.headers),
        "body": body,
    }

    result = lambda_handler(event, None)

    status_code = result.get("statusCode", 200)
    raw_body = result.get("body", "{}")
    try:
        response_body = json.loads(raw_body)
    except (TypeError, ValueError):
        response_body = {"error": "Invalid response body from handler"}
        status_code = 500
    response = jsonify(response_body)
    response.status_code = status_code

    # Copy headers from Lambda response. Reject any header whose name or value
    # contains CR/LF to prevent HTTP response splitting (CWE-113) — though
    # Werkzeug also rejects these, defending in depth keeps CodeQL happy.
    for key, value in result.get("headers", {}).items():
        if key.lower() == "content-type":
            continue
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        if "\r" in key or "\n" in key or "\r" in value or "\n" in value:
            continue
        response.headers[key] = value

    return response


@app.route("/api/endpoints", methods=["GET"])
def get_endpoints():
    return invoke_handler("GET", "/api/endpoints")


@app.route("/api/forecast", methods=["POST"])
def post_forecast():
    return invoke_handler("POST", "/api/forecast", body=request.get_data(as_text=True))


@app.route("/api/forecast/<request_id>", methods=["GET"])
def poll_forecast(request_id):
    return invoke_handler(
        "GET",
        f"/api/forecast/{request_id}",
        path_params={"request_id": request_id},
    )


@app.route("/api/status/<endpoint_name>", methods=["GET"])
def get_status(endpoint_name):
    return invoke_handler(
        "GET",
        f"/api/status/{endpoint_name}",
        path_params={"endpoint_name": endpoint_name},
    )


if __name__ == "__main__":
    # Local dev server defaults: bind to loopback only and disable Flask's
    # debug reloader/PIN console. The Vite dev server (port 5173) proxies
    # /api/* to this process, and Vite runs on the same host, so 127.0.0.1
    # is sufficient for the documented dev workflow.
    #
    # If you genuinely need to reach this from another machine on your LAN,
    # set FLASK_HOST=0.0.0.0 explicitly (and consider that this skips Cognito
    # auth — never expose this beyond localhost in untrusted networks).
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", "3001"))
    debug = os.environ.get("FLASK_DEBUG", "0").lower() in ("1", "true", "yes")

    print()
    print("=" * 60)
    print("  🌤️  Earth2Studio Local API Server")
    print("=" * 60)
    print()
    print(f"  API:      http://{host}:{port}/api/")
    print("  Frontend: http://localhost:5173/ (run 'npm run dev' in frontend/)")
    print()
    print("  Routes:")
    print("    GET  /api/endpoints")
    print("    POST /api/forecast")
    print("    GET  /api/status/<endpoint_name>")
    print()
    print("  Using AWS credentials from your environment.")
    print("  Ctrl+C to stop.")
    print("=" * 60)
    print()

    app.run(host=host, port=port, debug=debug)
