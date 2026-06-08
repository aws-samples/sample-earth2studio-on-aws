"""
Lambda handler for Earth2Studio Weather Forecast API.

Routes:
  GET  /api/endpoints          — List active SageMaker endpoints
  POST /api/forecast           — Run async SageMaker forecast
  GET  /api/status/{name}      — Check endpoint status
"""

import json
import logging
import os
import time
import uuid
from urllib.parse import urlparse

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REGION = os.environ.get("REGION") or os.environ.get("AWS_REGION")
S3_BUCKET = os.environ.get("S3_BUCKET")
S3_PREFIX = os.environ.get("S3_PREFIX", "earth2-weather-models")

if not REGION:
    raise RuntimeError("REGION (or AWS_REGION) environment variable is required.")
if not S3_BUCKET:
    raise RuntimeError("S3_BUCKET environment variable is required.")

# Clients (reused across warm invocations)
_sm_client = None
_sm_runtime = None
_s3_client = None


def _sagemaker():
    global _sm_client
    if _sm_client is None:
        _sm_client = boto3.client("sagemaker", region_name=REGION)
    return _sm_client


def _sagemaker_runtime():
    global _sm_runtime
    if _sm_runtime is None:
        _sm_runtime = boto3.client("sagemaker-runtime", region_name=REGION)
    return _sm_runtime


def _s3():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3", region_name=REGION)
    return _s3_client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _json_response(status_code, body):
    """Return an API Gateway-compatible JSON response."""
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        },
        "body": json.dumps(body, default=str),
    }


# ---------------------------------------------------------------------------
# Route: GET /api/endpoints
# ---------------------------------------------------------------------------

def handle_list_endpoints():
    """List all InService earth2-* SageMaker endpoints."""
    try:
        sm = _sagemaker()
        response = sm.list_endpoints(
            StatusEquals="InService",
            MaxResults=20,
            SortBy="Name",
        )
        endpoints = []
        for ep in response.get("Endpoints", []):
            name = ep["EndpointName"]
            if name.startswith("earth2-"):
                model_name = name.replace("earth2-", "").replace("-endpoint", "").upper()
                endpoints.append({
                    "endpoint_name": name,
                    "model": model_name,
                    "status": "InService",
                    "created": ep.get("CreationTime", ""),
                })
        return _json_response(200, {"endpoints": endpoints})
    except Exception as e:
        logger.exception("Error listing endpoints")
        return _json_response(500, {"error": str(e)})


# ---------------------------------------------------------------------------
# Route: POST /api/forecast
# ---------------------------------------------------------------------------

POLL_INTERVAL = 3   # seconds between S3 polls
MAX_POLL_TIME = 25  # must finish well within API GW 29s limit


def handle_forecast(body):
    """
    Run an async SageMaker forecast.

    1. Upload payload to S3
    2. Call InvokeEndpointAsync
    3. Poll S3 for output (up to MAX_POLL_TIME seconds)
    4. Return result or "pending" with request_id
    """
    try:
        endpoint_name = body.get("endpoint_name")
        if not endpoint_name:
            return _json_response(400, {"error": "endpoint_name is required"})

        date_str = body.get("date")
        lead_time_hours = body.get("lead_time_hours", 24)
        variables = body.get("variables")
        return_grid = body.get("return_grid", True)

        # Build inference payload
        payload = {
            "lead_time_hours": lead_time_hours,
            "return_grid": return_grid,
        }
        if date_str:
            payload["date"] = date_str
        if variables:
            payload["variables"] = variables

        s3 = _s3()
        sm_runtime = _sagemaker_runtime()

        request_id = str(uuid.uuid4())[:8]
        input_key = f"{S3_PREFIX}/async-input/{request_id}.json"
        input_uri = f"s3://{S3_BUCKET}/{input_key}"

        # Step 1: Upload payload to S3
        logger.info(f"Uploading request to {input_uri}")
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=input_key,
            Body=json.dumps(payload),
            ContentType="application/json",
        )

        # Step 2: Invoke async endpoint
        logger.info(f"Invoking async endpoint: {endpoint_name}")
        try:
            response = sm_runtime.invoke_endpoint_async(
                EndpointName=endpoint_name,
                InputLocation=input_uri,
                ContentType="application/json",
                Accept="application/json",
            )
        except Exception as e:
            logger.error(f"Failed to invoke endpoint: {e}")
            # Clean up input — best-effort, don't mask the real error.
            try:
                s3.delete_object(Bucket=S3_BUCKET, Key=input_key)
            except Exception as cleanup_err:
                logger.debug(f"Input cleanup failed (non-fatal): {cleanup_err}")
            return _json_response(502, {"error": f"SageMaker invocation failed: {str(e)}"})

        output_uri = response.get("OutputLocation", "")
        if not output_uri:
            return _json_response(502, {"error": "No OutputLocation returned from SageMaker"})

        parsed = urlparse(output_uri)
        output_bucket = parsed.netloc
        output_key = parsed.path.lstrip("/")

        # Step 3: Poll S3 for result
        logger.info(f"Polling for result at {output_uri}")
        start_time = time.time()

        while (time.time() - start_time) < MAX_POLL_TIME:
            time.sleep(POLL_INTERVAL)
            try:
                result_obj = s3.get_object(Bucket=output_bucket, Key=output_key)
                result_body = result_obj["Body"].read().decode("utf-8")
                result = json.loads(result_body)

                elapsed = time.time() - start_time
                logger.info(f"Result received in {elapsed:.1f}s")

                # Clean up input file — best-effort.
                try:
                    s3.delete_object(Bucket=S3_BUCKET, Key=input_key)
                except Exception as cleanup_err:
                    logger.debug(f"Input cleanup failed (non-fatal): {cleanup_err}")

                return _json_response(200, result)

            except s3.exceptions.NoSuchKey:
                # Not ready yet
                continue
            except Exception as e:
                # Check for error output
                error_key = output_key + ".error"
                try:
                    error_obj = s3.get_object(Bucket=output_bucket, Key=error_key)
                    error_body = error_obj["Body"].read().decode("utf-8")
                    logger.error(f"Model error: {error_body[:500]}")
                    return _json_response(500, {"status": "error", "error": error_body})
                except Exception as err_lookup:
                    logger.debug(f"No error sidecar at {error_key}: {err_lookup}")
                logger.warning(f"Unexpected error polling result: {e}")
                continue

        # Timed out — return pending for frontend to poll
        logger.info(f"Result not ready after {MAX_POLL_TIME}s, returning pending")
        return _json_response(202, {
            "status": "pending",
            "request_id": request_id,
            "output_location": output_uri,
            "message": "Forecast is running. Poll GET /api/forecast/{request_id} for results.",
        })

    except Exception as e:
        logger.exception("Error in forecast handler")
        return _json_response(500, {"error": str(e)})


# ---------------------------------------------------------------------------
# Route: GET /api/forecast/{request_id}  — poll for async result
# ---------------------------------------------------------------------------

def handle_poll_forecast(request_id, query_params):
    """Poll S3 for an async forecast result."""
    try:
        output_location = (query_params or {}).get("output_location", "")
        if not output_location:
            return _json_response(400, {"error": "output_location query param is required"})

        parsed = urlparse(output_location)
        output_bucket = parsed.netloc
        output_key = parsed.path.lstrip("/")

        s3 = _s3()

        # Check for result. NoSuchKey just means the forecast is still running;
        # fall through to the error-sidecar check below.
        try:
            result_obj = s3.get_object(Bucket=output_bucket, Key=output_key)
            result_body = result_obj["Body"].read().decode("utf-8")
            result = json.loads(result_body)
            return _json_response(200, result)
        except s3.exceptions.NoSuchKey:
            logger.debug(f"Forecast result not yet at s3://{output_bucket}/{output_key}")

        # Check for error sidecar (".error" suffix). NoSuchKey here means no
        # error has been written either, so the forecast is genuinely pending.
        error_key = output_key + ".error"
        try:
            error_obj = s3.get_object(Bucket=output_bucket, Key=error_key)
            error_body = error_obj["Body"].read().decode("utf-8")
            return _json_response(500, {"status": "error", "error": error_body})
        except s3.exceptions.NoSuchKey:
            logger.debug(f"No error sidecar at s3://{output_bucket}/{error_key}")

        # Still pending
        return _json_response(202, {
            "status": "pending",
            "request_id": request_id,
            "output_location": output_location,
            "message": "Forecast still running.",
        })

    except Exception as e:
        logger.exception("Error polling forecast result")
        return _json_response(500, {"error": str(e)})


# ---------------------------------------------------------------------------
# Route: GET /api/status/{endpoint_name}
# ---------------------------------------------------------------------------

def handle_status(endpoint_name):
    """Describe a SageMaker endpoint status."""
    try:
        sm = _sagemaker()
        response = sm.describe_endpoint(EndpointName=endpoint_name)
        return _json_response(200, {
            "endpoint_name": endpoint_name,
            "status": response["EndpointStatus"],
        })
    except sm.exceptions.ClientError as e:
        if "Could not find endpoint" in str(e) or "ValidationException" in str(e):
            return _json_response(404, {
                "endpoint_name": endpoint_name,
                "status": "NotFound",
                "error": f"Endpoint '{endpoint_name}' not found",
            })
        return _json_response(500, {"error": str(e)})
    except Exception as e:
        logger.exception("Error describing endpoint")
        return _json_response(500, {"error": str(e)})


# ---------------------------------------------------------------------------
# Lambda entry point
# ---------------------------------------------------------------------------

def lambda_handler(event, context):
    """Route incoming API Gateway events to the appropriate handler."""
    logger.info(f"Event: {json.dumps(event, default=str)}")

    http_method = event.get("httpMethod", "GET")
    path = event.get("path", "/")
    path_params = event.get("pathParameters") or {}

    # OPTIONS — CORS preflight
    if http_method == "OPTIONS":
        return _json_response(200, {})

    # GET /api/endpoints
    if path == "/api/endpoints" and http_method == "GET":
        return handle_list_endpoints()

    # POST /api/forecast
    if path == "/api/forecast" and http_method == "POST":
        try:
            body = json.loads(event.get("body", "{}") or "{}")
        except json.JSONDecodeError:
            return _json_response(400, {"error": "Invalid JSON body"})
        return handle_forecast(body)

    # GET /api/forecast/{request_id} — poll for async result
    if path.startswith("/api/forecast/") and http_method == "GET":
        request_id = path_params.get("request_id") or path.split("/api/forecast/")[-1]
        query_params = event.get("queryStringParameters") or {}
        return handle_poll_forecast(request_id, query_params)

    # GET /api/status/{endpoint_name}
    if path.startswith("/api/status/") and http_method == "GET":
        endpoint_name = path_params.get("endpoint_name") or path.split("/api/status/")[-1]
        if not endpoint_name:
            return _json_response(400, {"error": "endpoint_name is required"})
        return handle_status(endpoint_name)

    # Unknown route
    return _json_response(404, {"error": f"Not found: {http_method} {path}"})
