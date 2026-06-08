#!/usr/bin/env python3
"""
Invoke an NVIDIA Earth-2 Weather Model deployed on Amazon SageMaker (Async).

This script sends inference requests to a deployed SageMaker async endpoint.
The workflow:
  1. Upload JSON payload to S3
  2. Call InvokeEndpointAsync
  3. Poll S3 for the output result (up to 10 minutes)
  4. Display weather forecast results

Usage:
    # Basic: 24-hour forecast from the latest available GFS cycle
    python invoke_endpoint.py --endpoint-name earth2-dlwp-endpoint

    # Specific date and forecast length
    python invoke_endpoint.py \
        --endpoint-name earth2-fcn3-endpoint \
        --date 2025-06-01T00:00:00 \
        --lead-time-hours 48

    # Request specific variables only
    python invoke_endpoint.py \
        --endpoint-name earth2-fcn3-endpoint \
        --variables t2m z500 msl u10m v10m

    # Save results to file
    python invoke_endpoint.py \
        --endpoint-name earth2-dlwp-endpoint \
        --output-file forecast_results.json

Requirements:
    pip install boto3 numpy
    aws configure  (must have SageMaker invoke + S3 permissions)
"""

import argparse
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime
from urllib.parse import urlparse

import boto3

# Add parent directory to path for config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    AWS_REGION,
    S3_BUCKET,
    S3_PREFIX,
    DEFAULT_LEAD_TIME_HOURS,
    ENDPOINT_TIMEOUT_SECONDS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ACTIVE_ENDPOINT_FILE = os.path.join(SCRIPT_DIR, "active_endpoint.json")

# Maximum time to wait for async result (seconds)
ASYNC_POLL_TIMEOUT = 600  # 10 minutes
ASYNC_POLL_INTERVAL = 10  # Check every 10 seconds


def get_active_endpoint_name():
    """Read the active endpoint name from active_endpoint.json (written by deploy.py)."""
    try:
        if os.path.exists(ACTIVE_ENDPOINT_FILE):
            with open(ACTIVE_ENDPOINT_FILE, "r") as f:
                info = json.load(f)
            return info.get("endpoint_name")
    except Exception as e:
        # File missing/corrupt — fall back to CLI-provided endpoint name.
        logger.debug(f"Could not read {ACTIVE_ENDPOINT_FILE}: {e}")
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Invoke NVIDIA Earth-2 weather model on SageMaker (async)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python invoke_endpoint.py --endpoint-name earth2-dlwp-endpoint
  python invoke_endpoint.py --endpoint-name earth2-fcn3-endpoint --date 2025-06-01T00:00:00 --lead-time-hours 48
  python invoke_endpoint.py --endpoint-name earth2-fcn3-endpoint --variables t2m z500 msl u10m v10m
        """,
    )

    parser.add_argument(
        "--endpoint-name",
        type=str,
        default=None,
        help="Name of the SageMaker endpoint to invoke (default: read from active_endpoint.json)",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help=(
            "Initial condition date (ISO format: YYYY-MM-DDTHH:MM:SS). "
            "Must align to a 6-hour GFS cycle (00, 06, 12, 18 UTC). "
            "Default: latest available GFS cycle."
        ),
    )
    parser.add_argument(
        "--lead-time-hours",
        type=int,
        default=DEFAULT_LEAD_TIME_HOURS,
        help=f"Forecast length in hours (default: {DEFAULT_LEAD_TIME_HOURS})",
    )
    parser.add_argument(
        "--variables",
        type=str,
        nargs="+",
        default=None,
        help="Specific weather variables to return (e.g., t2m z500 msl). Default: all.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Save forecast results to this JSON file",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=AWS_REGION,
        help=f"AWS region (default: {AWS_REGION})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print sample forecast data values",
    )

    return parser.parse_args()


def get_s3_bucket(region):
    """Get the S3 bucket for async input/output."""
    bucket = S3_BUCKET
    if bucket == "your-sagemaker-bucket":
        try:
            import sagemaker
            sess = sagemaker.Session(boto_session=boto3.Session(region_name=region))
            bucket = sess.default_bucket()
        except Exception as e:
            # No default SageMaker bucket configured — caller must pass one explicitly.
            logger.debug(f"Default SageMaker bucket unavailable: {e}")
    return bucket


def check_endpoint_status(endpoint_name, region):
    """Check if the endpoint is InService before invoking."""
    sm_client = boto3.client("sagemaker", region_name=region)

    try:
        response = sm_client.describe_endpoint(EndpointName=endpoint_name)
        status = response["EndpointStatus"]

        if status == "InService":
            return True
        elif status == "Creating":
            logger.warning(
                f"Endpoint '{endpoint_name}' is still creating. "
                "Please wait for it to reach 'InService' status.\n"
                f"Check with: aws sagemaker describe-endpoint --endpoint-name {endpoint_name}"
            )
            return False
        elif status == "Failed":
            reason = response.get("FailureReason", "Unknown")
            logger.error(f"Endpoint '{endpoint_name}' FAILED: {reason}")
            return False
        else:
            logger.warning(f"Endpoint '{endpoint_name}' status: {status}")
            return False

    except sm_client.exceptions.ClientError:
        logger.error(
            f"Endpoint '{endpoint_name}' not found.\n"
            "Deploy it first with: python deploy.py --model dlwp"
        )
        return False


def invoke_endpoint_async(endpoint_name, payload, region):
    """
    Send an async inference request to the SageMaker endpoint.

    Workflow:
      1. Upload the JSON payload to S3
      2. Call InvokeEndpointAsync with the S3 input URI
      3. Poll S3 for the output file
      4. Parse and return the result

    Args:
        endpoint_name: Name of the deployed SageMaker endpoint
        payload: Dictionary with the inference request
        region: AWS region

    Returns:
        Parsed JSON response from the endpoint
    """
    bucket = get_s3_bucket(region)
    s3_client = boto3.client("s3", region_name=region)
    sm_runtime = boto3.client("sagemaker-runtime", region_name=region)

    # Generate unique request ID
    request_id = str(uuid.uuid4())[:8]
    input_key = f"{S3_PREFIX}/async-input/{request_id}.json"
    input_uri = f"s3://{bucket}/{input_key}"

    logger.info(f"Invoking endpoint (async): {endpoint_name}")
    logger.info(f"  Date:       {payload.get('date', '(auto - latest GFS cycle)')}")
    logger.info(f"  Lead time:  {payload.get('lead_time_hours')} hours")
    if payload.get("variables"):
        logger.info(f"  Variables:  {', '.join(payload['variables'])}")
    else:
        logger.info("  Variables:  all")

    # Step 1: Upload payload to S3
    logger.info(f"  Uploading request to {input_uri}")
    s3_client.put_object(
        Bucket=bucket,
        Key=input_key,
        Body=json.dumps(payload),
        ContentType="application/json",
    )

    # Step 2: Invoke async endpoint
    start_time = time.time()
    logger.info("  Submitting async inference request...")

    try:
        response = sm_runtime.invoke_endpoint_async(
            EndpointName=endpoint_name,
            InputLocation=input_uri,
            ContentType="application/json",
            Accept="application/json",
        )
    except Exception as e:
        logger.error(f"Failed to invoke async endpoint: {e}")
        # Clean up input — best-effort, don't mask the real error.
        try:
            s3_client.delete_object(Bucket=bucket, Key=input_key)
        except Exception as cleanup_err:
            logger.debug(f"Input cleanup failed (non-fatal): {cleanup_err}")
        raise

    output_uri = response.get("OutputLocation", "")
    logger.info(f"  Request accepted! Output will be at: {output_uri}")

    # Step 3: Poll S3 for the result
    if not output_uri:
        logger.error("No OutputLocation returned from async invocation.")
        return {"status": "error", "error": "No output location returned"}

    parsed = urlparse(output_uri)
    output_bucket = parsed.netloc
    output_key = parsed.path.lstrip("/")

    logger.info(f"  Waiting for result (polling every {ASYNC_POLL_INTERVAL}s, max {ASYNC_POLL_TIMEOUT}s)...")

    elapsed = 0
    while elapsed < ASYNC_POLL_TIMEOUT:
        time.sleep(ASYNC_POLL_INTERVAL)
        elapsed = time.time() - start_time

        try:
            result_obj = s3_client.get_object(Bucket=output_bucket, Key=output_key)
            result_body = result_obj["Body"].read().decode("utf-8")
            result = json.loads(result_body)

            total_time = time.time() - start_time
            logger.info(f"  ✅ Response received in {total_time:.1f}s")

            # Clean up S3 input file — best-effort.
            try:
                s3_client.delete_object(Bucket=bucket, Key=input_key)
            except Exception as cleanup_err:
                logger.debug(f"Input cleanup failed (non-fatal): {cleanup_err}")

            return result

        except s3_client.exceptions.NoSuchKey:
            # Result not ready yet
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            print(f"\r  ⏳ Waiting... {mins}m {secs}s elapsed", end="", flush=True)
            continue

        except Exception as e:
            # Check for error output (SageMaker writes .out for errors)
            error_key = output_key + ".error"
            try:
                error_obj = s3_client.get_object(Bucket=output_bucket, Key=error_key)
                error_body = error_obj["Body"].read().decode("utf-8")
                logger.error(f"\n  Model error: {error_body[:500]}")
                return {"status": "error", "error": error_body}
            except Exception as err_lookup:
                logger.debug(f"No error sidecar at {error_key}: {err_lookup}")

            logger.warning(f"\n  Unexpected error checking result: {e}")
            continue

    print()  # New line after progress
    total_time = time.time() - start_time
    logger.error(f"  Timed out after {total_time:.1f}s waiting for async result.")
    logger.error("  Check CloudWatch logs:")
    logger.error(
        f"    aws logs tail /aws/sagemaker/Endpoints/{endpoint_name} --since 15m --region {region}"
    )
    return {
        "status": "error",
        "error": f"Async inference timed out after {ASYNC_POLL_TIMEOUT}s",
    }


def display_results(result, verbose=False):
    """Pretty-print the forecast results."""

    status = result.get("status", "unknown")

    if status == "error":
        print()
        print("=" * 60)
        print("  ❌ FORECAST FAILED")
        print(f"  Error: {result.get('error', 'Unknown error')}")
        if "traceback" in result:
            print()
            print("  Traceback (from SageMaker container):")
            for line in result["traceback"].split("\n")[:10]:
                print(f"    {line}")
        print("=" * 60)
        return

    model = result.get("model", "unknown")
    init_time = result.get("init_time", "unknown")
    lead_hours = result.get("lead_time_hours", 0)
    n_steps = result.get("n_steps", 0)
    step_hours = result.get("step_hours", 6)

    print()
    print("=" * 70)
    print("  🌤️  WEATHER FORECAST RESULTS")
    print("=" * 70)
    print(f"  Model:           {model}")
    print(f"  Init Time:       {init_time}")
    print(f"  Lead Time:       {lead_hours} hours ({lead_hours / 24:.1f} days)")
    print(f"  Steps:           {n_steps} × {step_hours}h")

    # Display valid times
    valid_times = result.get("valid_times", [])
    if valid_times:
        print(f"  Valid From:      {valid_times[0]}")
        print(f"  Valid To:        {valid_times[-1]}")

    # Display forecast variables
    forecasts = result.get("forecasts", {})
    if forecasts:
        print()
        print(f"  {'Variable':<12} {'Unit':<10} {'Min':>12} {'Mean':>12} {'Max':>12} {'Std':>12}")
        print(f"  {'-' * 12} {'-' * 10} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 12}")

        for var_name, var_data in sorted(forecasts.items()):
            unit = var_data.get("unit", "?")
            vmin = var_data.get("min", 0)
            vmean = var_data.get("mean", 0)
            vmax = var_data.get("max", 0)
            vstd = var_data.get("std", 0)

            # Convert temperature to Celsius and pressure to hPa for display
            if unit == "K" and vmean > 100:
                display_unit = "°C"
                vmin_d = f"{vmin - 273.15:>10.2f}"
                vmean_d = f"{vmean - 273.15:>10.2f}"
                vmax_d = f"{vmax - 273.15:>10.2f}"
                vstd_d = f"{vstd:>10.2f}"
            elif unit == "Pa" and vmean > 10000:
                display_unit = "hPa"
                vmin_d = f"{vmin / 100:>10.1f}"
                vmean_d = f"{vmean / 100:>10.1f}"
                vmax_d = f"{vmax / 100:>10.1f}"
                vstd_d = f"{vstd / 100:>10.1f}"
            else:
                display_unit = unit
                vmin_d = f"{vmin:>10.4g}"
                vmean_d = f"{vmean:>10.4g}"
                vmax_d = f"{vmax:>10.4g}"
                vstd_d = f"{vstd:>10.4g}"

            print(f"  {var_name:<12} {display_unit:<10} {vmin_d:>12} {vmean_d:>12} {vmax_d:>12} {vstd_d:>12}")

        print()
        print(f"  Total variables: {len(forecasts)}")

        # Show data shape info
        first_var = next(iter(forecasts.values()), {})
        shape = first_var.get("shape", [])
        if shape:
            print(f"  Data shape:      {shape}")

        if verbose:
            print()
            print("  Sample data (last timestep, last 10 grid points):")
            for var_name, var_data in sorted(forecasts.items()):
                sample = var_data.get("data_sample", [])[-10:]
                print(f"    {var_name}: {sample}")

    print()
    print("=" * 70)


def save_results(result, output_file):
    """Save forecast results to a JSON file."""
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"Results saved to: {output_file}")


def main():
    args = parse_args()

    # Resolve endpoint name: CLI arg > active_endpoint.json
    if not args.endpoint_name:
        args.endpoint_name = get_active_endpoint_name()
        if args.endpoint_name:
            logger.info(f"Using active endpoint from active_endpoint.json: {args.endpoint_name}")
        else:
            logger.error(
                "No endpoint name provided and no active_endpoint.json found.\n"
                "Either pass --endpoint-name or deploy a model first with deploy.py"
            )
            sys.exit(1)

    # Check endpoint status first
    if not check_endpoint_status(args.endpoint_name, args.region):
        sys.exit(1)

    # Build the inference payload
    payload = {
        "lead_time_hours": args.lead_time_hours,
    }

    if args.date:
        payload["date"] = args.date

    if args.variables:
        payload["variables"] = args.variables

    # Invoke the async endpoint
    result = invoke_endpoint_async(args.endpoint_name, payload, args.region)

    # Display results
    display_results(result, verbose=args.verbose)

    # Save to file if requested
    if args.output_file:
        save_results(result, args.output_file)

    # Return non-zero exit code on error
    if result.get("status") == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()
