"""
SageMaker Inference Handler for NVIDIA Earth2Studio Weather Models.

This script runs INSIDE the SageMaker PyTorch container. It implements the
standard SageMaker inference contract:
  - model_fn()    → Load the model onto GPU
  - input_fn()    → Deserialize the incoming request
  - predict_fn()  → Run the weather forecast
  - output_fn()   → Serialize the forecast results

Verified against:
  - Earth2Studio source: https://github.com/NVIDIA/earth2studio
  - Earth2Studio models/px/__init__.py for correct class names
  - Earth2Studio run.py for deterministic() signature
  - Earth2Studio io/zarr.py for ZarrBackend API

Supported models (Apache-2.0 only, from earth2studio.models.px):
  DLWP

The inference request is a JSON payload specifying:
  - date: initial condition date (ISO format, e.g., "2025-06-01T00:00:00")
  - lead_time_hours: how many hours ahead to forecast (default: 24)
  - variables: optional list of specific variables to return

The response is a JSON containing forecast summary stats and metadata.
"""

import importlib
import json
import logging
import os
import sys
import traceback
from collections import OrderedDict
from datetime import datetime, timedelta
from io import BytesIO

import numpy as np
import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
logger.addHandler(handler)


# ============================================================
# SECURITY: Allow-list for dynamic model imports
# ============================================================
# Mitigates SAST finding `python.lang.security.audit.non-literal-import`.
# `importlib.import_module()` is called below with values that originate
# from `model_config.json`, which is packaged into model.tar.gz at build
# time (see ../deploy.py::package_model_artifacts). It is NOT user input.
# We still hard-code the only acceptable (module, class) pairs so that an
# attacker who somehow tampers with the artifact cannot load arbitrary code.
ALLOWED_MODEL_IMPORTS = {
    "earth2studio.models.px": {"DLWP"},
}



# ============================================================
# MODEL LOADING
# ============================================================

def model_fn(model_dir):
    """
    Load the Earth2Studio weather model.

    Called once when the SageMaker endpoint starts.
    The model_dir contains the extracted model.tar.gz contents.

    Earth2Studio models use AutoModelMixin with load_default_package()
    and load_model() class methods. Model weights are downloaded from
    NVIDIA's model registry on first load and cached.

    Args:
        model_dir: Path to the directory containing model artifacts

    Returns:
        dict with loaded model objects and metadata
    """
    logger.info(f"Loading model from {model_dir}")
    logger.info(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Read model configuration from the packaged config
    config_path = os.path.join(model_dir, "model_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            model_config = json.load(f)
    else:
        model_config = {}

    model_class_name = model_config.get("model_class", os.environ.get("EARTH2_MODEL_CLASS", "FCN"))
    import_path = model_config.get("import_path", "earth2studio.models.px")

    logger.info(f"Loading model: {import_path}.{model_class_name}")

    try:
        # Dynamically import the model class from earth2studio.
        # This handler is wired up for DLWP; other Apache-2.0-licensed
        # earth2studio classes can be selected via model_config.json.
        #
        # SECURITY: Validate against ALLOWED_MODEL_IMPORTS before calling
        # importlib.import_module(). This prevents arbitrary code loading
        # even if an attacker tampers with the packaged model_config.json.
        # See SAST finding: python.lang.security.audit.non-literal-import.
        allowed_classes = ALLOWED_MODEL_IMPORTS.get(import_path)
        if not allowed_classes or model_class_name not in allowed_classes:
            raise ValueError(
                f"Refusing to load disallowed model {import_path}.{model_class_name}. "
                f"Permitted: {ALLOWED_MODEL_IMPORTS}"
            )
        module = importlib.import_module(import_path)  # nosec B403 - module/class allow-listed above
        ModelClass = getattr(module, model_class_name)


        # Earth2Studio models follow the AutoModelMixin pattern:
        #   package = ModelClass.load_default_package()
        #   model = ModelClass.load_model(package)
        logger.info(f"Downloading model package for {model_class_name}...")
        package = ModelClass.load_default_package()
        model = ModelClass.load_model(package)
        logger.info(f"{model_class_name} loaded successfully")

        # Move model to device — Earth2Studio models are torch.nn.Module subclasses
        model = model.to(device)

        # Initialize GFS data source for fetching initial conditions
        # GFS is freely available without API keys
        data_source = None
        try:
            from earth2studio.data import GFS
            data_source = GFS()
            logger.info("GFS data source initialized for initial conditions")
        except Exception as e:
            logger.warning(f"GFS data source unavailable: {e}")
            logger.warning("Will attempt to use provided input data in requests")

        return {
            "model": model,
            "model_class_name": model_class_name,
            "config": model_config,
            "device": device,
            "data_source": data_source,
        }

    except ImportError as e:
        logger.error(f"Failed to import {import_path}.{model_class_name}: {e}")
        logger.error(
            "Ensure earth2studio is installed with the correct extras. "
            "Check model_code/requirements.txt"
        )
        raise
    except Exception as e:
        logger.error(f"Failed to load model: {traceback.format_exc()}")
        raise


# ============================================================
# INPUT DESERIALIZATION
# ============================================================

def input_fn(request_body, request_content_type):
    """
    Deserialize the inference request.

    Accepts JSON with the following fields:
    {
        "date": "2025-06-01T00:00:00",   # Initial condition date (ISO format)
        "lead_time_hours": 24,             # Forecast length in hours (default: 24)
        "variables": ["t2m", "z500"],      # Optional: subset of variables to return
    }

    If "date" is omitted, defaults to the most recent GFS analysis cycle
    (rounded down to the nearest 6-hour mark).

    Args:
        request_body: Raw request bytes
        request_content_type: MIME type

    Returns:
        Parsed request dictionary
    """
    if request_content_type == "application/json":
        request = json.loads(request_body)

        # Validate and set defaults
        if "date" not in request:
            # Default to most recent GFS cycle (rounded down to nearest 6h)
            now = datetime.utcnow()
            cycle_hour = (now.hour // 6) * 6
            request["date"] = now.replace(
                hour=cycle_hour, minute=0, second=0, microsecond=0
            ).strftime("%Y-%m-%dT%H:%M:%S")

        request.setdefault("lead_time_hours", 24)

        # Validate lead time
        lead_hours = request["lead_time_hours"]
        if not isinstance(lead_hours, (int, float)) or lead_hours <= 0:
            raise ValueError(f"lead_time_hours must be a positive number, got: {lead_hours}")
        if lead_hours > 240:
            logger.warning(
                f"Lead time {lead_hours}h exceeds recommended max of 240h (10 days). "
                "Forecast quality degrades at longer lead times."
            )

        logger.info(f"Request: date={request['date']}, lead_time={lead_hours}h")
        return request

    else:
        raise ValueError(
            f"Unsupported content type: {request_content_type}. "
            f"Use 'application/json'."
        )


# ============================================================
# PREDICTION
# ============================================================

def predict_fn(request, model_dict):
    """
    Run the weather forecast using Earth2Studio's deterministic run pipeline.

    The earth2studio.run.deterministic function:
    1. Fetches initial conditions from the data source (GFS by default)
    2. Runs the prognostic model forward in autoregressive time steps
    3. Writes output to an IO backend (ZarrBackend in-memory)

    Verified against earth2studio/run.py deterministic() signature:
        deterministic(
            time: list[str] | list[datetime] | list[np.datetime64],
            nsteps: int,
            prognostic: PrognosticModel,
            data: DataSource,
            io: IOBackend,
            output_coords: CoordSystem = OrderedDict({}),
            device: torch.device | None = None,
            verbose: bool = True,
        ) -> IOBackend

    Args:
        request: Parsed request dictionary from input_fn
        model_dict: Model dictionary from model_fn

    Returns:
        Dictionary with forecast results
    """
    model = model_dict["model"]
    model_class_name = model_dict["model_class_name"]
    device = model_dict["device"]
    data_source = model_dict.get("data_source")

    date_str = request["date"]
    lead_time_hours = int(request["lead_time_hours"])
    requested_vars = request.get("variables")

    logger.info(f"Running {model_class_name} forecast: date={date_str}, lead={lead_time_hours}h")

    if data_source is None:
        return {
            "status": "error",
            "error": (
                "No data source available. GFS initialization failed. "
                "Ensure the SageMaker container has internet access for "
                "downloading GFS initial conditions from NOAA."
            ),
            "model": model_class_name,
            "date": date_str,
        }

    try:
        from earth2studio.run import deterministic as run_deterministic
        from earth2studio.io import ZarrBackend

        # Determine time step from model
        # Earth2Studio models report time_step via output_coords lead_time
        # input_coords lead_time is typically [0h] (the initial state),
        # so we need output_coords to get the actual forecast step size.
        try:
            input_coords = model.input_coords()
            output_coords = model.output_coords(input_coords)
            lead_time_coord = output_coords.get("lead_time", np.array([np.timedelta64(6, "h")]))
            if len(lead_time_coord) > 0:
                step_td = lead_time_coord[-1]
                if isinstance(step_td, np.timedelta64):
                    step_hours = float(step_td / np.timedelta64(1, "h"))
                else:
                    step_hours = 6.0
            else:
                step_hours = 6.0
        except Exception:
            step_hours = 6.0

        # Safeguard against zero or negative step_hours
        if step_hours <= 0:
            step_hours = 6.0

        n_steps = max(1, int(lead_time_hours / step_hours))
        logger.info(f"Forecast: {n_steps} steps × {step_hours}h = {n_steps * step_hours}h")

        # Create in-memory Zarr backend for output
        # ZarrBackend() with no file_name creates a MemoryStore
        io_backend = ZarrBackend()

        # Run deterministic forecast
        # time parameter is a list of date strings — matches the README examples:
        #   run(["2025-01-01T00:00:00"], 10, model, data, io)
        io_result = run_deterministic(
            time=[date_str],
            nsteps=n_steps,
            prognostic=model,
            data=data_source,
            io=io_backend,
            device=device,
            verbose=False,  # Suppress tqdm in SageMaker container
        )

        # Extract results from Zarr backend
        # io_result.root is a zarr.Group; io_result.coords has the coordinate system
        result = {
            "status": "success",
            "model": model_class_name,
            "init_time": date_str,
            "lead_time_hours": lead_time_hours,
            "step_hours": step_hours,
            "n_steps": n_steps,
            "forecasts": {},
        }

        # Read coordinates from the IO backend
        coords = io_result.coords
        lead_times = coords.get("lead_time", np.array([]))
        variables = coords.get("variable", np.array([]))

        # Build valid times list
        init_dt = np.datetime64(date_str)
        result["valid_times"] = [
            str(init_dt + lt) for lt in lead_times
        ] if len(lead_times) > 0 else []

        # Extract variable data from the zarr root group
        # The ZarrBackend stores arrays keyed by variable name
        # It also stores coordinate arrays (time, lead_time, lat, lon) —
        # we skip those and only process numeric weather variables.
        coord_names = {"time", "lead_time", "lat", "lon", "latitude", "longitude", "batch", "face"}
        for arr_name in io_result.root:
            # Skip coordinate arrays
            if arr_name in coord_names:
                continue

            arr = io_result.root[arr_name]
            data = np.array(arr)

            # Skip non-numeric arrays (e.g., timedelta, datetime, string)
            if not np.issubdtype(data.dtype, np.number):
                continue

            # Filter to requested variables if specified
            if requested_vars and arr_name not in requested_vars:
                continue

            # Compute summary statistics (avoid sending full grids over HTTP)
            var_result = {
                "shape": list(data.shape),
                "dtype": str(data.dtype),
                "min": float(np.nanmin(data)),
                "max": float(np.nanmax(data)),
                "mean": float(np.nanmean(data)),
                "std": float(np.nanstd(data)),
                "unit": _get_variable_unit(arr_name),
                # Sample: last time step, flattened, first 50 values
                "data_sample": data.flat[-min(50, data.size):].tolist(),
            }

            # If return_grid is requested, include downsampled 2D grids for ALL
            # time steps (for animated map visualization). Downsample by factor
            # of 8 to keep response size manageable (~90x180 per step per var).
            if request.get("return_grid"):
                step_ds = 8  # downsample factor
                all_step_grids = []

                if data.ndim >= 4:
                    # Shape: (batch, time_steps, lat, lon) — typical
                    n_time = data.shape[1]
                    ref_2d = data[0, 0]
                    for t in range(n_time):
                        frame = data[0, t]
                        all_step_grids.append(frame[::step_ds, ::step_ds].tolist())
                elif data.ndim == 3:
                    # Shape: (time_steps, lat, lon)
                    n_time = data.shape[0]
                    ref_2d = data[0]
                    for t in range(n_time):
                        all_step_grids.append(data[t, ::step_ds, ::step_ds].tolist())
                elif data.ndim == 2:
                    # Shape: (lat, lon) — single step
                    ref_2d = data
                    all_step_grids.append(data[::step_ds, ::step_ds].tolist())
                else:
                    ref_2d = data
                    all_step_grids.append(data.tolist())

                # Multi-step grids: list of 2D arrays, one per timestep
                var_result["grids"] = all_step_grids
                # Also keep "grid" as the last step for backward compatibility
                var_result["grid"] = all_step_grids[-1] if all_step_grids else []
                var_result["grid_lat"] = np.linspace(90, -90, ref_2d.shape[-2], endpoint=True)[::step_ds].tolist() if ref_2d.ndim >= 2 else []
                var_result["grid_lon"] = np.linspace(0, 360, ref_2d.shape[-1], endpoint=False)[::step_ds].tolist() if ref_2d.ndim >= 1 else []

            result["forecasts"][arr_name] = var_result

        # ── Save full-resolution data to S3 as compressed NumPy archive ──
        # This preserves the original 721×1440 (or model-native) resolution
        # for scientists to download, while the JSON response only has downsampled grids.
        try:
            import uuid as _uuid
            full_res_data = {}
            for arr_name in io_result.root:
                if arr_name in coord_names:
                    continue
                arr_data = np.array(io_result.root[arr_name])
                if not np.issubdtype(arr_data.dtype, np.number):
                    continue
                if requested_vars and arr_name not in requested_vars:
                    continue
                full_res_data[arr_name] = arr_data.astype(np.float32)

            if full_res_data:
                # Also save coordinate arrays
                for cname in ["lat", "lon", "latitude", "longitude"]:
                    if cname in io_result.root:
                        cdata = np.array(io_result.root[cname])
                        if np.issubdtype(cdata.dtype, np.number):
                            full_res_data[f"_coord_{cname}"] = cdata

                # Save valid_times as metadata
                full_res_data["_valid_times"] = np.array(result["valid_times"], dtype="U")
                full_res_data["_model"] = np.array([model_class_name], dtype="U")
                full_res_data["_init_time"] = np.array([date_str], dtype="U")

                # Compress and upload to S3
                buf = BytesIO()
                np.savez_compressed(buf, **full_res_data)
                buf.seek(0)

                forecast_id = str(_uuid.uuid4())[:8]
                s3_bucket = os.environ.get("S3_BUCKET")
                if not s3_bucket:
                    logger.info("S3_BUCKET not set — skipping full-res upload.")
                    raise RuntimeError("S3_BUCKET not configured")
                s3_key = f"earth2-weather-models/full-res/{date_str[:10]}_{model_class_name}_{forecast_id}.npz"

                import boto3 as _boto3
                _s3 = _boto3.client("s3", region_name=os.environ.get("AWS_REGION"))
                _s3.put_object(Bucket=s3_bucket, Key=s3_key, Body=buf.getvalue(), ContentType="application/octet-stream")

                result["full_res_s3"] = f"s3://{s3_bucket}/{s3_key}"
                result["full_res_size_mb"] = round(buf.tell() / 1024 / 1024, 1)
                logger.info(f"Full-res data saved to s3://{s3_bucket}/{s3_key} ({result['full_res_size_mb']} MB)")
        except Exception as e:
            logger.warning(f"Failed to save full-res data to S3: {e}")
            # Non-fatal — the downsampled response still works

        logger.info(
            f"Forecast complete: {len(result['forecasts'])} variables, "
            f"{n_steps} steps, valid times: {len(result['valid_times'])}"
        )
        return result

    except Exception as e:
        logger.error(f"Prediction failed: {traceback.format_exc()}")
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "model": model_class_name,
            "date": date_str,
        }


# ============================================================
# OUTPUT SERIALIZATION
# ============================================================

def output_fn(prediction, accept):
    """
    Serialize the forecast results to JSON.

    Args:
        prediction: Dictionary from predict_fn
        accept: Requested response MIME type

    Returns:
        Tuple of (serialized response body, content type)
    """
    if accept == "application/json" or accept == "*/*":
        return json.dumps(prediction, default=str), "application/json"
    else:
        raise ValueError(f"Unsupported accept type: {accept}. Use 'application/json'.")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _get_variable_unit(var_name):
    """Map weather variable names to their physical units."""
    units = {
        # Temperature (Kelvin)
        "t2m": "K", "t850": "K", "t500": "K", "t1000": "K", "t250": "K",
        "t925": "K", "t700": "K", "t600": "K", "t400": "K", "t300": "K",
        "t200": "K", "t150": "K", "t100": "K", "t50": "K",
        # Wind (m/s)
        "u10m": "m/s", "v10m": "m/s", "u100m": "m/s", "v100m": "m/s",
        "u850": "m/s", "v850": "m/s", "u500": "m/s", "v500": "m/s",
        "u1000": "m/s", "v1000": "m/s", "u250": "m/s", "v250": "m/s",
        "u925": "m/s", "v925": "m/s", "u700": "m/s", "v700": "m/s",
        "u600": "m/s", "v600": "m/s", "u400": "m/s", "v400": "m/s",
        "u300": "m/s", "v300": "m/s", "u200": "m/s", "v200": "m/s",
        "u150": "m/s", "v150": "m/s", "u100": "m/s", "v100": "m/s",
        "u50": "m/s", "v50": "m/s",
        # Pressure (Pa)
        "sp": "Pa", "msl": "Pa",
        # Geopotential (m²/s²)
        "z500": "m²/s²", "z300": "m²/s²", "z700": "m²/s²",
        "z1000": "m²/s²", "z50": "m²/s²", "z850": "m²/s²",
        "z250": "m²/s²", "z925": "m²/s²", "z600": "m²/s²",
        "z400": "m²/s²", "z200": "m²/s²", "z150": "m²/s²",
        "z100": "m²/s²",
        # Moisture
        "tcwv": "kg/m²",
        "r500": "%", "r850": "%",
        "q1000": "kg/kg", "q925": "kg/kg", "q850": "kg/kg", "q700": "kg/kg",
        "q600": "kg/kg", "q500": "kg/kg", "q400": "kg/kg", "q300": "kg/kg",
        "q250": "kg/kg", "q200": "kg/kg", "q150": "kg/kg", "q100": "kg/kg",
        "q50": "kg/kg",
    }
    return units.get(var_name, "unknown")
