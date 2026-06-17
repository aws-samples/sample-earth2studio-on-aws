# Security deep dive

[← Back to the main README](../README.md)

This document expands on the [Security](../README.md#security) summary in the main README. It covers the static-analysis findings that are fixed in code, and the known production-hardening gaps that are deliberately left for you to resolve when you fork.

## Code-level hardening

The repo addresses every high-severity finding reported by static-analysis tooling (Bandit, Semgrep, Checkov, ASH). Each finding below is either fixed in code or documented inline next to the relevant line so a future scanner run is auditable:

| Finding | File(s) | Resolution |
|---|---|---|
| **B201 / debug-enabled** — Flask `debug=True` in dev server | `backend/local_server.py` | Debug is **off by default**. Override via `FLASK_DEBUG=1` only when you need the Werkzeug debugger. |
| **B104 / avoid_app_run_with_bad_host** — Flask bound to `0.0.0.0` | `backend/local_server.py` | Defaults to `127.0.0.1`. Override via `FLASK_HOST=0.0.0.0` if you genuinely need to expose the dev server on a trusted LAN. |
| **B104** — Flask bound to `0.0.0.0` (BYOC dev fallback) | `sagemaker_deploy/container_fcn3*/sagemaker_serve.py` | The `__main__` blocks (developer-only) default to `127.0.0.1`. The production path — `serve` script + gunicorn on `0.0.0.0:8080` — is **unchanged** because the SageMaker BYOC contract requires it. |
| **B108** — `tempfile.mkdtemp(prefix=…)` flagged as insecure tempdir | `sagemaker_deploy/deploy.py` | Documented inline with `# nosec B108`: `tempfile.mkdtemp` is the secure modern API (atomic creation, mode `0o700`, OS-default temp location). The Bandit B108 rule is a regex pattern that cannot tell `mkdtemp()` already mitigates the risk it's looking for. |
| **`non-literal-import`** (Semgrep, `python.lang.security.audit.non-literal-import`) on `importlib.import_module(import_path)` | `sagemaker_deploy/model_code_dlwp/inference.py`, `sagemaker_deploy/model_code_fcn3/inference.py` | **Fixed.** Each handler now defines a hard-coded `ALLOWED_MODEL_IMPORTS = {"earth2studio.models.px": {"DLWP"}}` (or `{"FCN3"}`) and validates the `(module, class)` pair *before* calling `importlib.import_module`. Anything outside the allow-list raises `ValueError` and refuses the load — even if a future change to `model_config.json` introduced an unintended class. |
| **CKV_DOCKER_2** — missing Dockerfile `HEALTHCHECK` | `sagemaker_deploy/container_fcn3/Dockerfile`, `sagemaker_deploy/container_fcn3_dlc/Dockerfile` | Both Dockerfiles include `HEALTHCHECK CMD curl -fsS http://127.0.0.1:8080/ping`, with a `--start-period=1500s` to allow the first-call NVIDIA model-weight download to complete. |

## Known gaps you must address before going to production

The findings below are **deliberately left in the sample** because the fix is environment-specific and would either break the prototype or hide a decision you should make consciously when you fork. Each one is a real production hardening item — please do not ship to a regulated workload without resolving them:

1. **CKV_DOCKER_3 / `dockerfile.security.missing-user-entrypoint`** — both BYOC containers run as `root`.
   - **Why this isn't fixed in the sample**:
     - `nvcr.io/nvidia/pytorch:25.12-py3` (the NGC base used by `container_fcn3/Dockerfile`) is published with a root-only contract: CUDA driver mounts, NCCL, `torch-harmonics` CUDA-extension build outputs, and the `EARTH2STUDIO_CACHE` weight-download path all assume root ownership. NVIDIA does not currently publish a non-root variant.
     - `pytorch-training:2.7.1-gpu-py312-cu128-ubuntu22.04-sagemaker` (the AWS Training DLC base used by `container_fcn3_dlc/Dockerfile`) inherits the same expectation. SageMaker BYOC inference containers historically run as root because of `/opt/ml/model`, `/opt/ml/input`, and `/opt/ml/output` mount-permission semantics.
   - **Compensating controls already in place**: the containers are deployed only as **SageMaker BYOC inference endpoints**, which run inside AWS-managed isolated container runtimes with no host shell; the endpoints sit behind IAM-authenticated `InvokeEndpointAsync` calls; no public ingress reaches the container; and the inference handler does not exec sub-processes from request input.
   - **What to do in your fork**: add a non-root user to both Dockerfiles and validate end-to-end inference under that UID. Sketch:

     ```dockerfile
     # near the end of the Dockerfile, after all `pip install` / build steps:
     RUN groupadd --system --gid 1000 earth2 \
      && useradd  --system --uid 1000 --gid 1000 --home-dir /home/earth2 --create-home earth2 \
      && mkdir -p /opt/ml/model /opt/ml/input /opt/ml/output /tmp/earth2_cache \
      && chown -R earth2:earth2 /opt/ml /opt/program /tmp/earth2_cache /home/earth2
     USER earth2
     ENV HOME=/home/earth2
     ```

     Then redeploy and confirm `/ping` returns 200 and a real forecast still completes — the failure modes are usually missing write permission on `/tmp/earth2_cache` (NVIDIA registry weight download) or `/opt/ml/output` (SageMaker async output). Consider also setting CodeBuild's `--isolation-mode` and adding `securityContext` constraints once you wrap this in EKS/ECS.

2. **Container `USER` is also surfaced as `dockerfile.security.missing-user-entrypoint` by Semgrep** — this is the same finding under a different rule ID. The same remediation applies.

3. **OS package and base-image freshness** — the published Dockerfiles pin to `pytorch:25.12-py3` (NGC) and `pytorch-training:2.7.1-…` (AWS DLC) at the time of writing. Periodically rebuild against the latest patched tag and rerun the SAST/SCA scan; CVE remediation in the base image is the customer's responsibility once you fork.
