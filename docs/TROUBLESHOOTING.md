# Troubleshooting

[← Back to the main README](../README.md)

Step references below (e.g. "Step 6", "Step 7") point to the [Walkthrough](../README.md#walkthrough) in the main README.

| Symptom | Cause | Fix |
|---|---|---|
| `UnresolvedAccount` from `cdk deploy` | No credentials or region resolved | Run `aws sts get-caller-identity && aws configure get region` to verify both. Then `eval "$(aws configure export-credentials --format env)"` and retry. |
| `S3 bucket for model artifacts is not configured` | SSM params not populated | Deploy `Earth2SageMaker` first. |
| `Could not access model data` (SageMaker) | IAM role missing S3 permissions | Re-deploy `Earth2SageMaker`. The role grants are scoped by SSM-published prefix. |
| `Endpoint config already exists` | Stale resource from a failed deploy | `aws sagemaker delete-endpoint-config --endpoint-config-name <name>` |
| FCN3 inference returns CUDA kernel error | Container missing Blackwell support | Rebuild via Step 8 — make sure you're using `container_fcn3/` (NGC) or `container_fcn3_dlc/`, NOT the AWS PyTorch DLC. |
| `User pool client ... does not exist` in browser | Stale `frontend/.env.local` | Update with the IDs from the latest `Earth2UI` outputs and rerun Step 6 Pass 2. |
| `User does not exist` on sign-in | Self-signup is disabled — that user was never created | Run the `admin-create-user` + `admin-set-user-password` commands from Step 7. |
| `Username should be an email` from `admin-create-user` | The pool uses `UsernameAttributes=email`, but the email wasn't recognized | Pass the email as `--username` AND include `Name=email,Value=<email>` in `--user-attributes`. The example in Step 7 is correct. |
| `NEW_PASSWORD_REQUIRED` on sign-in | The password is still in temporary state from `admin-create-user` | Run `admin-set-user-password ... --permanent` to skip the first-time reset. |
| DLWP endpoint reports `InService` but every `/ping` returns 500 | `nvidia-physicsnemo` 2.x imports `warp.context`, which `warp-lang` ≥ 1.10 removed | Already fixed in `model_code_dlwp/requirements.txt` with version pins. If you forked an older copy, pin: `earth2studio<0.14.0`, `nvidia-physicsnemo<2.0.0`, `warp-lang<1.10.0`. |
| FCN3 forecast times out at 600 s | Model weights cold-cache download from NVIDIA registry | First-call latency is ~1–3 min. Re-issue the call; the second one will be in the 30-second range. |
