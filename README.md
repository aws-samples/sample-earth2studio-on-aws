# Run your own global AI weather forecast on AWS with NVIDIA Earth2Studio, Amazon SageMaker, and AWS CDK

> **Source repository:** <https://github.com/aws-samples/sample-earth2studio-on-aws>

Accurate global weather forecasts sit underneath almost every part of the modern economy — aviation routing, energy trading, grid operations, agriculture, shipping, emergency response, insurance pricing, and the morning news all depend on knowing what the atmosphere will look like six hours, one day, or ten days from now. For roughly 50 years, the only way to produce that forecast was **Numerical Weather Prediction (NWP)** — discretizing the atmosphere into a 3-D grid and numerically solving the conservation equations of mass, momentum, and energy at every grid cell, every few seconds of simulated time, for ten simulated days. Operational NWP models such as NOAA’s **GFS**, ECMWF’s **IFS**, and the UK Met Office’s **UM** routinely cost tens of thousands of CPU-hours per forecast cycle and run on dedicated supercomputers that cost national meteorological centers hundreds of millions of dollars per year.

Then, in late 2022, a series of papers — **FourCastNet** (NVIDIA), **Pangu-Weather** (Huawei), **GraphCast** (Google DeepMind), and **FuXi** (Fudan University) — showed that machine-learning models trained on **ERA5**, a 40-year reanalysis archive of past weather, could produce 10-day global forecasts of comparable or *better* skill than traditional NWP, in **under a minute**, on a **single GPU**. Some of these models now match or beat ECMWF IFS-HRES — the world’s most accurate operational physics model — on standard skill scores like RMSE of 500 hPa geopotential height.

The catch is that these models live as PyTorch checkpoints scattered across NVIDIA NGC, GitHub, and Hugging Face, with finicky CUDA dependencies (`torch-harmonics`, `makani`, `physicsnemo`), a hard dependency on real-time initial-condition data (typically NOAA’s GFS analysis), and varying licenses. **Running them in production is non-trivial.**

In this post, we close that gap. We walk through a turnkey reference architecture — fully captured as Infrastructure as Code with the [AWS Cloud Development Kit (AWS CDK)](https://aws.amazon.com/cdk/) — that serves two open-source AI weather models from the [NVIDIA Earth2Studio](https://github.com/NVIDIA/earth2studio) framework on [Amazon SageMaker](https://aws.amazon.com/sagemaker/) asynchronous endpoints, behind a secure single-page web application protected by [Amazon CloudFront](https://aws.amazon.com/cloudfront/), [AWS WAF](https://aws.amazon.com/waf/), and [Amazon Cognito](https://aws.amazon.com/cognito/). The entire deployment runs in your own AWS account and goes from **`git clone` to first forecast in well under an hour**. Both models are licensed under Apache-2.0, and the only manual step is providing AWS credentials.

If you’ve ever wondered *“could I run my own weather forecast?”* — yes, you can. This post (and its accompanying repository) shows how.

---

## Table of contents

1. [What you’ll build](#what-youll-build)
2. [Solution overview](#solution-overview)
3. [The two models, explained](#the-two-models-explained)
4. [How to read the forecast output](#how-to-read-the-forecast-output)
5. [Costs and performance](#costs-and-performance)
6. [Prerequisites](#prerequisites)
7. [Walkthrough](#walkthrough)
   - [Step 1 — Configure AWS credentials and region](#step-1--configure-aws-credentials-and-region)
   - [Step 2 — Clone the repository and install dependencies](#step-2--clone-the-repository-and-install-dependencies)
   - [Step 3 — Run `setup.sh`](#step-3--run-setupsh)
   - [Step 4 — Bootstrap AWS CDK](#step-4--bootstrap-aws-cdk)
   - [Step 5 — Deploy the SageMaker infrastructure stack](#step-5--deploy-the-sagemaker-infrastructure-stack)
   - [Step 6 — Build and deploy the frontend](#step-6--build-and-deploy-the-frontend)
   - [Step 7 — Create the first user and sign in](#step-7--create-the-first-user-and-sign-in)
   - [Step 8 — (Optional) Build the FCN3 BYOC container](#step-8--optional-build-the-fcn3-byoc-container)
   - [Step 9 — Deploy SageMaker endpoints](#step-9--deploy-sagemaker-endpoints)
   - [Step 10 — Run a forecast](#step-10--run-a-forecast)
8. [Configuration reference](#configuration-reference)
9. [Project structure](#project-structure)
10. [Security](#security)
11. [Cleanup](#cleanup)
12. [Local development](#local-development)
13. [Troubleshooting](#troubleshooting)
14. [Conclusion](#conclusion)
15. [License and model attribution](#license-and-model-attribution)
16. [Further reading](#further-reading)

---

## What you’ll build

By the end of this walkthrough you will have, **in your own AWS account**, a complete, production-style global weather-forecasting platform with three layers:

1. **An inference layer** — two prognostic ML models, each on its own SageMaker asynchronous-inference endpoint:
   - **DLWP** on `ml.g5.2xlarge` (24 GB A10G GPU, ~$1.52/hr) — a fast, low-resolution baseline.
   - **FCN3** (FourCastNet v3) on `ml.g7e.2xlarge` (96 GB Blackwell RTX PRO 6000 GPU, ~$2.80/hr) — operational-grade, 0.25° global, 72 atmospheric variables.
   - Both endpoints automatically fetch initial conditions from NOAA’s GFS analysis (no API key required) and return both a downsampled JSON summary for the UI and the full-resolution NumPy archive on Amazon S3 for scientists to download.
2. **An application layer** — a secured Web UI:
   - Amazon CloudFront distribution with TLS 1.2 minimum.
   - Amazon S3 for the React/Vite SPA assets.
   - Amazon API Gateway REST API protected by AWS WAFv2 (managed rules + per-IP rate limiting).
   - AWS Lambda (Python 3.13) brokering all SageMaker calls.
   - Amazon Cognito for user authentication (email + password, SRP-only, no MFA, **self-signup disabled** — admins create users via the AWS Command Line Interface).
3. **An infrastructure layer** — two AWS CDK stacks that pass [CDK Nag](https://github.com/cdklabs/cdk-nag)’s `AwsSolutions` rule pack:
   - **`Earth2SageMaker`**: S3 bucket (with lifecycle cleanup), least-privilege AWS Identity and Access Management (IAM) execution role, Amazon Elastic Container Registry (Amazon ECR) repository, AWS CodeBuild project, and AWS Systems Manager Parameter Store entries that everything else auto-discovers.
   - **`Earth2UI`**: Cognito User Pool, AWS WAFv2 Web ACL, S3 SPA bucket, API Gateway, Lambda function, and the CloudFront distribution.

A small `setup.sh` script does the only manual step: storing the NVIDIA NGC API key in AWS Secrets Manager so AWS CodeBuild can `docker login nvcr.io` and pull the FCN3 base image.

A user signs in to the SPA, picks a model and an initial date, chooses which variables to plot (for example `t2m`, `z500`, `msl`), clicks **Run forecast**, and 10–60 seconds later sees animated maps of the forecast.

The screenshot below shows the web UI (the React single-page application served by CloudFront) after a successful FCN3 forecast run — the sidebar on the left drives the model, init date, lead time, and variable selection, while the main panel renders the animated global forecast maps and per-variable statistics:

![Earth2Studio Weather Forecast Platform — frontend web UI screenshot (demo)](docs/UI_Screenshot.png)

> *Demo screenshot — included for illustration only. The exact look-and-feel of your deployment may evolve as you customize the React/Vite frontend in `frontend/`.*

> **The pitch in one sentence:** *“Run `cdk deploy` and `./deploy-all.sh`, wait fifteen minutes, and you have a global AI weather-forecasting service running in your own AWS account, behind your own login, with end-to-end encryption, AWS WAF protection, and a clean web UI to drive it.”*

**Total time to deploy from a fresh AWS account: ~40 minutes**, of which roughly 30 are AWS provisioning (CloudFront propagation + GPU endpoint warm-up).

---

## Solution overview

The following diagram shows the end-to-end architecture. The full editable diagram is available at [`docs/architecture.drawio`](docs/architecture.drawio); a rendered PNG is embedded below.

![Earth2Studio Weather Forecast Platform on AWS — architecture diagram](docs/architecture.png)

The request path, end to end:

1. A user opens the SPA URL in the browser. CloudFront serves the React bundle from S3 over TLS 1.2+.
2. When the user clicks **Run forecast**, the SPA sends the call to `https://<cloudfront>/api/*`. CloudFront forwards `/api/*` to API Gateway.
3. AWS WAFv2 evaluates the request against AWS managed rule groups and a per-IP rate limit. API Gateway then enforces the Cognito authorizer, which validates the user’s SRP-issued JWT.
4. API Gateway invokes the Lambda function. Lambda reads the SageMaker endpoint name and S3 prefix from SSM Parameter Store and calls `InvokeEndpointAsync` against the appropriate SageMaker endpoint.
5. The SageMaker endpoint pulls initial conditions from NOAA’s public GFS analysis, runs the model forward, writes the full-resolution NumPy archive to the S3 model bucket, and returns a downsampled JSON summary.
6. Lambda returns the JSON to API Gateway, which returns it through CloudFront to the SPA. The SPA renders an animated map.

The architecture uses a **hybrid lifecycle**. The presentation layer (CloudFront, Cognito, Lambda, S3 SPA, API Gateway, AWS WAFv2) costs only ~$5–10/month and is **always on** because tearing it down would invalidate the SPA URL and any users you’ve created. GPU endpoints, by contrast, cost real money — between $1.52/hr and $2.80/hr each — so they have to be **ephemeral**: created on demand and deleted when you’re done. AWS CDK manages the always-on layer; the `deploy-all.sh` shell script in `sagemaker_deploy/` manages the ephemeral endpoints.

---

## The two models, explained

The repository ships only **Apache-2.0-licensed** weather models. Other Earth2Studio models exist (Pangu-Weather, GraphCast, FuXi, etc.) but carry research-only or non-commercial licenses that are incompatible with publishing this repo as a permissive sample. If you need them in your own fork, double-check each model’s license against your use case first.

### DLWP — Deep Learning Weather Prediction

| Property | Value |
|---|---|
| Origin | University of Washington (Karlbauer et al., 2023) |
| License | Apache-2.0 |
| Architecture | Convolutional neural network on a **cubed-sphere** grid (six face-projections of Earth onto a cube; convolutions wrap across face boundaries) |
| Native resolution | **1.0°** (≈ 111 km at equator) |
| Variables | Limited dynamic set — primarily `t2m`, `z500` |
| Time step | 6 hours |
| GPU memory | ~4 GB |
| Instance | `ml.g5.2xlarge` (NVIDIA A10G) |
| Cost | **~$1.52/hr** |
| Forecast wall time (24 h lead) | **~10 seconds** |

**When to use DLWP**: rapid prototyping, baseline runs, smoke-testing pipelines, and education. It’s a small, fast model — ideal for *“I just want a global temperature map for tomorrow”* without paying for a Blackwell GPU.

### FCN3 — FourCastNet v3

| Property | Value |
|---|---|
| Origin | NVIDIA (Pathak et al., 2024) |
| License | Apache-2.0 |
| Architecture | **Spherical Fourier Neural Operator** with a probabilistic ensemble head (uses the spherical-harmonic transform via `torch-harmonics` to handle the sphere properly, unlike rectangular CNNs) |
| Native resolution | **0.25°** (≈ 28 km at equator), 721×1440 grid |
| Variables | **72** — see [How to read the forecast output](#how-to-read-the-forecast-output) for the full list |
| Time step | 6 hours |
| GPU memory | ~80 GB (needs a Blackwell-class GPU) |
| Instance | `ml.g7e.2xlarge` (1× RTX PRO 6000 Blackwell, 96 GB VRAM) |
| Cost | **~$2.80/hr** |
| Forecast wall time (24 h lead) | **~30 seconds** |

**When to use FCN3**: this is the production model. 0.25° matches operational NWP centers; the 72-variable output covers surface fields plus a full 3-D atmosphere on 13 pressure levels, suitable for deriving thousands of downstream products (jet streams, cyclone tracking, energy generation forecasts, agricultural risk, etc.). FCN3 needs a custom container because:

1. PyTorch 2.6 (the latest SageMaker inference DLC) doesn’t support Blackwell GPUs (compute capability `sm_120`).
2. FCN3’s `torch-harmonics` extension must be **CUDA-compiled** for both Ada Lovelace (`sm_89`) and Blackwell (`sm_120`) to get the fast path.
3. Stock SageMaker inference DLCs ≥ 2.7 don’t exist (TorchServe is in maintenance mode), so we ship two **bring-your-own-container** (BYOC) variants:
   - **`container_fcn3/`** (NGC base, pre-compiled CUDA `torch-harmonics`) — fastest, requires NGC API key.
   - **`container_fcn3_dlc/`** (AWS Training DLC base, PyPI `torch-harmonics`) — no NGC key, slightly slower in float32 fallback.

Both BYOC variants use Flask + gunicorn on port 8080 to implement the SageMaker `/ping` + `/invocations` contract.

### How the two models compare on a real forecast

For an init time of `2026-05-19T00:00:00 UTC` and a +24 h forecast:

| Metric | DLWP | FCN3 |
|---|---|---|
| Wall-clock | 10 sec | 30 sec |
| Output variables | 2 (`t2m`, `z500`) | 72 |
| Grid (lat × lon) | 721 × 1440 (resampled from native 1°) | 721 × 1440 (native) |
| Global mean `t2m` | 6.94 °C | 6.80 °C |
| Global mean `z500` | 5437 dam | 5437 dam |
| `msl` min / max | n/a (not in DLWP) | 949 / 1069 hPa |

The two models agree on shared variables to better than 1 % at the global mean — both are in the realistic physical range. DLWP’s strength is speed; FCN3’s is breadth and resolution.

---

## How to read the forecast output

Every weather model in this repo speaks in terms of **prognostic variables** — atmospheric quantities that describe state at a given moment. If you’ve never worked with weather data before, here’s a quick guide.

### Surface variables (one value per grid cell)

| Symbol | Name | Unit | Typical range | Use |
|---|---|---|---|---|
| `t2m` | 2-meter air temperature | K (display: °C) | −80 °C polar … +50 °C desert | the “temperature” you see on weather apps |
| `msl` | Mean sea-level pressure | Pa (display: hPa) | 950 (deep low) … 1050 (strong high) | drives all weather; storm centers are low-pressure |
| `u10m` / `v10m` | 10-m wind components (east / north) | m/s | typically ±30 m/s | surface wind |
| `u100m` / `v100m` | 100-m wind components | m/s | similar | wind-energy applications |
| `tcwv` | Total column water vapor | kg/m² | 0 (poles) … 70 (tropics) | atmospheric “fuel” for storms |

### Pressure-level variables (one value per cell, **per pressure level**)

These describe a 3-D slice of the atmosphere. Pressure decreases with height: surface ≈ 1000 hPa, 500 hPa is roughly 5.5 km up, 50 hPa is in the stratosphere. FCN3 outputs 13 levels: **50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 hPa**.

| Symbol | Name | Unit | Use |
|---|---|---|---|
| `z<level>` | Geopotential at pressure level | m²/s² (display: dam, divide by 98.0665) | `z500` is the most-watched single field in synoptic meteorology — it tells you where the jet stream and big weather systems are |
| `t<level>` | Temperature at pressure level | K | thermal structure of the atmosphere |
| `u<level>` / `v<level>` | Wind components at pressure level | m/s | jet-stream winds (especially `u250`, `u300`) |
| `q<level>` | Specific humidity at pressure level | kg/kg | moisture available for precipitation |

### What a sample response looks like

After a successful FCN3 24-h forecast, the JSON response (truncated) contains:

```json
{
  "status": "success",
  "model": "FCN3",
  "init_time": "2026-05-19T00:00:00",
  "lead_time_hours": 24,
  "step_hours": 6,
  "n_steps": 4,
  "valid_times": [
    "2026-05-19T00:00:00",
    "2026-05-19T06:00:00",
    "2026-05-19T12:00:00",
    "2026-05-19T18:00:00",
    "2026-05-20T00:00:00"
  ],
  "forecasts": {
    "t2m":  {"shape": [1,5,721,1440], "min": -72.04, "mean": 6.80,  "max": 47.79, "unit": "K", "grids": [[...], ...]},
    "msl":  {"shape": [1,5,721,1440], "min": 94910,  "mean": 101100,"max": 106900,"unit": "Pa", "grids": [[...], ...]},
    "z500": {"shape": [1,5,721,1440], "min": 46830,  "mean": 54370, "max": 58350, "unit": "m²/s²", "grids": [[...], ...]}
  },
  "full_res_s3": "s3://earth2sagemaker-…/earth2-weather-models/full-res/2026-05-19_FCN3_a1b2c3d4.npz",
  "full_res_size_mb": 142.7
}
```

**Two things to know:**

1. `grids` is a **downsampled** array (factor of 8 along each axis) so the JSON stays small enough to transfer over HTTP. The frontend uses these for the animated maps.
2. `full_res_s3` is the path to a compressed NumPy archive containing the **native 0.25°** data for every variable and every timestep. This is the file scientists download for analysis (~140 MB per FCN3 forecast at 24 h lead).

You can grab the full-resolution archive locally and load it like this:

```python
import numpy as np, boto3
s3 = boto3.client("s3", region_name="us-west-2")
s3.download_file("earth2sagemaker-…", "earth2-weather-models/full-res/2026-05-19_FCN3_a1b2c3d4.npz", "/tmp/forecast.npz")
data = np.load("/tmp/forecast.npz", allow_pickle=False)
t2m = data["t2m"]              # shape (1, 5, 721, 1440)
print(t2m.mean(), t2m.shape)
```

---

## Costs and performance

There are two cost lines to track:

| Resource | Cost | Lifecycle |
|---|---|---|
| **CDK long-lived infra** (CloudFront, Lambda, S3, Cognito, AWS WAF, API Gateway) | **~$5–10 / month** total | Always on |
| **DLWP endpoint** (`ml.g5.2xlarge`) | **~$1.52 / hr** | On-demand only |
| **FCN3 endpoint** (`ml.g7e.2xlarge`) | **~$2.80 / hr** | On-demand only |
| **Combined (both endpoints up)** | **~$4.32 / hr ≈ $104 / day** | On-demand only |

> ⚠️ **GPU endpoints are billed per second they exist, not per inference call.** A SageMaker async endpoint with 0 % utilization still costs you the full hourly rate. Always run `./deploy-all.sh --delete` when you’re done experimenting.

Inference latency from a warm endpoint:

| Model | 24 h forecast (3 vars) | 240 h forecast (10 days, all vars) |
|---|---|---|
| DLWP | ~10 sec | ~30–60 sec |
| FCN3 | ~30 sec | ~3–5 min |

The first inference after endpoint creation is slower (~1–3 min) because the model weights are downloaded from NVIDIA’s registry into the container’s `/tmp` cache. Subsequent calls are fast.

---

## Prerequisites

You need the following installed locally **before** you start the walkthrough:

| Tool | Version | Install |
|---|---|---|
| AWS CLI v2 | latest | <https://aws.amazon.com/cli/> |
| Node.js | ≥ 18 | <https://nodejs.org/> |
| Python | ≥ 3.13 | <https://www.python.org/downloads/> |
| AWS CDK CLI | latest | `npm install -g aws-cdk` |
| Docker (optional) | latest | only for local container builds; AWS CodeBuild handles cloud builds |

You also need:

- An **AWS account** you own or have admin access to (the deploy creates IAM roles, S3 buckets, ECR repositories, CloudFront distributions, and so on).
- An **NVIDIA NGC API key** if (and only if) you intend to deploy the FCN3 model using the **NGC** container variant. The DLC variant works without one.

### Getting an NGC API key

Skip this section if you’ll only use the FCN3 **DLC** variant or none of the FCN3 endpoints at all.

1. Visit <https://ngc.nvidia.com/setup/api-key> and sign in with a free NVIDIA account.
2. Choose **Generate API Key** → **Confirm**.
3. Copy the key — it starts with `nvapi-…`. You won’t see it again, so save it now.

The `setup.sh` script (described in the next section) puts this key into AWS Secrets Manager at `/earth2/ngc-api-key`. The CDK stack and the FCN3 CodeBuild project read it from there. The key never gets committed to the repo.

---

## Walkthrough

This section is a full deploy-from-zero runbook. Each step is independent and verifiable — if something fails, the troubleshooting hint at the end of the step (or the [Troubleshooting](#troubleshooting) table at the end of the post) tells you what to check.

### Step 1 — Configure AWS credentials and region

The whole project picks up AWS account and region from the **standard AWS CLI configuration**. Nothing in this repo hardcodes either.

Edit (or create) `~/.aws/credentials`:

```ini
[default]
aws_access_key_id     = AKIA...
aws_secret_access_key = ...
# aws_session_token = ...   # only for temporary creds (SSO, STS, Isengard, etc.)
```

Edit (or create) `~/.aws/config`:

```ini
[default]
region = us-west-2
output = json
```

> **A note about config-file syntax:** the **default** profile uses the bare header `[default]` in *both* files. Other profiles use `[NAME]` in `credentials` but `[profile NAME]` in `config`. The region key is lowercase `region =`, not `AWS_REGION =`.

Verify both work:

```bash
aws sts get-caller-identity   # → prints your Account, UserId, Arn
aws configure get region      # → prints your region (e.g. us-west-2)
```

If either command fails, fix that first. AWS CDK will fail with `UnresolvedAccount` or `NoCredentials` until both succeed.

### Step 2 — Clone the repository and install dependencies

```bash
git clone https://github.com/aws-samples/sample-earth2studio-on-aws.git
cd sample-earth2studio-on-aws

# Python virtualenv for CDK
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-cdk.txt

# Node.js dependencies for the React frontend.
# The first `npm run build` is required because the CDK app references
# `frontend/dist/` as a deployment asset; every `cdk` command (bootstrap,
# synth, diff, deploy) fails at app-load time if that directory does not
# exist. Step 6 rebuilds it later with the real Cognito IDs.
cd frontend && npm install && npm run build && cd ..
```

### Step 3 — Run `setup.sh`

`setup.sh` does two things:

1. Validates AWS credentials and prints the account and region you’re about to deploy into. **Read this carefully** — if it shows the wrong account, fix your AWS profile before continuing.
2. Stores your NGC API key in Secrets Manager at `/earth2/ngc-api-key` (skip with `Ctrl-C` if you don’t have a key).

Run it **once per AWS account** before the first `cdk deploy`. There are three ways to invoke it:

```bash
# A. Interactive — script prompts for the NGC key (recommended for first-time setup)
./setup.sh

# B. Non-interactive — supply the key via env var (CI / repeat setups)
NGC_API_KEY=nvapi-xxxxxxxxxxxxxxxxxx ./setup.sh

# C. Credentials check only — does NOT write anything to Secrets Manager
./setup.sh --check-only
```

If the secret already exists, the script asks before overwriting. To target a different profile or region, set them first:

```bash
AWS_PROFILE=my-sandbox AWS_REGION=us-east-1 ./setup.sh
```

### Step 4 — Bootstrap AWS CDK

AWS CDK needs a small set of staging resources (an S3 bucket, ECR repository, IAM roles) in the target account+region before any stack can deploy. This is a one-time operation per **account + region** combination.

```bash
npx cdk bootstrap
```

Verify:

```bash
aws cloudformation describe-stacks --stack-name CDKToolkit \
  --query 'Stacks[0].StackStatus' --output text
# → CREATE_COMPLETE  or  UPDATE_COMPLETE
```

### Step 5 — Deploy the SageMaker infrastructure stack

This creates the long-lived resources that all SageMaker endpoints share: the model bucket, the IAM execution role, the FCN3 ECR repository, the CodeBuild project, and the SSM parameters that every other component reads.

```bash
npx cdk deploy Earth2SageMaker --require-approval never
```

You’ll see outputs like the following (the random suffixes will differ for you — write them down):

```
Earth2SageMaker.ModelBucketName  = earth2sagemaker-modelartifactsbucket80acad84-<random>
Earth2SageMaker.SageMakerRoleArn = arn:aws:iam::<account>:role/earth2-sagemaker-execution-role
Earth2SageMaker.SSMPrefix        = /earth2/sagemaker
```

Verify auto-discovery is wired up:

```bash
for p in bucket-name s3-prefix role-arn region; do
  echo "$p = $(aws ssm get-parameter --name /earth2/sagemaker/$p --query Parameter.Value --output text)"
done
```

> **Troubleshooting**: if you see `NoCredentials: Need to perform AWS calls for account ..., but no credentials have been configured` from `npx`, your credentials aren’t propagating. Run `eval "$(aws configure export-credentials --format env)"` and re-issue the deploy.

### Step 6 — Build and deploy the frontend

The React SPA needs the Cognito User Pool ID and Client ID **before** it can be built. Those values come from the *Earth2UI* stack outputs, which you haven’t deployed yet — chicken-and-egg. The workaround is a two-pass build.

**Pass 1 — deploy `Earth2UI` once with the placeholder bundle** so Cognito gets created:

```bash
cp frontend/.env.local.example frontend/.env.local
# Leave the placeholders for now — we'll fix them in Pass 2.
cd frontend && npm run build && cd ..
npx cdk deploy Earth2UI --require-approval never
```

Capture the outputs:

```
Earth2UI.UserPoolId       = us-west-2_XXXXXXXXX
Earth2UI.UserPoolClientId = XXXXXXXXXXXXXXXXXXXXXXXXXX
Earth2UI.CloudFrontURL    = https://XXXXXXXXXX.cloudfront.net
Earth2UI.ApiURL           = https://XXXXXXXXXX.execute-api.<region>.amazonaws.com/prod/
```

**Pass 2 — rebuild the frontend with the real Cognito IDs**:

```bash
# Edit frontend/.env.local with the values from above:
#   VITE_DEV_MODE=false
#   VITE_USER_POOL_ID=us-west-2_XXXXXXXXX
#   VITE_USER_POOL_CLIENT_ID=XXXXXXXXXXXXXXXXXXXXXXXXXX

cd frontend && npm run build && cd ..
npx cdk deploy Earth2UI --require-approval never
```

CDK detects the changed `frontend/dist/` asset, pushes it to S3, and invalidates CloudFront automatically. This second deploy is fast (~2 min).

> **Troubleshooting**: if the SPA loads but auth fails with “User pool client … does not exist”, the bundle was built with stale IDs. Recheck `frontend/.env.local` and rerun Pass 2.

### Step 7 — Create the first user and sign in

Open the CloudFront URL in a browser:

```
https://<your-cloudfront-id>.cloudfront.net
```

You should see a sign-in page. **Self-signup is disabled** at the Cognito User Pool level — administrators must provision users explicitly. Create the first user from the AWS CLI:

```bash
USER_POOL_ID=$(aws cloudformation describe-stacks --stack-name Earth2UI \
  --query 'Stacks[0].Outputs[?OutputKey==`UserPoolId`].OutputValue' --output text)

# 1. Create the user. Pass the email as --username; Cognito assigns an
#    internal UUID and uses the email as the sign-in alias (because the
#    pool was configured with sign_in_aliases=email).
aws cognito-idp admin-create-user \
  --user-pool-id "$USER_POOL_ID" \
  --username you@example.com \
  --user-attributes Name=email,Value=you@example.com Name=email_verified,Value=true \
  --message-action SUPPRESS \
  --temporary-password 'TempPass!1234'

# 2. Promote the password to permanent so the user doesn't get a
#    NEW_PASSWORD_REQUIRED challenge on first sign-in.
aws cognito-idp admin-set-user-password \
  --user-pool-id "$USER_POOL_ID" \
  --username you@example.com \
  --password 'YourRealStrongPassword!1' \
  --permanent
```

The password must satisfy: **min 8 chars, upper + lower + digit + symbol**. A common gotcha is a password like `password@1234567890` — it lacks an uppercase letter and Cognito will reject it with `InvalidPasswordException`.

Now sign in to the SPA with those credentials. You’ll land on the main dashboard. The SageMaker endpoints list will be empty until Step 9.

> **Pool quirk to know:** this stack sets `sign_in_aliases=email`, which Cognito implements as `UsernameAttributes=["email"]`. That means the email *is* the username — `admin-get-user` will show a UUID in the `Username` field, but you reference the user by email everywhere. To list users:
> ```bash
> aws cognito-idp list-users --user-pool-id "$USER_POOL_ID" \
>   --query 'Users[].[Username,UserStatus,Attributes[?Name==`email`]|[0].Value]' --output table
> ```
> To remove a user: `aws cognito-idp admin-delete-user --user-pool-id "$USER_POOL_ID" --username you@example.com`.

### Step 8 — (Optional) Build the FCN3 BYOC container

> **Heads-up before skipping:** the default Step 9 command — `./deploy-all.sh` — deploys **both** DLWP **and** FCN3, and the FCN3 deploy will fail if the BYOC image isn't built first. If you want to skip the FCN3 BYOC build, either deploy DLWP only (`python deploy.py --model dlwp`) or use the alternative `dlc` variant (`python deploy.py --model fcn3 --container-variant dlc`), which is built from a public AWS DLC and does **not** require an NGC API key. The standalone `deploy-all.sh --variant dlc` flag also works.

Skip this step if you only plan to use DLWP, which runs on the standard AWS-managed PyTorch DLC. FCN3 needs a custom container because:

1. PyTorch 2.6 (latest SageMaker inference DLC) doesn’t support Blackwell GPUs (`sm_120`).
2. FCN3 needs `torch-harmonics` + `makani`, which aren’t in standard DLCs.

Two BYOC variants exist (both run on `ml.g7e.2xlarge` Blackwell):

| | NGC (`container_fcn3/`) | Training DLC (`container_fcn3_dlc/`) |
|---|---|---|
| Base image | `nvcr.io/nvidia/pytorch:25.12-py3` | `pytorch-training:2.7.1-gpu-py312-cu128` |
| `torch-harmonics` | CUDA-compiled (faster) | PyPI wheel (float32 fallback) |
| Needs NGC key | yes (Secrets Manager) | no |
| Image size | ~15-20 GB | ~10-12 GB |

Build the **NGC** variant (the default):

```bash
BUCKET=$(aws ssm get-parameter --name /earth2/sagemaker/bucket-name --query Parameter.Value --output text)

# Zip the repo source (CodeBuild reads from S3, not GitHub)
zip -rq /tmp/fcn3-source.zip . \
  -x '.git/*' 'node_modules/*' 'frontend/node_modules/*' \
     '.venv/*' '__pycache__/*' '*/__pycache__/*' \
     'frontend/dist/*' 'cdk.out/*' \
     '.env' 'frontend/.env.local'

aws s3 cp /tmp/fcn3-source.zip s3://$BUCKET/codebuild/fcn3-source.zip
aws codebuild start-build --project-name earth2-fcn3-container-build
```

CodeBuild takes **~25–30 minutes** (the CUDA compile of `torch-harmonics` is the slow part). Monitor:

```bash
BUILD_ID=$(aws codebuild list-builds-for-project --project-name earth2-fcn3-container-build \
  --query 'ids[0]' --output text)
aws codebuild batch-get-builds --ids $BUILD_ID \
  --query 'builds[0].{Phase:currentPhase,Status:buildStatus}' --output table

aws logs tail /aws/codebuild/earth2-fcn3-container-build --follow
```

When complete, the image is at `<account>.dkr.ecr.<region>.amazonaws.com/earth2-fcn3:latest`.

### Step 9 — Deploy SageMaker endpoints

⚠️ **GPU instances are expensive — about $4.32/hr if both endpoints are running.** Always run `--delete` when you’re done.

```bash
cd sagemaker_deploy

# Deploy both default models (DLWP + FCN3-NGC)
./deploy-all.sh

# Or deploy a single model
python deploy.py --model dlwp --no-wait
python deploy.py --model fcn3 --container-variant ngc --no-wait
python deploy.py --model fcn3 --container-variant dlc --no-wait

# Check status
./deploy-all.sh --status

# Stop billing — delete all earth2-* endpoints
./deploy-all.sh --delete
```

A single endpoint takes 5–15 min to reach `InService`. The frontend’s `/api/endpoints` route auto-discovers them and shows them in the UI.

### Step 10 — Run a forecast

In the browser SPA:

1. Choose a model from the sidebar (DLWP or FCN3).
2. Pick an initial date (defaults to the most recent GFS analysis cycle).
3. Choose a lead time (24–240 hours).
4. Choose variables to plot (for example, `t2m`, `z500`, `msl`).
5. Choose **Run forecast**. The first call fetches GFS initial conditions (~30 sec) then runs the model (10–60 sec depending on model + lead time).

Or via the CLI:

```bash
python sagemaker_deploy/invoke_endpoint.py \
  --endpoint-name earth2-fcn3-endpoint \
  --date 2026-05-19T00:00:00 \
  --lead-time-hours 24 \
  --variables t2m z500 msl u10m v10m \
  --verbose
```

A sample successful FCN3 output:

```
======================================================================
  🌤️  WEATHER FORECAST RESULTS
======================================================================
  Model:           FCN3
  Init Time:       2026-05-19T00:00:00
  Lead Time:       24 hours (1.0 days)
  Steps:           4 × 6.0h
  Valid From:      2026-05-19T00:00:00
  Valid To:        2026-05-20T00:00:00

  Variable     Unit                Min         Mean          Max          Std
  ------------ ---------- ------------ ------------ ------------ ------------
  msl          hPa               949.1       1011.0       1069.0         11.8
  t2m          °C               -72.04         6.80        47.79        21.57
  z500         m²/s²         4.683e+04    5.437e+04    5.835e+04         3260

  Total variables: 3
  Data shape:      [1, 5, 721, 1440]
======================================================================
```

These are the **global statistics** at `2026-05-20T00:00:00 UTC` — that is, across all 721 × 1440 grid cells: a global mean temperature of 6.80 °C and a mean sea-level pressure (`msl`) range from 949 hPa (a deep extratropical low somewhere) to 1069 hPa (an intense Siberian high) — which is exactly what you’d expect for late May. Use the SPA or load `full_res_s3` to see *where* on the globe these extremes are.

---

## Configuration reference

This project does **not** require editing source files for account / region / bucket. The resolution order at deploy time is:

| Resource | Source |
|---|---|
| AWS account / region | `~/.aws/credentials` + `~/.aws/config` (via `CDK_DEFAULT_*` env vars). Override with `cdk -c account=… -c region=…`. |
| S3 model bucket | SSM `/earth2/sagemaker/bucket-name` (created by `Earth2SageMaker`). Override with `EARTH2_S3_BUCKET`. |
| SageMaker IAM role | SSM `/earth2/sagemaker/role-arn`. Override with `EARTH2_SAGEMAKER_ROLE`. |
| ECR image URI | Built dynamically from caller account + region + repo name. |
| Cognito User Pool / Client IDs | `frontend/.env.local` (copy from `.env.local.example`, fill from `Earth2UI` outputs). |
| NGC API key (FCN3 NGC only) | Secrets Manager `/earth2/ngc-api-key` (provisioned by `setup.sh`). |

---

## Project structure

```
sample-earth2studio-on-aws/
├── app.py                       # CDK app entry point
├── cdk.json                     # CDK configuration
├── setup.sh                     # One-time AWS creds check + NGC secret provision
├── requirements-cdk.txt         # CDK Python dependencies
│
├── docs/                        # Architecture diagrams (PNG/SVG/draw.io source) + UI screenshot
│
├── stacks/                      # CDK stack definitions
│   ├── sagemaker_infra_stack.py # S3 bucket, IAM role, ECR, CodeBuild, SSM
│   └── ui_stack.py              # CloudFront, API GW, Lambda, Cognito, WAFv2
│
├── backend/                     # Lambda function code
│   ├── handler.py               # API routes (endpoints, forecast, status)
│   ├── config.py                # Runtime config (reads env vars from CDK)
│   ├── local_server.py          # Flask dev server wrapping the Lambda handler
│   └── requirements.txt         # Lambda dependencies
│
├── frontend/                    # React SPA (Vite + TypeScript + Tailwind)
│   ├── .env.local.example       # Cognito config template (NEVER commit .env.local)
│   ├── src/api/                 # API client (forecast, endpoints)
│   ├── src/auth/                # Cognito authentication (SRP)
│   ├── src/components/          # Map, sidebar, charts
│   └── src/config/              # Variable metadata
│
├── sagemaker_deploy/            # SageMaker endpoint deployment
│   ├── deploy-all.sh            # Deploy/delete/status for all models
│   ├── deploy.py                # Single model deploy
│   ├── config.py                # Model definitions + SSM auto-discovery
│   ├── invoke_endpoint.py       # Test endpoint invocation
│   ├── model_code_dlwp/         # Inference handler — DLWP (pinned dependencies, see file)
│   ├── model_code_fcn3/         # Inference handler — FCN3 (needs torch-harmonics + makani)
│   ├── container_fcn3/          # BYOC NGC base (production, compiled CUDA)
│   └── container_fcn3_dlc/      # BYOC Training DLC base (no NGC key needed)
```

### Earth2SageMaker stack (infrastructure)

Long-lived supporting infrastructure for SageMaker endpoints:

- **S3 Bucket** — Model artifacts + async inference I/O, with lifecycle cleanup (1-day expiry on `async-input/` and `async-output/`).
- **IAM Role** — Least-privilege execution role (S3 paths scoped, ECR pull, CloudWatch logs).
- **ECR Repository** — `earth2-fcn3` for the BYOC container.
- **CodeBuild Project** — Builds the FCN3 image from the source zip in S3.
- **SSM Parameters** — Published under `/earth2/sagemaker/*` for auto-discovery.

### Earth2UI stack (frontend + backend)

Full-stack web application:

- **CloudFront** — CDN with two origins (S3 SPA + API Gateway for `/api/*`).
- **S3** — Static hosting for the React build.
- **API Gateway** — REST API with Cognito authorizer + AWS WAFv2 + request validation.
- **Lambda (Python 3.13)** — Backend that lists endpoints, runs forecasts, polls status.
- **Cognito** — Email + password SRP authentication, **self-signup disabled** (admins create users), adaptive auth on, no MFA.
- **AWS WAFv2** — AWS managed rules + 1000 req / 5 min per-IP rate limit.

---

## Security

All infrastructure passes the **CDK Nag `AwsSolutions`** rule pack:

- Cognito authentication on all API endpoints, **self-signup disabled** (admin-create-user only — addresses Palisade `SelfRegistrationEnabledAWS`).
- AWS WAFv2 with AWS managed rules + per-IP rate limiting.
- Least-privilege IAM (scoped to `earth2-*` endpoints + specific S3 prefixes).
- S3 SSE-S3 + enforced SSL + blocked public access + access logging.
- CloudFront TLS 1.2 minimum.
- API Gateway access logging + request validation.

### Code-level hardening

The repo addresses every high-severity finding reported by static-analysis tooling (Bandit, Semgrep, Checkov, ASH). Each finding below is either fixed in code or documented inline next to the relevant line so a future scanner run is auditable:

| Finding | File(s) | Resolution |
|---|---|---|
| **B201 / debug-enabled** — Flask `debug=True` in dev server | `backend/local_server.py` | Debug is **off by default**. Override via `FLASK_DEBUG=1` only when you need the Werkzeug debugger. |
| **B104 / avoid_app_run_with_bad_host** — Flask bound to `0.0.0.0` | `backend/local_server.py` | Defaults to `127.0.0.1`. Override via `FLASK_HOST=0.0.0.0` if you genuinely need to expose the dev server on a trusted LAN. |
| **B104** — Flask bound to `0.0.0.0` (BYOC dev fallback) | `sagemaker_deploy/container_fcn3*/sagemaker_serve.py` | The `__main__` blocks (developer-only) default to `127.0.0.1`. The production path — `serve` script + gunicorn on `0.0.0.0:8080` — is **unchanged** because the SageMaker BYOC contract requires it. |
| **B108** — `tempfile.mkdtemp(prefix=…)` flagged as insecure tempdir | `sagemaker_deploy/deploy.py` | Documented inline with `# nosec B108`: `tempfile.mkdtemp` is the secure modern API (atomic creation, mode `0o700`, OS-default temp location). The Bandit B108 rule is a regex pattern that cannot tell `mkdtemp()` already mitigates the risk it's looking for. |
| **`non-literal-import`** (Semgrep, `python.lang.security.audit.non-literal-import`) on `importlib.import_module(import_path)` | `sagemaker_deploy/model_code_dlwp/inference.py`, `sagemaker_deploy/model_code_fcn3/inference.py` | **Fixed.** Each handler now defines a hard-coded `ALLOWED_MODEL_IMPORTS = {"earth2studio.models.px": {"DLWP"}}` (or `{"FCN3"}`) and validates the `(module, class)` pair *before* calling `importlib.import_module`. Anything outside the allow-list raises `ValueError` and refuses the load — even if a future change to `model_config.json` introduced an unintended class. |
| **CKV_DOCKER_2** — missing Dockerfile `HEALTHCHECK` | `sagemaker_deploy/container_fcn3/Dockerfile`, `sagemaker_deploy/container_fcn3_dlc/Dockerfile` | Both Dockerfiles include `HEALTHCHECK CMD curl -fsS http://127.0.0.1:8080/ping`, with a `--start-period=1500s` to allow the first-call NVIDIA model-weight download to complete. |

### Known gaps you must address before going to production

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

---


## Cleanup

⚠️ **SageMaker endpoints are expensive GPU instances. Always delete them when you’re done.**

```bash
cd sagemaker_deploy
./deploy-all.sh --status    # See what's running
./deploy-all.sh --delete    # Stop billing immediately
```

| Resource | Cost | Lifecycle |
|---|---|---|
| CDK infrastructure (CloudFront, Lambda, S3, etc.) | ~$5–10 / month | Always on (minimal) |
| SageMaker endpoints (DLWP + FCN3) | **~$4.32 / hr** | On-demand only |

`./deploy-all.sh --delete` removes the SageMaker endpoints; the rest of the infra (S3 bucket, Cognito, CloudFront) keeps running cheaply. To take everything down, destroy the stacks in **reverse order** of creation, because `Earth2UI` imports the model bucket from `Earth2SageMaker`:

```bash
# 1. Stop any SageMaker endpoints (do this first to halt GPU charges)
cd sagemaker_deploy && ./deploy-all.sh --delete && cd ..

# 2. Destroy the UI stack (CloudFront takes 5–15 min to disable + delete)
npx cdk destroy Earth2UI --force

# 3. Destroy the SageMaker infra stack (purges S3 + ECR images via auto_delete)
npx cdk destroy Earth2SageMaker --force
```

The NGC secret in Secrets Manager is **not** managed by AWS CDK and survives teardown. To remove it:

```bash
aws secretsmanager delete-secret --secret-id /earth2/ngc-api-key \
  --force-delete-without-recovery
```

---

## Local development

```bash
# Frontend dev server (Vite, hot-reload). Proxies /api/* to backend port 3001.
cd frontend && npm run dev

# Backend local server — wraps the Lambda handler in Flask.
# Requires S3_BUCKET set; quickest way is to source it from SSM:
export AWS_REGION=us-west-2
export S3_BUCKET=$(aws ssm get-parameter --name /earth2/sagemaker/bucket-name \
  --query Parameter.Value --output text)
cd backend && python local_server.py

# CDK diff and synth (preview changes without deploying)
npx cdk diff
npx cdk synth
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `UnresolvedAccount` from `cdk deploy` | No credentials or region resolved | Run `aws sts get-caller-identity && aws configure get region` to verify both. Then `eval "$(aws configure export-credentials --format env)"` and retry. |
| `S3 bucket for model artifacts is not configured` | SSM params not populated | Deploy `Earth2SageMaker` first. |
| `Could not access model data` (SageMaker) | IAM role missing S3 permissions | Re-deploy `Earth2SageMaker`. The role grants are scoped by SSM-published prefix. |
| `Endpoint config already exists` | Stale resource from a failed deploy | `aws sagemaker delete-endpoint-config --endpoint-config-name <name>` |
| FCN3 inference returns CUDA kernel error | Container missing Blackwell support | Rebuild via Step 8 — make sure you’re using `container_fcn3/` (NGC) or `container_fcn3_dlc/`, NOT the AWS PyTorch DLC. |
| `User pool client ... does not exist` in browser | Stale `frontend/.env.local` | Update with the IDs from the latest `Earth2UI` outputs and rerun Step 6 Pass 2. |
| `User does not exist` on sign-in | Self-signup is disabled — that user was never created | Run the `admin-create-user` + `admin-set-user-password` commands from Step 7. |
| `Username should be an email` from `admin-create-user` | The pool uses `UsernameAttributes=email`, but the email wasn’t recognized | Pass the email as `--username` AND include `Name=email,Value=<email>` in `--user-attributes`. The example in Step 7 is correct. |
| `NEW_PASSWORD_REQUIRED` on sign-in | The password is still in temporary state from `admin-create-user` | Run `admin-set-user-password ... --permanent` to skip the first-time reset. |
| DLWP endpoint reports `InService` but every `/ping` returns 500 | `nvidia-physicsnemo` 2.x imports `warp.context`, which `warp-lang` ≥ 1.10 removed | Already fixed in `model_code_dlwp/requirements.txt` with version pins. If you forked an older copy, pin: `earth2studio<0.14.0`, `nvidia-physicsnemo<2.0.0`, `warp-lang<1.10.0`. |
| FCN3 forecast times out at 600 s | Model weights cold-cache download from NVIDIA registry | First-call latency is ~1–3 min. Re-issue the call; the second one will be in the 30-second range. |

---

## Conclusion

In this post we showed how to take two open-source AI weather models — DLWP and FourCastNet v3 — and deliver them as a secure, multi-tenant, production-style web application running entirely on AWS. The same `cdk deploy` and `./deploy-all.sh` workflow gets you from an empty AWS account to a global AI weather forecast in under an hour, for less than $10 per month of always-on infrastructure plus on-demand GPU time you can switch off when you’re done.

The same pattern — long-lived presentation infrastructure provisioned by AWS CDK in front of ephemeral GPU inference managed by a script — generalizes well beyond weather. If you have any other large open-source model that needs an A10G or Blackwell GPU and an authenticated UI, you can fork this repository, swap out the inference handler, and reuse essentially everything else.

We’d love to see what you build with it.

---

## License and model attribution

The code in this repository is licensed under [MIT-0](LICENSE) — the AWS Samples standard permissive license. See [`LICENSE`](LICENSE) for the full text.

The two weather models packaged here both carry permissive **Apache-2.0** licenses. Each model’s weights are downloaded at container start time from NVIDIA’s model registry and bundled into the running endpoint, so the Apache-2.0 license terms apply to your use of the resulting forecasts.

| Model | Origin | License | Source |
|---|---|---|---|
| **DLWP** | University of Washington (Karlbauer et al.), packaged via NVIDIA Earth2Studio | Apache-2.0 | <https://github.com/NVIDIA/earth2studio/blob/main/earth2studio/models/px/dlwp.py> |
| **FCN3** (FourCastNet v3) | NVIDIA, packaged via NVIDIA Earth2Studio | Apache-2.0 | <https://github.com/NVIDIA/earth2studio/blob/main/earth2studio/models/px/fcn3.py> |

**Why only these two?** Other models exposed by `earth2studio` (Pangu-Weather from Huawei, GraphCast from Google DeepMind, FuXi from Fudan University, etc.) carry research-only / non-commercial / CC-BY-NC licenses that are incompatible with publication under aws-samples without per-model legal review. They are intentionally not registered in `sagemaker_deploy/config.py`. If you need them for an internal or research deployment, you can add them back to your fork — but verify each model’s license against your use case first.

The NVIDIA Earth2Studio framework itself: <https://github.com/NVIDIA/earth2studio> (Apache-2.0).

---

## Further reading

If this is your first contact with AI-based weather forecasting, the following are worth bookmarking:

- **NVIDIA Earth2Studio**: [github.com/NVIDIA/earth2studio](https://github.com/NVIDIA/earth2studio) — the framework this project sits on top of, with model loaders, data sources (GFS, ERA5, ARCO, IFS), IO backends, and ensemble runners.
- **NVIDIA Modulus / PhysicsNeMo**: [github.com/NVIDIA/physicsnemo](https://github.com/NVIDIA/physicsnemo) — the underlying scientific-ML toolkit that supplies neural operators and data loaders.
- **NOAA GFS**: [www.nco.ncep.noaa.gov/pmb/products/gfs](https://www.nco.ncep.noaa.gov/pmb/products/gfs/) — the freely-available analysis data this project uses for initial conditions, updated every 6 hours.
- **ECMWF ERA5 reanalysis**: [www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5) — the 40-year archive most ML weather models are trained on.
- **WeatherBench 2**: [sites.research.google/weatherbench](https://sites.research.google/weatherbench/) — the standard benchmark for ML weather models against IFS HRES.
- **ECMWF AIFS**: [www.ecmwf.int/en/about/media-centre/aifs-blog](https://www.ecmwf.int/en/about/media-centre/aifs-blog) — ECMWF’s own ML-based forecast model, now operational.

The original AI-weather papers, in rough chronological order:

- **FourCastNet** (NVIDIA, 2022): [arxiv.org/abs/2202.11214](https://arxiv.org/abs/2202.11214)
- **Pangu-Weather** (Huawei, 2022): [www.nature.com/articles/s41586-023-06185-3](https://www.nature.com/articles/s41586-023-06185-3)
- **GraphCast** (DeepMind, 2023): [www.science.org/doi/10.1126/science.adi2336](https://www.science.org/doi/10.1126/science.adi2336)
- **FuXi** (Fudan, 2023): [www.nature.com/articles/s41612-023-00512-1](https://www.nature.com/articles/s41612-023-00512-1)
