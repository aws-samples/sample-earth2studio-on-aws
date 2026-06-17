# Project structure and stack reference

[← Back to the main README](../README.md)

This document describes the repository layout and what each of the two AWS CDK stacks provisions. For the end-to-end request path and the architecture diagram, see the [Solution overview](../README.md#solution-overview) in the main README.

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

## Earth2SageMaker stack (infrastructure)

Long-lived supporting infrastructure for SageMaker endpoints:

- **S3 Bucket** — Model artifacts + async inference I/O, with lifecycle cleanup (1-day expiry on `async-input/` and `async-output/`).
- **IAM Role** — Least-privilege execution role (S3 paths scoped, ECR pull, CloudWatch logs).
- **ECR Repository** — `earth2-fcn3` for the BYOC container.
- **CodeBuild Project** — Builds the FCN3 image from the source zip in S3.
- **SSM Parameters** — Published under `/earth2/sagemaker/*` for auto-discovery.

## Earth2UI stack (frontend + backend)

Full-stack web application:

- **CloudFront** — CDN with two origins (S3 SPA + API Gateway for `/api/*`).
- **S3** — Static hosting for the React build.
- **API Gateway** — REST API with Cognito authorizer + AWS WAFv2 + request validation.
- **Lambda (Python 3.13)** — Backend that lists endpoints, runs forecasts, polls status.
- **Cognito** — Email + password SRP authentication, **self-signup disabled** (admins create users), adaptive auth on, no MFA.
- **AWS WAFv2** — AWS managed rules + 1000 req / 5 min per-IP rate limit.
