#!/usr/bin/env bash
# ============================================================
# Earth2Studio Weather Forecast Platform — one-time setup
# ============================================================
#
# What this script does:
#   1. Validates your AWS credentials and prints the account/region you'll
#      deploy into (so you don't accidentally deploy to the wrong account).
#   2. Stores your NVIDIA NGC API key in AWS Secrets Manager at
#      /earth2/ngc-api-key. The Earth2SageMaker CDK stack grants the FCN3
#      CodeBuild project read access to this secret so it can `docker login`
#      to nvcr.io and pull the base image.
#
# Everything else (S3 bucket, IAM role, ECR repo, CodeBuild project, SSM
# parameters) is created by `npx cdk deploy Earth2SageMaker`.
#
# ── How to get an NGC API key ────────────────────────────────
#   1. Visit https://ngc.nvidia.com/setup/api-key (free NVIDIA account).
#   2. Click "Generate API Key" → "Confirm".
#   3. Copy the key (starts with "nvapi-..."). You won't see it again.
# Skip this script entirely if you only plan to use the FCN3 *DLC* variant
# (which uses an AWS-managed base image and needs no NGC key).
#
# ── How to run ────────────────────────────────────────────────
#   # A. Interactive — script prompts for the NGC key (recommended)
#   ./setup.sh
#
#   # B. Non-interactive — supply the key via env var (CI / re-runs)
#   NGC_API_KEY=nvapi-xxxxxxxxxxxxxxxxxx ./setup.sh
#
#   # C. Credentials check only — does NOT write anything to Secrets Manager
#   ./setup.sh --check-only
#
# To target a specific AWS account/region, set the AWS env vars first:
#   export AWS_PROFILE=my-sandbox AWS_REGION=us-east-1
#   ./setup.sh

set -euo pipefail

CHECK_ONLY=0
[[ "${1:-}" == "--check-only" ]] && CHECK_ONLY=1
[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && { sed -n '2,40p' "$0"; exit 0; }

NGC_SECRET_NAME="/earth2/ngc-api-key"

# ── 1. Verify AWS credentials ──────────────────────────────────────
if ! command -v aws >/dev/null 2>&1; then
  echo "❌ AWS CLI not found. Install: https://aws.amazon.com/cli/" >&2
  exit 1
fi

if ! IDENTITY=$(aws sts get-caller-identity --output json 2>/dev/null); then
  echo "❌ AWS credentials are not configured or have expired."
  echo "   Run: aws configure   (or export AWS_PROFILE / SSO login)"
  exit 1
fi

ACCOUNT=$(echo "$IDENTITY" | sed -n 's/.*"Account": *"\([^"]*\)".*/\1/p')
ARN=$(echo "$IDENTITY" | sed -n 's/.*"Arn": *"\([^"]*\)".*/\1/p')
REGION=$(aws configure get region 2>/dev/null || echo "${AWS_REGION:-${AWS_DEFAULT_REGION:-}}")

if [[ -z "$REGION" ]]; then
  echo "❌ No AWS region configured for the active profile."
  echo "   Set one (replace <your-region> with e.g. us-west-2, us-east-1, eu-west-1):"
  echo "     aws configure set region <your-region>"
  echo "   or: export AWS_REGION=<your-region>"
  exit 1
fi

echo "✅ AWS credentials OK"
echo "   Account: $ACCOUNT"
echo "   Region:  $REGION"
echo "   Caller:  $ARN"
echo

if [[ "$CHECK_ONLY" == "1" ]]; then
  echo "(--check-only set; skipping NGC secret setup)"
  exit 0
fi

# ── 2. NGC API key in Secrets Manager ──────────────────────────────
echo "── NGC API key ─────────────────────────────────────────────"
echo "The FCN3 BYOC container pulls from nvcr.io and needs an NGC API key."
echo "Get one at: https://ngc.nvidia.com/setup/api-key"
echo "(If you only plan to use the DLC variant of FCN3, you can skip this"
echo " by pressing Ctrl-C now.)"
echo

if aws secretsmanager describe-secret --secret-id "$NGC_SECRET_NAME" --region "$REGION" >/dev/null 2>&1; then
  echo "ℹ️  Secret $NGC_SECRET_NAME already exists in $REGION."
  read -rp "Overwrite the existing key? [y/N] " yn
  if [[ "$yn" != "y" && "$yn" != "Y" ]]; then
    echo "Skipped."
    exit 0
  fi
  ACTION=put
else
  ACTION=create
fi

if [[ -z "${NGC_API_KEY:-}" ]]; then
  read -rsp "NGC API key: " NGC_API_KEY
  echo
fi

if [[ -z "$NGC_API_KEY" ]]; then
  echo "❌ No key provided. Aborting."
  exit 1
fi

if [[ "$ACTION" == "create" ]]; then
  aws secretsmanager create-secret \
    --name "$NGC_SECRET_NAME" \
    --description "NVIDIA NGC API key for Earth2Studio FCN3 container builds" \
    --secret-string "$NGC_API_KEY" \
    --region "$REGION" >/dev/null
else
  aws secretsmanager put-secret-value \
    --secret-id "$NGC_SECRET_NAME" \
    --secret-string "$NGC_API_KEY" \
    --region "$REGION" >/dev/null
fi

echo "✅ NGC API key stored at $NGC_SECRET_NAME ($REGION)"
echo
echo "Next steps:"
echo "  1. python3 -m venv .venv && source .venv/bin/activate"
echo "  2. pip install -r requirements-cdk.txt"
echo "  3. npx cdk bootstrap"
echo "  4. npx cdk deploy Earth2SageMaker"
echo "  5. cd frontend && npm install && npm run build && cd .."
echo "  6. npx cdk deploy Earth2UI"
