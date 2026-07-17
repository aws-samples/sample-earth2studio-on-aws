#!/bin/bash
# ============================================================
# Deploy All Earth2Studio Weather Models to SageMaker
# ============================================================
#
# This script deploys multiple weather models to separate
# SageMaker endpoints. The React frontend (deployed via CDK)
# discovers these endpoints automatically.
#
# Usage:
#   ./deploy-all.sh              # Deploy all models
#   ./deploy-all.sh --delete     # Delete all endpoints
#   ./deploy-all.sh --status     # Check endpoint status
#
# Prerequisites:
#   - AWS credentials configured (aws sts get-caller-identity)
#   - Python venv activated with boto3, sagemaker installed
#   - config.py configured with S3_BUCKET and SAGEMAKER_ROLE
#
# Cost Warning:
#   DLWP:    ml.g5.2xlarge  ~$1.52/hr  (24GB A10G)
#   FCN3:    ml.g7e.2xlarge ~$4.20/hr  (96GB RTX PRO Server 6000)
#   ─────────────────────────────────
#   Total:                  ~$5.72/hr (~$137/day)
#
#   DELETE ENDPOINTS WHEN DONE! Run: ./deploy-all.sh --delete
#
# Container Variant (FCN3 only):
#   ./deploy-all.sh --variant ngc    # NVIDIA NGC base (default, compiled CUDA)
#   ./deploy-all.sh --variant dlc    # AWS Training DLC base (no NGC auth needed)
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ============================================================
# Model Definitions
# ============================================================
# Format: "model_name:instance_type"
# Only Apache-2.0-licensed models are shipped in this sample.
MODELS=(
    "dlwp:ml.g5.2xlarge"
    "fcn3:ml.g7e.2xlarge"
)

# ============================================================
# Colors for output
# ============================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================
# Parse variant option
# ============================================================
CONTAINER_VARIANT=""
REMAINING_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --variant)
            shift_next=true
            ;;
        ngc|dlc)
            if [ "$shift_next" = true ]; then
                CONTAINER_VARIANT="$arg"
                shift_next=false
            else
                REMAINING_ARGS+=("$arg")
            fi
            ;;
        --variant=*)
            CONTAINER_VARIANT="${arg#*=}"
            ;;
        *)
            REMAINING_ARGS+=("$arg")
            shift_next=false
            ;;
    esac
done
set -- "${REMAINING_ARGS[@]}"

# ============================================================
# Functions
# ============================================================

check_credentials() {
    echo -e "${BLUE}🔑 Checking AWS credentials...${NC}"
    if ! aws sts get-caller-identity --region us-west-2 > /dev/null 2>&1; then
        echo -e "${RED}❌ AWS credentials not configured or expired.${NC}"
        echo "   Run: aws configure  OR  set AWS_ACCESS_KEY_ID / AWS_SESSION_TOKEN"
        exit 1
    fi
    ACCOUNT=$(aws sts get-caller-identity --query Account --output text 2>/dev/null)
    echo -e "${GREEN}✅ Authenticated (Account: ${ACCOUNT})${NC}"
    echo ""
}

deploy_all() {
    check_credentials

    echo -e "${BLUE}🚀 Deploying ${#MODELS[@]} weather models to SageMaker...${NC}"
    echo "============================================================"
    echo ""

    FAILED=0
    for entry in "${MODELS[@]}"; do
        IFS=':' read -r model instance <<< "$entry"
        endpoint="earth2-${model}-endpoint"

        echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${BLUE}📦 Deploying: ${model} → ${endpoint} (${instance})${NC}"
        echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

        VARIANT_FLAG=""
        if [ -n "$CONTAINER_VARIANT" ] && [ "$model" = "fcn3" ]; then
            VARIANT_FLAG="--container-variant $CONTAINER_VARIANT"
            echo -e "${BLUE}  Container variant: ${CONTAINER_VARIANT}${NC}"
        fi

        if python deploy.py --model "$model" --instance-type "$instance" $VARIANT_FLAG; then
            echo -e "${GREEN}✅ ${model} deployed successfully!${NC}"
        else
            echo -e "${RED}❌ ${model} deployment failed!${NC}"
            FAILED=$((FAILED + 1))
        fi
        echo ""
    done

    echo "============================================================"
    if [ $FAILED -eq 0 ]; then
        echo -e "${GREEN}🎉 All ${#MODELS[@]} models deployed successfully!${NC}"
    else
        echo -e "${RED}⚠️  ${FAILED}/${#MODELS[@]} deployments failed.${NC}"
    fi
    echo ""
    echo -e "${YELLOW}💰 IMPORTANT: These endpoints cost ~\$5.72/hr total.${NC}"
    echo -e "${YELLOW}   Delete when done: ./deploy-all.sh --delete${NC}"
}

delete_all() {
    check_credentials

    echo -e "${RED}🗑️  Deleting all Earth2Studio endpoints...${NC}"
    echo "============================================================"
    echo ""

    for entry in "${MODELS[@]}"; do
        IFS=':' read -r model instance <<< "$entry"
        endpoint="earth2-${model}-endpoint"

        echo -e "${YELLOW}Deleting: ${endpoint}${NC}"
        if python deploy.py --delete --endpoint-name "$endpoint" 2>&1; then
            echo -e "${GREEN}✅ ${endpoint} deleted${NC}"
        else
            echo -e "${YELLOW}⚠️  ${endpoint} not found or already deleted${NC}"
        fi
        echo ""
    done

    echo "============================================================"
    echo -e "${GREEN}✅ All endpoints deleted. Billing stopped.${NC}"
}

show_status() {
    check_credentials

    echo -e "${BLUE}📊 Earth2Studio Endpoint Status${NC}"
    echo "============================================================"

    aws sagemaker list-endpoints \
        --region us-west-2 \
        --query 'Endpoints[?starts_with(EndpointName, `earth2-`)].{Name:EndpointName,Status:EndpointStatus,Created:CreationTime}' \
        --output table 2>&1

    echo ""
    RUNNING=$(aws sagemaker list-endpoints \
        --region us-west-2 \
        --status-equals InService \
        --query 'Endpoints[?starts_with(EndpointName, `earth2-`)] | length(@)' \
        --output text 2>/dev/null)

    if [ "$RUNNING" -gt 0 ] 2>/dev/null; then
        echo -e "${GREEN}🟢 ${RUNNING} endpoint(s) running${NC}"
        echo -e "${YELLOW}💰 Remember to delete when done: ./deploy-all.sh --delete${NC}"
    else
        echo -e "${YELLOW}⚪ No endpoints running${NC}"
    fi
}

# ============================================================
# Main
# ============================================================

case "${1:-}" in
    --delete|-d)
        delete_all
        ;;
    --status|-s)
        show_status
        ;;
    --help|-h)
        echo "Usage: ./deploy-all.sh [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  (no args)           Deploy all weather models"
        echo "  --delete, -d        Delete all endpoints (stop billing)"
        echo "  --status, -s        Show endpoint status"
        echo "  --variant ngc|dlc   FCN3 container: 'ngc' (default) or 'dlc' (no NGC auth)"
        echo "  --help, -h          Show this help"
        echo ""
        echo "Models deployed:"
        for entry in "${MODELS[@]}"; do
            IFS=':' read -r model instance <<< "$entry"
            echo "  • ${model} → ml instance: ${instance}"
        done
        ;;
    "")
        deploy_all
        ;;
    *)
        echo "Unknown option: $1"
        echo "Run: ./deploy-all.sh --help"
        exit 1
        ;;
esac
