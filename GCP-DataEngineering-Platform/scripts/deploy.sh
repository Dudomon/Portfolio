#!/bin/bash

###############################################################################
# GCP Data Engineering Platform Deployment Script
#
# This script deploys the complete data engineering platform to GCP including:
# - Infrastructure (Terraform)
# - Pipelines (Dataflow templates, Dataproc jobs)
# - Orchestration (Airflow DAGs)
# - API (Cloud Run)
# - Monitoring (dashboards and alerts)
###############################################################################

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-}"
REGION="${GCP_REGION:-us-central1}"
ENVIRONMENT="${ENVIRONMENT:-production}"

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check if gcloud is installed
    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI not found. Please install: https://cloud.google.com/sdk/docs/install"
        exit 1
    fi

    # Check if terraform is installed
    if ! command -v terraform &> /dev/null; then
        log_error "Terraform not found. Please install: https://www.terraform.io/downloads"
        exit 1
    fi

    # Check if project ID is set
    if [ -z "$PROJECT_ID" ]; then
        log_error "GCP_PROJECT_ID environment variable not set"
        exit 1
    fi

    # Verify authentication
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
        log_error "Not authenticated with gcloud. Run: gcloud auth login"
        exit 1
    fi

    log_info "Prerequisites check passed"
}

enable_apis() {
    log_info "Enabling required GCP APIs..."

    APIS=(
        "bigquery.googleapis.com"
        "dataflow.googleapis.com"
        "dataproc.googleapis.com"
        "pubsub.googleapis.com"
        "storage.googleapis.com"
        "composer.googleapis.com"
        "monitoring.googleapis.com"
        "logging.googleapis.com"
        "run.googleapis.com"
        "cloudbuild.googleapis.com"
    )

    for api in "${APIS[@]}"; do
        log_info "Enabling $api..."
        gcloud services enable "$api" --project="$PROJECT_ID" || true
    done

    log_info "API enablement complete"
}

deploy_infrastructure() {
    log_info "Deploying infrastructure with Terraform..."

    cd infrastructure/terraform

    # Initialize Terraform
    terraform init

    # Create workspace if it doesn't exist
    terraform workspace select "$ENVIRONMENT" 2>/dev/null || terraform workspace new "$ENVIRONMENT"

    # Plan deployment
    log_info "Running Terraform plan..."
    terraform plan \
        -var="project_id=$PROJECT_ID" \
        -var="region=$REGION" \
        -var="environment=$ENVIRONMENT" \
        -out=tfplan

    # Apply if plan succeeds
    log_info "Applying Terraform configuration..."
    terraform apply tfplan

    # Save outputs
    terraform output -json > ../../terraform_outputs.json
    log_info "Terraform outputs saved to terraform_outputs.json"

    cd ../..

    log_info "Infrastructure deployment complete"
}

build_dataflow_templates() {
    log_info "Building Dataflow pipeline templates..."

    # Get bucket names from Terraform output
    ARTIFACTS_BUCKET=$(jq -r '.storage_buckets.value.pipeline_artifacts' terraform_outputs.json)

    # Build streaming pipeline template
    log_info "Building streaming pipeline template..."
    python pipelines/beam/streaming_pipeline.py \
        --runner=DataflowRunner \
        --project="$PROJECT_ID" \
        --region="$REGION" \
        --template_location="gs://$ARTIFACTS_BUCKET/templates/streaming-pipeline-template.json" \
        --setup_file=./setup.py \
        --save_main_session

    # Build batch pipeline template
    log_info "Building batch pipeline template..."
    python pipelines/beam/batch_pipeline.py \
        --runner=DataflowRunner \
        --project="$PROJECT_ID" \
        --region="$REGION" \
        --template_location="gs://$ARTIFACTS_BUCKET/templates/batch-pipeline-template.json" \
        --setup_file=./setup.py \
        --save_main_session

    log_info "Dataflow templates built successfully"
}

upload_spark_jobs() {
    log_info "Uploading Spark jobs to GCS..."

    ARTIFACTS_BUCKET=$(jq -r '.storage_buckets.value.pipeline_artifacts' terraform_outputs.json)

    # Upload PySpark jobs
    gsutil cp pipelines/spark/*.py "gs://$ARTIFACTS_BUCKET/pipelines/spark/"

    log_info "Spark jobs uploaded successfully"
}

deploy_airflow_dags() {
    log_info "Deploying Airflow DAGs to Cloud Composer..."

    # Get Composer environment details
    COMPOSER_ENV_NAME="data-platform-composer"
    COMPOSER_LOCATION="$REGION"

    # Get DAGs bucket
    DAGS_BUCKET=$(gcloud composer environments describe "$COMPOSER_ENV_NAME" \
        --location="$COMPOSER_LOCATION" \
        --format="get(config.dagGcsPrefix)")

    # Upload DAGs
    gsutil -m cp infrastructure/composer/dags/*.py "$DAGS_BUCKET/"

    log_info "Airflow DAGs deployed successfully"
}

deploy_api() {
    log_info "Deploying API to Cloud Run..."

    cd api

    # Build container image
    log_info "Building container image..."
    gcloud builds submit \
        --tag="gcr.io/$PROJECT_ID/data-platform-api:latest" \
        --project="$PROJECT_ID"

    # Deploy to Cloud Run
    log_info "Deploying to Cloud Run..."
    gcloud run deploy data-platform-api \
        --image="gcr.io/$PROJECT_ID/data-platform-api:latest" \
        --platform=managed \
        --region="$REGION" \
        --allow-unauthenticated \
        --set-env-vars="PROJECT_ID=$PROJECT_ID" \
        --memory=2Gi \
        --cpu=2 \
        --project="$PROJECT_ID"

    cd ..

    log_info "API deployment complete"
}

setup_monitoring() {
    log_info "Setting up monitoring and alerting..."

    # Dashboards are created by Terraform
    # This section handles any additional monitoring setup

    log_info "Monitoring setup complete"
}

run_smoke_tests() {
    log_info "Running smoke tests..."

    # Test BigQuery connectivity
    log_info "Testing BigQuery..."
    bq query --use_legacy_sql=false \
        "SELECT COUNT(*) as health_check FROM \`$PROJECT_ID.raw_data.transactions\` LIMIT 1" \
        || log_warn "BigQuery test failed - table may not have data yet"

    # Test API health endpoint
    log_info "Testing API..."
    API_URL=$(gcloud run services describe data-platform-api \
        --region="$REGION" \
        --format="value(status.url)" \
        --project="$PROJECT_ID")

    curl -f "$API_URL/health" || log_warn "API health check failed"

    log_info "Smoke tests complete"
}

print_deployment_info() {
    log_info "=========================================="
    log_info "Deployment Summary"
    log_info "=========================================="
    log_info "Project ID: $PROJECT_ID"
    log_info "Region: $REGION"
    log_info "Environment: $ENVIRONMENT"
    log_info ""
    log_info "Resources Created:"
    log_info "- BigQuery datasets: raw_data, staging_data, analytics_data, data_quality"
    log_info "- Pub/Sub topics: user-events, transactions, pipeline-errors"
    log_info "- Dataproc cluster: data-processing-cluster"
    log_info "- Cloud Composer environment: data-platform-composer"
    log_info "- Cloud Run API: data-platform-api"
    log_info ""
    log_info "Next Steps:"
    log_info "1. Start streaming pipeline: See docs/deployment.md"
    log_info "2. Trigger batch processing DAG in Airflow UI"
    log_info "3. View monitoring dashboards in Cloud Console"
    log_info "4. Access API at: $(gcloud run services describe data-platform-api --region=$REGION --format='value(status.url)')"
    log_info "=========================================="
}

main() {
    log_info "Starting deployment of GCP Data Engineering Platform"
    log_info "Project: $PROJECT_ID"
    log_info "Region: $REGION"
    log_info "Environment: $ENVIRONMENT"

    check_prerequisites
    enable_apis
    deploy_infrastructure
    build_dataflow_templates
    upload_spark_jobs
    deploy_airflow_dags
    deploy_api
    setup_monitoring
    run_smoke_tests
    print_deployment_info

    log_info "Deployment completed successfully!"
}

# Run main function
main "$@"
