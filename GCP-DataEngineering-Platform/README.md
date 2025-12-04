# GCP Data Engineering Platform with Pipeline Observability

## Overview

This project demonstrates a production-grade data engineering platform built on Google Cloud Platform (GCP). It implements end-to-end data pipelines with comprehensive monitoring, data quality validation, and collaboration features.

The platform processes data from multiple sources (streaming and batch), transforms it through multiple layers, and makes it available for analytics and machine learning workloads.

## Architecture

### Data Flow

```
Sources -> Ingestion -> Processing -> Storage -> Consumption
```

**Ingestion Layer**
- Cloud Pub/Sub for real-time event streaming
- Cloud Storage for batch file ingestion (CSV, Parquet, JSON)
- Cloud SQL (PostgreSQL) for transactional data replication

**Processing Layer**
- Apache Beam (Dataflow) for unified batch and stream processing
- Apache Spark (Dataproc) for large-scale batch transformations
- dbt for SQL-based transformations in BigQuery

**Orchestration**
- Cloud Composer (Apache Airflow) for workflow management
- Scheduled queries for recurring analytics

**Storage Layer**
- BigQuery data warehouse with three-tier architecture:
  - Raw layer: Unprocessed data from sources
  - Staging layer: Cleaned and validated data
  - Analytics layer: Business-ready aggregations

**Monitoring & Quality**
- Cloud Monitoring for metrics and dashboards
- Cloud Logging for structured log analysis
- Great Expectations for data quality validation
- Custom SLA tracking and alerting

**Collaboration**
- FastAPI REST API for data access
- Looker Studio dashboards for business intelligence
- Jupyter notebooks for ad-hoc analysis

## Technical Stack

**Languages & Frameworks**
- Python 3.11 (Apache Beam, PySpark, FastAPI, Great Expectations)
- SQL (BigQuery SQL, dbt)
- Bash (deployment scripts)

**GCP Services**
- BigQuery (data warehouse)
- Cloud Dataflow (managed Beam)
- Cloud Dataproc (managed Spark/Hadoop)
- Cloud Pub/Sub (messaging)
- Cloud Storage (object storage)
- Cloud Composer (managed Airflow)
- Cloud Monitoring & Logging
- Cloud SQL (PostgreSQL)

**Big Data Tools**
- Apache Beam 2.52.0
- Apache Spark 3.5.0
- Apache Airflow 2.7.0
- dbt 1.7.0

**Infrastructure & DevOps**
- Terraform 1.6+ for infrastructure as code
- Docker for containerization
- GitHub Actions for CI/CD

## Project Structure

```
GCP-DataEngineering-Platform/
├── infrastructure/
│   ├── terraform/              # Infrastructure as code
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   ├── outputs.tf
│   │   ├── bigquery.tf
│   │   ├── dataflow.tf
│   │   ├── dataproc.tf
│   │   ├── pubsub.tf
│   │   ├── composer.tf
│   │   └── monitoring.tf
│   └── composer/               # Airflow DAGs
│       ├── dags/
│       └── plugins/
│
├── pipelines/
│   ├── beam/                   # Dataflow pipelines
│   │   ├── streaming_pipeline.py
│   │   ├── batch_pipeline.py
│   │   ├── pipeline_options.py
│   │   └── transforms/
│   ├── spark/                  # PySpark jobs
│   │   ├── batch_processor.py
│   │   ├── aggregations.py
│   │   └── utils/
│   └── dbt/                    # SQL transformations
│       ├── models/
│       ├── tests/
│       └── dbt_project.yml
│
├── monitoring/
│   ├── dashboards/             # Cloud Monitoring configs
│   │   ├── pipeline_health.json
│   │   └── data_freshness.json
│   ├── alerts/                 # Alerting policies
│   │   ├── sla_breach.yaml
│   │   └── pipeline_failure.yaml
│   └── data-quality/           # Great Expectations
│       ├── expectations/
│       └── checkpoints/
│
├── api/
│   ├── src/
│   │   ├── main.py             # FastAPI application
│   │   ├── routers/
│   │   ├── models/
│   │   └── services/
│   ├── tests/
│   └── requirements.txt
│
├── notebooks/
│   └── analysis/               # Jupyter notebooks
│
├── looker-studio/
│   └── dashboard-configs/      # BI dashboard exports
│
├── docs/
│   ├── architecture.md         # Detailed architecture
│   ├── deployment.md           # Deployment guide
│   ├── runbook.md              # Troubleshooting guide
│   ├── optimization.md         # Performance tuning
│   └── api.md                  # API documentation
│
├── scripts/
│   ├── deploy.sh               # Deployment automation
│   ├── test-local.sh           # Local testing
│   └── generate-sample-data.py # Test data generator
│
├── tests/
│   ├── unit/                   # Unit tests (Beam, Spark, API, Data Quality)
│   ├── integration/            # Integration and E2E tests
│   └── fixtures/               # Test data and fixtures
│
├── .github/workflows/          # CI/CD pipeline
├── .gitignore
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
└── Makefile
```

## Key Features

### Pipeline Monitoring

**Health Metrics**
- Pipeline execution time and throughput
- Error rates and retry counts
- Data freshness by table
- Resource utilization (CPU, memory)
- Cost per pipeline run

**Custom Dashboards**
- Real-time pipeline status
- SLA compliance tracking
- Historical trend analysis
- Alerting integration

### Troubleshooting Capabilities

**Structured Logging**
- Context-rich log entries with correlation IDs
- Error categorization (schema, network, data quality)
- Stack traces and debugging information

**Error Handling**
- Dead letter queues for failed messages
- Automatic retry with exponential backoff
- Circuit breaker pattern for external services
- Detailed error reporting and aggregation

**Root Cause Analysis**
- Log aggregation by error type
- Performance bottleneck identification
- Data lineage tracking

### Performance Optimization

**BigQuery Optimization**
- Table partitioning by ingestion date
- Clustering by frequently filtered columns
- Materialized views for common queries
- Query result caching
- Cost monitoring per query

**Dataflow Optimization**
- Autoscaling worker configuration
- Batch vs streaming trade-off analysis
- Windowing strategies for streaming
- Side input optimization

**Spark Optimization**
- Dynamic resource allocation
- Broadcast joins for small datasets
- Partition tuning
- Caching strategy for iterative operations

### Data Quality

**Validation Rules**
- Schema validation on ingestion
- Null checks on required fields
- Range validation for numeric columns
- Referential integrity checks
- Format validation (email, phone, etc.)

**Great Expectations Integration**
- Automated expectation suite generation
- Data profiling reports
- Validation result storage in BigQuery
- Alert integration for quality failures

### Collaboration Features

**For Data Scientists**
- REST API with endpoints for common queries
- Jupyter notebook templates
- Sample datasets for experimentation
- Documentation with example queries

**For Business Intelligence**
- Looker Studio dashboards with key metrics
- Scheduled reports via email
- Self-service query interface
- Data dictionary with business definitions

**For Engineering Teams**
- Git-based workflow for pipeline code
- Code review process
- Automated testing in CI/CD
- Slack integration for alerts

## Prerequisites

**Local Development**
- Python 3.11+
- Terraform 1.6+
- Docker 24.0+
- gcloud CLI

**GCP Requirements**
- GCP Project with billing enabled
- Service account with appropriate IAM roles:
  - BigQuery Admin
  - Dataflow Admin
  - Dataproc Admin
  - Pub/Sub Admin
  - Storage Admin
  - Composer Admin
  - Monitoring Admin

**APIs to Enable**
```bash
gcloud services enable \
  bigquery.googleapis.com \
  dataflow.googleapis.com \
  dataproc.googleapis.com \
  pubsub.googleapis.com \
  storage.googleapis.com \
  composer.googleapis.com \
  monitoring.googleapis.com \
  logging.googleapis.com
```

## Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone <repository-url>
cd GCP-DataEngineering-Platform

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure GCP Credentials

```bash
# Authenticate with GCP
gcloud auth login
gcloud auth application-default login

# Set project
export GCP_PROJECT_ID="your-project-id"
gcloud config set project $GCP_PROJECT_ID
```

### 3. Deploy Infrastructure

```bash
cd infrastructure/terraform

# Initialize Terraform
terraform init

# Review plan
terraform plan -var="project_id=$GCP_PROJECT_ID"

# Deploy (review carefully before applying)
terraform apply -var="project_id=$GCP_PROJECT_ID"
```

### 4. Deploy Pipelines

```bash
# Deploy Beam pipeline to Dataflow
python pipelines/beam/streaming_pipeline.py \
  --project=$GCP_PROJECT_ID \
  --region=us-central1 \
  --runner=DataflowRunner \
  --temp_location=gs://your-bucket/temp

# Submit Spark job to Dataproc
gcloud dataproc jobs submit pyspark \
  pipelines/spark/batch_processor.py \
  --cluster=data-processing-cluster \
  --region=us-central1
```

### 5. Run Data Quality Checks

```bash
cd monitoring/data-quality
great_expectations checkpoint run ecommerce_transactions
```

### 6. Start API Server

```bash
cd api
uvicorn src.main:app --reload --port 8000
```

Access API documentation at `http://localhost:8000/docs`

## Testing

### Running Tests

The project includes comprehensive unit and integration tests with 75%+ code coverage:

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run all tests with coverage
make test

# Run unit tests only (fast)
make test-unit

# Run integration tests
make test-integration

# View coverage report
pytest --cov --cov-report=html
# Open htmlcov/index.html
```

### Test Organization

- **Unit Tests** (tests/unit/): Fast, isolated tests for Beam pipelines, Spark jobs, API endpoints, and data quality
- **Integration Tests** (tests/integration/): End-to-end workflow validation
- **CI/CD**: Automated testing on every commit via GitHub Actions

See [Testing Guide](docs/testing.md) for comprehensive documentation.

### Testing Locally

```bash
# Run local pipeline tests with DirectRunner (Beam)
pytest tests/unit/test_beam_pipeline.py -v

# Generate sample data for testing
python scripts/generate-sample-data.py --num-transactions 1000

# Test API locally
cd api && uvicorn src.main:app --reload
# Access http://localhost:8000/docs
```

## Monitoring

Access monitoring dashboards:
- Cloud Console: `https://console.cloud.google.com/monitoring`
- Custom dashboards are created during Terraform deployment

Key metrics to monitor:
- Pipeline lag (streaming)
- Job success rate (batch)
- Data freshness
- Query cost
- API latency

## Cost Optimization

**Implemented Strategies**
- Partition pruning in BigQuery queries
- Autoscaling for Dataflow and Dataproc
- Preemptible VMs for non-critical workloads
- Table expiration for temporary data
- Query result caching
- Scheduled cluster shutdown

**Expected Costs** (us-central1, moderate usage)
- BigQuery storage: ~$0.02/GB/month
- BigQuery analysis: ~$5/TB processed
- Dataflow: ~$0.08/vCPU-hour + $0.008/GB-hour
- Dataproc: ~$0.06/vCPU-hour (with preemptible)
- Pub/Sub: First 10GB free, then $0.06/GB
- Cloud Storage: ~$0.02/GB/month

## Documentation

Detailed documentation available in `/docs`:
- [Architecture Overview](docs/architecture.md)
- [Deployment Guide](docs/deployment.md)
- [Troubleshooting Runbook](docs/runbook.md)
- [Performance Optimization](docs/optimization.md)
- [API Reference](docs/api.md)

## Security Considerations

**Implemented Measures**
- Service accounts with principle of least privilege
- VPC Service Controls for data exfiltration prevention
- Customer-managed encryption keys (CMEK) support
- Audit logging enabled for all services
- Secrets stored in Secret Manager
- Network isolation for Dataproc clusters

## Contributing

This is a portfolio project demonstrating data engineering capabilities. The code follows industry best practices:

- PEP 8 style guide for Python
- Type hints throughout codebase
- Comprehensive unit tests
- Integration tests for critical paths
- Documentation for all public APIs

## License

This project is part of a professional portfolio and is available for review and evaluation purposes.

## Contact

Eduardo Peiter
- GitHub: @Dudomon
- Location: Brazil

## Acknowledgments

This project demonstrates practical implementation of data engineering patterns commonly used in production environments at scale-ups and enterprises.
