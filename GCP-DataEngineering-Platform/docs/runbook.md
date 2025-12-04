# Data Engineering Platform Runbook

## Overview

This runbook provides operational procedures for troubleshooting and resolving common issues in the GCP Data Engineering Platform. It is organized by component and includes step-by-step resolution procedures.

## Table of Contents

1. [Dataflow Pipeline Issues](#dataflow-pipeline-issues)
2. [Dataproc Job Failures](#dataproc-job-failures)
3. [BigQuery Errors](#bigquery-errors)
4. [Data Quality Failures](#data-quality-failures)
5. [Airflow DAG Issues](#airflow-dag-issues)
6. [API Service Issues](#api-service-issues)
7. [Monitoring and Alerting](#monitoring-and-alerting)

## Dataflow Pipeline Issues

### Symptom: Streaming Pipeline High Lag

**Indicators:**
- System lag metric > 5 minutes
- Alert: "Pipeline Processing Lag High"
- Backlog increasing in Pub/Sub subscription

**Diagnosis:**

```bash
# Check current pipeline status
gcloud dataflow jobs list \
  --region=us-central1 \
  --status=active \
  --format="table(id,name,state,type)"

# Get detailed metrics for specific job
gcloud dataflow jobs describe JOB_ID \
  --region=us-central1 \
  --format="yaml(currentState,currentStateTime,systemLag)"
```

**Common Causes:**
1. Insufficient worker count
2. BigQuery streaming insert throttling
3. Downstream bottleneck
4. Large message size causing processing delays

**Resolution Steps:**

1. **Increase worker count:**
```bash
# Update pipeline with more workers
gcloud dataflow jobs update-options JOB_ID \
  --region=us-central1 \
  --max-num-workers=20
```

2. **Check BigQuery quotas:**
```bash
# View BigQuery streaming insert quotas
gcloud compute project-info describe \
  --format="json" | jq '.quotas[] | select(.metric | contains("bigquery"))'
```

3. **Review pipeline metrics in Cloud Console:**
   - Navigate to Dataflow > Jobs > [Job Name]
   - Check "System Lag" and "Data Watermark Lag"
   - Review "Elements Added" rate
   - Check worker CPU and memory utilization

4. **If BigQuery throttling:**
   - Consider batching inserts instead of streaming
   - Implement backoff and retry logic
   - Request quota increase if sustained high throughput

### Symptom: Pipeline Failure on Startup

**Indicators:**
- Pipeline state: FAILED
- Error in logs: "Permission denied" or "Resource not found"

**Diagnosis:**

```bash
# View pipeline logs
gcloud logging read \
  "resource.type=dataflow_step AND resource.labels.job_id=JOB_ID" \
  --limit=50 \
  --format=json
```

**Common Causes:**
1. Service account lacks permissions
2. Input topic/subscription does not exist
3. Output BigQuery table schema mismatch
4. Networking issues (VPC, firewall)

**Resolution Steps:**

1. **Verify service account permissions:**
```bash
# Check IAM policy for service account
gcloud projects get-iam-policy PROJECT_ID \
  --flatten="bindings[].members" \
  --filter="bindings.members:serviceAccount:dataflow-sa@PROJECT_ID.iam.gserviceaccount.com"
```

Required roles:
- roles/dataflow.worker
- roles/bigquery.dataEditor
- roles/pubsub.subscriber
- roles/storage.objectAdmin

2. **Verify Pub/Sub resources:**
```bash
# Check subscription exists
gcloud pubsub subscriptions describe SUBSCRIPTION_NAME

# Check for message backlog
gcloud pubsub subscriptions describe SUBSCRIPTION_NAME \
  --format="value(numUndeliveredMessages)"
```

3. **Verify BigQuery table:**
```bash
# Check table exists and schema
bq show --format=prettyjson PROJECT_ID:DATASET.TABLE
```

4. **Check network configuration:**
   - Verify VPC and subnet configuration
   - Check firewall rules allow worker communication
   - Verify private IP settings if applicable

## Dataproc Job Failures

### Symptom: PySpark Job Fails with OOM Error

**Indicators:**
- Job status: ERROR
- Error message contains "OutOfMemoryError" or "Container killed"
- YARN logs show memory exceeded

**Diagnosis:**

```bash
# View job output
gcloud dataproc jobs describe JOB_ID \
  --region=us-central1

# Get YARN application logs
gcloud dataproc jobs wait JOB_ID \
  --region=us-central1 \
  && gcloud dataproc jobs get-iam-policy JOB_ID
```

**Resolution Steps:**

1. **Increase executor memory:**

Edit `pipelines/spark/batch_processor.py`:
```python
spark = SparkSession.builder \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()
```

2. **Increase cluster resources:**

Update Terraform `infrastructure/terraform/dataproc.tf`:
```hcl
worker_config {
  num_instances = 4
  machine_type  = "n1-standard-8"  # Increased from n1-standard-4
}
```

3. **Optimize Spark job:**
   - Increase partition count for large datasets
   - Use `repartition()` or `coalesce()` appropriately
   - Avoid `collect()` on large DataFrames
   - Use broadcast joins for small lookup tables

4. **Enable dynamic allocation:**
```python
spark = SparkSession.builder \
    .config("spark.dynamicAllocation.enabled", "true") \
    .config("spark.dynamicAllocation.minExecutors", "2") \
    .config("spark.dynamicAllocation.maxExecutors", "10") \
    .getOrCreate()
```

### Symptom: Dataproc Cluster Creation Fails

**Indicators:**
- Terraform apply fails on dataproc_cluster resource
- Error: "Quota exceeded" or "Zone unavailable"

**Resolution Steps:**

1. **Check quota availability:**
```bash
# View compute quotas
gcloud compute project-info describe \
  --format="table(quotas.metric,quotas.limit,quotas.usage)"
```

2. **Try different zone:**
```bash
# Check available zones
gcloud compute zones list --filter="region:us-central1"
```

Update `variables.tf`:
```hcl
variable "zone" {
  default = "us-central1-b"  # Changed from us-central1-a
}
```

3. **Use preemptible workers:**
   - Reduces cost and quota usage
   - Suitable for fault-tolerant workloads

## BigQuery Errors

### Symptom: Query Timeout or Slot Contention

**Indicators:**
- Query runs for extended period
- Error: "Resources exceeded during query execution"
- Monitoring shows high slot utilization

**Diagnosis:**

```bash
# Check recent query performance
bq ls -j -a -n 10 PROJECT_ID

# Get query details
bq show -j --format=prettyjson JOB_ID
```

**Resolution Steps:**

1. **Optimize query:**
   - Add partition filters to WHERE clause
   - Use clustering columns in filters
   - Avoid SELECT *
   - Use approximate aggregation functions (APPROX_COUNT_DISTINCT)

2. **Check partition pruning:**
```sql
-- Bad: Full table scan
SELECT * FROM `project.dataset.table`
WHERE user_id = 'user123'

-- Good: Partition pruning
SELECT * FROM `project.dataset.table`
WHERE DATE(ingestion_timestamp) = CURRENT_DATE()
  AND user_id = 'user123'
```

3. **Use materialized views for frequent queries:**
```sql
CREATE MATERIALIZED VIEW `project.analytics_data.user_daily_summary`
PARTITION BY summary_date
AS
SELECT
  DATE(transaction_timestamp) AS summary_date,
  user_id,
  COUNT(*) AS transaction_count,
  SUM(amount) AS total_amount
FROM `project.raw_data.transactions`
GROUP BY summary_date, user_id
```

4. **Consider BI Engine for dashboards:**
   - Enables fast query response for Looker Studio
   - Reduces slot usage for repeated queries

### Symptom: Schema Mismatch Error

**Indicators:**
- Insert fails with "Provided Schema does not match Table"
- Pipeline logs show schema validation error

**Resolution Steps:**

1. **Compare schemas:**
```bash
# View table schema
bq show --schema --format=prettyjson PROJECT_ID:DATASET.TABLE > current_schema.json

# View expected schema from pipeline code
```

2. **Update table schema (if backwards compatible):**
```bash
# Add new nullable column
bq update --schema new_schema.json PROJECT_ID:DATASET.TABLE
```

3. **Recreate table if breaking change:**
```bash
# Backup data
bq extract PROJECT_ID:DATASET.TABLE gs://BUCKET/backup/*.parquet

# Drop and recreate
bq rm -f -t PROJECT_ID:DATASET.TABLE
bq mk --table PROJECT_ID:DATASET.TABLE schema.json
```

## Data Quality Failures

### Symptom: Great Expectations Validation Fails

**Indicators:**
- Alert: "Data quality validation failed"
- Airflow task "validate_data_quality" fails
- Validation success rate < 100%

**Diagnosis:**

```sql
-- Check recent validation results
SELECT
  checkpoint_name,
  success,
  success_percent,
  unsuccessful_expectations,
  validation_timestamp
FROM `project.data_quality.validation_results`
WHERE DATE(validation_timestamp) = CURRENT_DATE()
ORDER BY validation_timestamp DESC
```

**Resolution Steps:**

1. **Review specific failures:**
   - Check Great Expectations data docs
   - Identify which expectation failed
   - Query actual data to understand issue

2. **Common validation failures:**

**Null values in required field:**
```sql
-- Investigate null values
SELECT COUNT(*)
FROM `project.raw_data.transactions`
WHERE user_id IS NULL
  AND DATE(ingestion_timestamp) = CURRENT_DATE()
```

**Value out of expected range:**
```sql
-- Check for outliers
SELECT
  MIN(amount) as min_amount,
  MAX(amount) as max_amount,
  APPROX_QUANTILES(amount, 100)[OFFSET(99)] as p99_amount
FROM `project.raw_data.transactions`
WHERE DATE(ingestion_timestamp) = CURRENT_DATE()
```

3. **Update expectations if data has legitimately changed:**
   - Edit `monitoring/data-quality/great_expectations_config.py`
   - Update expectation thresholds
   - Redeploy updated configuration

4. **Fix data quality at source:**
   - Add validation in ingestion pipeline
   - Implement data quality checks in upstream systems
   - Update dead letter queue handling

## Airflow DAG Issues

### Symptom: DAG Not Appearing in UI

**Resolution Steps:**

1. **Check DAG syntax:**
```bash
# Parse DAG file locally
python -m py_compile infrastructure/composer/dags/daily_batch_pipeline.py
```

2. **View Cloud Composer logs:**
```bash
# Check scheduler logs
gcloud logging read \
  "resource.type=cloud_composer_environment AND log_name=scheduler" \
  --limit=50
```

3. **Verify DAG uploaded to GCS:**
```bash
# List DAGs in Composer bucket
gsutil ls gs://COMPOSER_BUCKET/dags/
```

### Symptom: Task Stuck in Running State

**Resolution Steps:**

1. **Check task logs:**
   - Navigate to Airflow UI > DAG > Task
   - Click "Log" to view real-time output

2. **Clear task state:**
```bash
# Using Airflow CLI in Composer
gcloud composer environments run ENVIRONMENT_NAME \
  --location=us-central1 \
  tasks clear -- DAGID -t TASK_ID -s START_DATE -e END_DATE
```

3. **Kill zombie tasks:**
```bash
# If task is truly stuck
gcloud composer environments run ENVIRONMENT_NAME \
  --location=us-central1 \
  tasks kill -- DAG_ID TASK_ID EXECUTION_DATE
```

## API Service Issues

### Symptom: API Returns 500 Internal Server Error

**Diagnosis:**

```bash
# Check API logs
gcloud logging read \
  "resource.type=cloud_run_revision AND textPayload:ERROR" \
  --limit=20 \
  --format=json
```

**Resolution Steps:**

1. **Verify BigQuery connectivity:**
   - Check service account permissions
   - Test query manually in BigQuery console

2. **Review application logs:**
```bash
# Stream logs in real-time
gcloud run services logs tail api-server --region=us-central1
```

3. **Check resource limits:**
   - Verify Cloud Run instance has sufficient memory
   - Check for timeout issues on long-running queries

4. **Restart service:**
```bash
# Deploy new revision
gcloud run services update api-server \
  --region=us-central1 \
  --update-env-vars="RESTART_TIMESTAMP=$(date +%s)"
```

## Monitoring and Alerting

### Key Metrics to Monitor

1. **Pipeline Health:**
   - Dataflow system lag < 5 minutes
   - Job success rate > 95%
   - Data freshness < 30 minutes

2. **BigQuery:**
   - Slot utilization < 80%
   - Query execution time p95 < 30 seconds
   - Failed query rate < 1%

3. **Data Quality:**
   - Validation success rate > 95%
   - Schema drift detection
   - Null rate in required fields < 1%

### Accessing Dashboards

```bash
# Pipeline Health Dashboard
https://console.cloud.google.com/monitoring/dashboards/custom/PIPELINE_HEALTH_ID

# Data Quality Dashboard
https://console.cloud.google.com/monitoring/dashboards/custom/DATA_QUALITY_ID
```

### Escalation Procedures

**P1 - Critical (Pipeline Down):**
1. Page on-call engineer immediately
2. Open incident channel in Slack
3. Follow incident response procedures

**P2 - High (Data Quality Issues):**
1. Notify data engineering team in Slack
2. Create JIRA ticket
3. Investigate within 2 hours

**P3 - Medium (Performance Degradation):**
1. Create JIRA ticket
2. Investigate during business hours

## Contact Information

- Data Engineering Team: data-engineering@example.com
- On-Call Rotation: PagerDuty
- Slack Channel: #data-platform

## Change Log

| Date | Author | Changes |
|------|--------|---------|
| 2024-01-15 | Data Team | Initial runbook creation |
