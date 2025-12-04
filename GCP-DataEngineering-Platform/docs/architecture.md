## Architecture Documentation

## System Overview

The GCP Data Engineering Platform is a production-grade, cloud-native data processing system designed for scalable analytics workloads. The architecture follows industry best practices for data engineering, including the Lambda architecture pattern (batch and streaming), data quality-first design, and comprehensive observability.

### Design Principles

1. **Separation of Concerns**: Clear boundaries between ingestion, processing, storage, and consumption layers
2. **Idempotency**: All pipelines can be safely re-run without data duplication
3. **Schema Evolution**: Support for backward-compatible schema changes
4. **Data Quality First**: Validation at every stage with automated monitoring
5. **Cost Optimization**: Partitioning, clustering, and lifecycle policies to minimize costs
6. **Observability**: Comprehensive logging, metrics, and alerting throughout the system

## Architecture Layers

### 1. Data Ingestion Layer

**Purpose**: Reliably ingest data from multiple sources into GCP

**Components**:
- Cloud Pub/Sub topics for real-time event streaming
- Cloud Storage for batch file uploads
- Cloud SQL connectors for database replication

**Design Decisions**:
- Pub/Sub chosen for its at-least-once delivery guarantee and native GCP integration
- Dead letter topics configured for failed messages to prevent data loss
- Message retention set to 24 hours to allow replay during incidents

**Data Flow**:
```
External Sources -> Cloud Pub/Sub Topics -> Subscriptions -> Dataflow
External Systems -> Cloud Storage Buckets -> Batch Jobs
```

### 2. Processing Layer

**Purpose**: Transform raw data into business-ready analytics

#### 2.1 Stream Processing (Apache Beam on Dataflow)

**Capabilities**:
- Real-time event processing with sub-minute latency
- Stateful transformations using windowing
- Auto-scaling from 1 to N workers based on throughput
- Exactly-once semantics for BigQuery writes

**Pipeline Architecture**:
```
Read from Pub/Sub
    |
    v
Parse & Validate JSON
    |
    v
Enrich with Metadata
    |
    v
Data Quality Checks
    |
    v
Format for BigQuery
    |
    v
Write to BigQuery (Streaming Inserts)
```

**Error Handling**:
- All validation failures tagged and routed to error topic
- Retry logic with exponential backoff
- Dead letter queue for permanently failed messages

#### 2.2 Batch Processing (Apache Spark on Dataproc)

**Capabilities**:
- Large-scale batch transformations (TB-scale)
- Complex aggregations and joins
- Machine learning feature engineering
- Cost-effective with preemptible workers

**Cluster Configuration**:
- Master: 1x n1-standard-4
- Workers: 2x n1-standard-4
- Preemptible: 4x n1-standard-4 (60% cost reduction)
- Auto-scaling enabled for variable workloads
- Idle cluster deletion after 1 hour

**Optimization Techniques**:
- Adaptive query execution enabled
- Broadcast joins for small lookup tables
- Partition pruning for time-series data
- Columnar caching for iterative operations

#### 2.3 SQL Transformations (dbt)

**Purpose**: Modular, testable SQL transformations in BigQuery

**Layer Structure**:
```
Raw Layer (raw_data.*)
    |
    v
Staging Layer (staging_data.*)
    - Basic cleaning
    - Type casting
    - Deduplication
    |
    v
Analytics Layer (analytics_data.*)
    - Business logic
    - Aggregations
    - Materialized views
```

**Features**:
- Version-controlled SQL transformations
- Automatic documentation generation
- Built-in testing framework
- Incremental model support for efficiency

### 3. Orchestration Layer

**Purpose**: Coordinate data pipeline execution and dependencies

**Technology**: Cloud Composer (managed Apache Airflow)

**DAG Design**:
- Daily batch processing DAG: Runs at 2 AM daily
- Data quality validation DAG: Runs every 6 hours
- Error recovery DAG: Automatically retries failed tasks

**Key Features**:
- Task dependency management
- SLA monitoring with alerting
- Parameterized execution for backfills
- Integration with GCP services (Dataproc, Dataflow, BigQuery)

**Best Practices Implemented**:
- Idempotent tasks for safe retries
- Atomic operations (create cluster, run job, delete cluster)
- Comprehensive logging at each step
- Email alerts on failure

### 4. Storage Layer

**Purpose**: Efficient, queryable storage of processed data

#### 4.1 BigQuery Data Warehouse

**Dataset Architecture**:

**raw_data** (unprocessed data from sources)
- Partitioned by ingestion_timestamp (daily)
- Clustered by user_id and transaction_type
- 90-day expiration on raw data
- Used for replay and audit purposes

**staging_data** (cleaned and validated)
- Partitioned by transaction_date (daily)
- Clustered by user_id
- Intermediate layer for transformations
- No expiration (used for historical analysis)

**analytics_data** (business-ready metrics)
- Partitioned by date dimension
- Clustered by relevant dimensions
- Optimized for query performance
- Materialized views for common queries

**data_quality** (validation results and metrics)
- Stores Great Expectations validation results
- Pipeline execution metrics
- Data freshness tracking

**Optimization Strategy**:
```sql
-- Example optimized table structure
CREATE TABLE analytics_data.user_transaction_metrics
PARTITION BY last_transaction_date
CLUSTER BY user_id, user_segment
AS SELECT ...
```

**Cost Controls**:
- Partitioning to minimize scanned data
- Clustering to reduce shuffle operations
- Query result caching (24 hours)
- Scheduled query management
- Budget alerts at 80% and 100% thresholds

### 5. Monitoring & Data Quality Layer

**Purpose**: Ensure data reliability and system health

#### 5.1 Data Quality (Great Expectations)

**Validation Categories**:
1. Schema validation (column existence, types)
2. Completeness checks (null rates)
3. Range validation (numeric boundaries)
4. Referential integrity (foreign key checks)
5. Business rule validation (custom logic)

**Implementation**:
```python
# Example expectation suite
expectations = [
    expect_column_values_to_not_be_null('transaction_id'),
    expect_column_values_to_be_unique('transaction_id'),
    expect_column_values_to_be_between('amount', min_value=0, max_value=1000000),
    expect_column_values_to_be_in_set('currency', ['USD', 'EUR', 'GBP', 'BRL'])
]
```

**Validation Workflow**:
```
Data Arrives -> Run Expectation Suite -> Pass/Fail
                                           |
                                           v
                                   Store Results in BigQuery
                                           |
                                           v
                                   Alert if Failure Rate > Threshold
```

#### 5.2 System Monitoring (Cloud Monitoring)

**Key Metrics**:
- Pipeline health (success rate, latency, throughput)
- Data freshness (time since last insert)
- BigQuery performance (slot utilization, query duration)
- Cost metrics (spend per service)

**Alerting Policies**:
- Pipeline failure: Immediate PagerDuty alert
- High lag (>5 min): Warning to Slack
- Data freshness SLA breach: Critical alert
- Budget threshold exceeded: Email to finance team

**Dashboards**:
1. Pipeline Health Dashboard
   - Real-time job status
   - Throughput and latency trends
   - Error rates by pipeline

2. Data Quality Dashboard
   - Validation success rate over time
   - Failed expectations by category
   - Data completeness metrics

### 6. API & Collaboration Layer

**Purpose**: Enable self-service data access for stakeholders

#### 6.1 REST API (FastAPI on Cloud Run)

**Endpoints**:
```
GET /api/v1/users/{user_id}/metrics
GET /api/v1/users/top?limit=10&order_by=total_amount
GET /api/v1/merchants/{merchant_id}/metrics
GET /api/v1/data-quality/status
GET /api/v1/pipelines/metrics
GET /api/v1/analytics/daily-summary
```

**Features**:
- Authentication via API keys (for production)
- Rate limiting to prevent abuse
- Query parameter validation
- Automatic API documentation (Swagger/OpenAPI)
- Response caching for frequently accessed data

**Deployment**:
- Serverless on Cloud Run
- Auto-scales from 0 to 100 instances
- Integrated with Cloud Load Balancing
- Deployed via CI/CD pipeline

#### 6.2 Business Intelligence (Looker Studio)

**Dashboard Categories**:
1. Executive Dashboard
   - Daily active users
   - Transaction volume trends
   - Revenue by segment

2. Operations Dashboard
   - Pipeline health status
   - Data quality scores
   - System performance metrics

3. Data Quality Dashboard
   - Validation results
   - Data completeness
   - Schema drift detection

## Security Architecture

### Identity and Access Management

**Service Accounts**:
- dataflow-sa: Dataflow worker operations
- dataproc-sa: Dataproc cluster operations
- composer-sa: Airflow orchestration
- api-sa: API server queries

**Principle of Least Privilege**:
- Each service account has minimum required permissions
- No broad "Owner" or "Editor" roles
- Audited quarterly for permission creep

### Data Security

**Encryption**:
- At rest: Default Google-managed encryption
- In transit: TLS 1.2+ for all connections
- Option for customer-managed encryption keys (CMEK)

**Network Security**:
- VPC Service Controls to prevent data exfiltration
- Private IP for Dataproc clusters
- Cloud NAT for controlled internet access

**Audit Logging**:
- All BigQuery queries logged
- IAM policy changes tracked
- Data access logs enabled

## Scalability Considerations

### Horizontal Scaling

**Dataflow**:
- Auto-scales workers based on Pub/Sub backlog
- Tested up to 100 workers (10k events/second)
- Linear scaling observed up to tested limits

**Dataproc**:
- Auto-scaling policy configured
- Scales up to 10 standard + 20 preemptible workers
- Handles batch jobs up to 10TB input data

**BigQuery**:
- Automatically scales to handle query load
- No manual scaling required
- Tested with tables up to 1 billion rows

### Performance Benchmarks

**Streaming Pipeline**:
- Latency: p50 < 30 seconds, p99 < 2 minutes
- Throughput: 10,000 events/second sustained
- Cost: ~$0.05 per million events

**Batch Processing**:
- 1TB of data processed in ~45 minutes
- Cost: ~$2 per TB with preemptible workers
- Daily batch completes within 2-hour SLA

**BigQuery Queries**:
- Simple aggregations: < 1 second
- Complex joins (1B rows): < 10 seconds
- Full table scan (100GB): < 30 seconds

## Disaster Recovery

### Backup Strategy

**BigQuery**:
- Snapshot exports to Cloud Storage (weekly)
- Cross-region replication for critical tables
- 7-day time travel for point-in-time recovery

**Pipeline Code**:
- Version controlled in Git
- Terraform state backed up to GCS
- Multi-region bucket for state files

### Recovery Procedures

**Data Loss Scenario**:
1. Identify last known good state (via data quality checks)
2. Stop all pipelines
3. Restore BigQuery tables from snapshot
4. Replay Pub/Sub messages from retention period
5. Resume pipelines with monitoring

**Service Outage**:
1. Automatic failover to backup region (if configured)
2. Pub/Sub retains messages during outage
3. Pipelines auto-resume on service recovery
4. Data quality validation post-recovery

### RTO and RPO

- Recovery Time Objective (RTO): 2 hours
- Recovery Point Objective (RPO): 24 hours (Pub/Sub retention)

## Cost Management

### Current Cost Breakdown (Moderate Usage)

- BigQuery: $500/month (storage + queries)
- Dataflow: $1,200/month (24/7 streaming)
- Dataproc: $600/month (daily batch with preemptible)
- Cloud Composer: $400/month (small environment)
- Pub/Sub: $100/month
- Storage: $50/month
- Monitoring/Logging: $100/month

**Total**: ~$3,000/month for production workload

### Optimization Strategies

1. Use preemptible VMs for fault-tolerant workloads (60% savings)
2. Partition and cluster BigQuery tables (70% query cost reduction)
3. Set table expiration for temporary data
4. Use BigQuery BI Engine for dashboard queries
5. Scheduled cluster shutdown for idle periods

## Future Enhancements

### Phase 2 (Next 6 Months)

1. **Machine Learning Integration**
   - Vertex AI pipelines for model training
   - Real-time feature serving
   - ML-based anomaly detection

2. **Multi-Region Deployment**
   - Active-active architecture
   - Reduced latency for global users
   - Enhanced disaster recovery

3. **Data Catalog**
   - Automated data discovery
   - Lineage tracking
   - PII detection and masking

### Phase 3 (12+ Months)

1. **Real-Time OLAP**
   - Sub-second query responses
   - In-memory caching layer
   - Streaming aggregations

2. **Advanced Data Governance**
   - Fine-grained access control
   - Dynamic data masking
   - Compliance reporting (GDPR, CCPA)

## Conclusion

This architecture provides a solid foundation for scalable, reliable data engineering on GCP. The design emphasizes data quality, observability, and cost-efficiency while maintaining flexibility for future growth.

For questions or suggestions, contact the Data Engineering team.
