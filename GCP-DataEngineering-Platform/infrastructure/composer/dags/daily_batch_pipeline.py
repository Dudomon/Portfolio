"""
Airflow DAG for orchestrating daily batch data processing pipeline.

This DAG demonstrates:
- Dataproc cluster management (create, submit jobs, delete)
- Dataflow job submission and monitoring
- BigQuery data quality checks
- Task dependencies and error handling
- SLA monitoring
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.google.cloud.operators.dataproc import (
    DataprocCreateClusterOperator,
    DataprocSubmitJobOperator,
    DataprocDeleteClusterOperator
)
from airflow.providers.google.cloud.operators.dataflow import DataflowTemplatedJobStartOperator
from airflow.providers.google.cloud.operators.bigquery import (
    BigQueryCheckOperator,
    BigQueryExecuteQueryOperator,
    BigQueryTableCheckOperator
)
from airflow.providers.google.cloud.sensors.bigquery import BigQueryTableExistenceSensor
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule


# Default arguments for the DAG
default_args = {
    'owner': 'data-engineering-team',
    'depends_on_past': False,
    'email': ['data-team@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'sla': timedelta(hours=4),
}

# GCP configuration
PROJECT_ID = 'your-project-id'
REGION = 'us-central1'
DATAPROC_CLUSTER_NAME = 'daily-batch-cluster-{{ ds_nodash }}'
BUCKET_NAME = f'{PROJECT_ID}-dataproc-staging'

# Dataproc cluster configuration
CLUSTER_CONFIG = {
    'master_config': {
        'num_instances': 1,
        'machine_type_uri': 'n1-standard-4',
        'disk_config': {'boot_disk_type': 'pd-standard', 'boot_disk_size_gb': 100},
    },
    'worker_config': {
        'num_instances': 2,
        'machine_type_uri': 'n1-standard-4',
        'disk_config': {'boot_disk_type': 'pd-standard', 'boot_disk_size_gb': 100},
    },
    'secondary_worker_config': {
        'num_instances': 2,
        'machine_type_uri': 'n1-standard-4',
        'disk_config': {'boot_disk_type': 'pd-standard', 'boot_disk_size_gb': 100},
        'is_preemptible': True,
    },
    'software_config': {
        'image_version': '2.1-debian11',
        'properties': {
            'spark:spark.executor.memory': '4g',
            'spark:spark.driver.memory': '4g',
            'spark:spark.sql.adaptive.enabled': 'true',
        },
    },
}

# PySpark job configuration
PYSPARK_JOB = {
    'reference': {'project_id': PROJECT_ID},
    'placement': {'cluster_name': DATAPROC_CLUSTER_NAME},
    'pyspark_job': {
        'main_python_file_uri': f'gs://{BUCKET_NAME}/pipelines/spark/batch_processor.py',
        'args': [
            '--project_id', PROJECT_ID,
            '--app_name', 'daily-transaction-processing-{{ ds }}'
        ],
        'jar_file_uris': [
            'gs://spark-lib/bigquery/spark-bigquery-latest_2.12.jar'
        ],
    },
}


def log_pipeline_metrics(**context):
    """
    Log pipeline execution metrics to BigQuery.

    Args:
        context: Airflow context with execution metadata
    """
    from google.cloud import bigquery

    client = bigquery.Client(project=PROJECT_ID)

    execution_date = context['execution_date']
    dag_run = context['dag_run']

    metrics = {
        'dag_id': context['dag'].dag_id,
        'execution_date': execution_date.isoformat(),
        'state': str(dag_run.state),
        'duration_seconds': (datetime.utcnow() - execution_date).total_seconds(),
        'metric_timestamp': datetime.utcnow().isoformat(),
    }

    table_id = f'{PROJECT_ID}.data_quality.pipeline_execution_metrics'

    errors = client.insert_rows_json(table_id, [metrics])
    if errors:
        raise RuntimeError(f'Failed to log metrics: {errors}')


# Define the DAG
with DAG(
    'daily_batch_pipeline',
    default_args=default_args,
    description='Daily batch processing pipeline for transaction analytics',
    schedule_interval='0 2 * * *',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['batch', 'dataproc', 'bigquery', 'production'],
    max_active_runs=1,
) as dag:

    start = EmptyOperator(task_id='start')

    # Check if source data is available
    check_source_data = BigQueryTableExistenceSensor(
        task_id='check_source_data',
        project_id=PROJECT_ID,
        dataset_id='raw_data',
        table_id='transactions',
        poke_interval=300,
        timeout=3600,
    )

    # Validate data freshness
    validate_data_freshness = BigQueryCheckOperator(
        task_id='validate_data_freshness',
        sql=f"""
            SELECT COUNT(*) > 0
            FROM `{PROJECT_ID}.raw_data.transactions`
            WHERE DATE(ingestion_timestamp) = CURRENT_DATE()
        """,
        use_legacy_sql=False,
    )

    # Create Dataproc cluster
    create_cluster = DataprocCreateClusterOperator(
        task_id='create_dataproc_cluster',
        project_id=PROJECT_ID,
        region=REGION,
        cluster_name=DATAPROC_CLUSTER_NAME,
        cluster_config=CLUSTER_CONFIG,
    )

    # Submit PySpark job
    submit_spark_job = DataprocSubmitJobOperator(
        task_id='submit_spark_job',
        job=PYSPARK_JOB,
        region=REGION,
        project_id=PROJECT_ID,
    )

    # Delete Dataproc cluster
    delete_cluster = DataprocDeleteClusterOperator(
        task_id='delete_dataproc_cluster',
        project_id=PROJECT_ID,
        region=REGION,
        cluster_name=DATAPROC_CLUSTER_NAME,
        trigger_rule=TriggerRule.ALL_DONE,
    )

    # Data quality checks on processed data
    check_user_metrics = BigQueryTableCheckOperator(
        task_id='check_user_metrics',
        project_id=PROJECT_ID,
        dataset_id='analytics_data',
        table_id='user_transaction_metrics',
        checks={
            'row_count_check': {
                'check_statement': 'COUNT(*) > 0',
                'partition_clause': None,
            },
            'total_amount_check': {
                'check_statement': 'total_amount >= 0',
                'partition_clause': None,
            },
        },
    )

    # Update derived tables
    update_daily_summary = BigQueryExecuteQueryOperator(
        task_id='update_daily_summary',
        sql=f"""
            CREATE OR REPLACE TABLE `{PROJECT_ID}.analytics_data.daily_transaction_summary`
            PARTITION BY transaction_date
            AS
            SELECT
                transaction_date,
                COUNT(DISTINCT user_id) as active_users,
                COUNT(transaction_id) as total_transactions,
                SUM(amount) as total_volume,
                AVG(amount) as avg_transaction_value,
                CURRENT_TIMESTAMP() as calculation_timestamp
            FROM `{PROJECT_ID}.staging_data.transactions_clean`
            WHERE transaction_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
            GROUP BY transaction_date
        """,
        use_legacy_sql=False,
    )

    # Log pipeline metrics
    log_metrics = PythonOperator(
        task_id='log_pipeline_metrics',
        python_callable=log_pipeline_metrics,
        provide_context=True,
    )

    end = EmptyOperator(task_id='end', trigger_rule=TriggerRule.ALL_SUCCESS)

    # Define task dependencies
    start >> check_source_data >> validate_data_freshness
    validate_data_freshness >> create_cluster >> submit_spark_job >> delete_cluster
    delete_cluster >> check_user_metrics >> update_daily_summary
    update_daily_summary >> log_metrics >> end
