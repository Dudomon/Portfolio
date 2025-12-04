"""
Airflow DAG for running scheduled data quality validations using Great Expectations.

This DAG demonstrates:
- Integration with Great Expectations
- Data quality validation on multiple datasets
- Alert generation on quality failures
- Quality metrics storage in BigQuery
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule


default_args = {
    'owner': 'data-engineering-team',
    'depends_on_past': False,
    'email': ['data-team@example.com'],
    'email_on_failure': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

PROJECT_ID = 'your-project-id'


def run_great_expectations_checkpoint(checkpoint_name: str, **context):
    """
    Execute a Great Expectations checkpoint.

    Args:
        checkpoint_name: Name of the checkpoint to run
        context: Airflow context
    """
    import great_expectations as gx
    from google.cloud import bigquery

    # Initialize Great Expectations context
    context_root_dir = '/home/airflow/gcs/data/great_expectations'
    ge_context = gx.get_context(context_root_dir=context_root_dir)

    # Run checkpoint
    results = ge_context.run_checkpoint(checkpoint_name=checkpoint_name)

    # Extract validation results
    validation_result = {
        'validation_id': str(results.run_id),
        'checkpoint_name': checkpoint_name,
        'success': results.success,
        'evaluated_expectations': results.statistics['evaluated_expectations'],
        'successful_expectations': results.statistics['successful_expectations'],
        'unsuccessful_expectations': results.statistics['unsuccessful_expectations'],
        'success_percent': results.statistics['success_percent'],
        'validation_timestamp': datetime.utcnow().isoformat(),
    }

    # Store results in BigQuery
    client = bigquery.Client(project=PROJECT_ID)
    table_id = f'{PROJECT_ID}.data_quality.validation_results'

    errors = client.insert_rows_json(table_id, [validation_result])
    if errors:
        raise RuntimeError(f'Failed to store validation results: {errors}')

    # Raise exception if validation failed
    if not results.success:
        raise ValueError(
            f'Data quality validation failed for checkpoint: {checkpoint_name}. '
            f'Success rate: {validation_result["success_percent"]}%'
        )

    return validation_result


with DAG(
    'data_quality_pipeline',
    default_args=default_args,
    description='Scheduled data quality validations',
    schedule_interval='0 */6 * * *',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['data-quality', 'great-expectations', 'monitoring'],
) as dag:

    start = EmptyOperator(task_id='start')

    # Validate raw transactions
    validate_transactions = PythonOperator(
        task_id='validate_raw_transactions',
        python_callable=run_great_expectations_checkpoint,
        op_kwargs={'checkpoint_name': 'transactions_checkpoint'},
    )

    # Validate user events
    validate_events = PythonOperator(
        task_id='validate_user_events',
        python_callable=run_great_expectations_checkpoint,
        op_kwargs={'checkpoint_name': 'user_events_checkpoint'},
    )

    # Validate analytics tables
    validate_analytics = PythonOperator(
        task_id='validate_analytics_tables',
        python_callable=run_great_expectations_checkpoint,
        op_kwargs={'checkpoint_name': 'analytics_checkpoint'},
    )

    # Compute quality scores
    compute_quality_scores = BigQueryInsertJobOperator(
        task_id='compute_quality_scores',
        configuration={
            'query': {
                'query': f"""
                    CREATE OR REPLACE TABLE `{PROJECT_ID}.data_quality.daily_quality_scores`
                    AS
                    SELECT
                        DATE(validation_timestamp) as validation_date,
                        checkpoint_name,
                        AVG(success_percent) as avg_success_percent,
                        COUNT(*) as validation_count,
                        SUM(CASE WHEN success = true THEN 1 ELSE 0 END) as successful_runs,
                        CURRENT_TIMESTAMP() as calculation_timestamp
                    FROM `{PROJECT_ID}.data_quality.validation_results`
                    WHERE DATE(validation_timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
                    GROUP BY validation_date, checkpoint_name
                    ORDER BY validation_date DESC, checkpoint_name
                """,
                'useLegacySql': False,
            }
        },
        trigger_rule=TriggerRule.ALL_DONE,
    )

    end = EmptyOperator(task_id='end', trigger_rule=TriggerRule.ALL_DONE)

    start >> [validate_transactions, validate_events, validate_analytics]
    [validate_transactions, validate_events, validate_analytics] >> compute_quality_scores >> end
