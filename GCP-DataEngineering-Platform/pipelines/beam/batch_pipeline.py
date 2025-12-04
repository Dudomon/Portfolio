"""
Batch pipeline for processing transaction files from Cloud Storage to BigQuery.

This pipeline demonstrates:
- Reading Parquet files from GCS
- Schema validation
- Data transformation and aggregation
- Writing to partitioned BigQuery tables
- Custom metrics and monitoring
"""

import logging
from typing import Dict, Any
from datetime import datetime, timezone

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.io import ReadFromParquet, WriteToBigQuery
from apache_beam.io.gcp.bigquery import BigQueryDisposition
from apache_beam.metrics import Metrics


class ValidateTransaction(beam.DoFn):
    """Validate transaction records."""

    def __init__(self):
        self.valid_count = Metrics.counter(self.__class__, 'valid_transactions')
        self.invalid_count = Metrics.counter(self.__class__, 'invalid_transactions')

    def process(self, transaction: Dict[str, Any]):
        """
        Validate transaction data quality.

        Args:
            transaction: Transaction record from Parquet file

        Yields:
            Valid transaction or tagged error
        """
        errors = []

        # Validate required fields exist
        required_fields = ['transaction_id', 'user_id', 'amount', 'currency', 'transaction_timestamp']
        for field in required_fields:
            if field not in transaction or transaction[field] is None:
                errors.append(f'Missing required field: {field}')

        # Validate amount is positive
        amount = transaction.get('amount')
        if amount is not None:
            try:
                if float(amount) <= 0:
                    errors.append('Amount must be positive')
            except (ValueError, TypeError):
                errors.append('Invalid amount format')

        # Validate currency code
        valid_currencies = ['USD', 'EUR', 'GBP', 'BRL']
        if transaction.get('currency') not in valid_currencies:
            errors.append(f"Invalid currency: {transaction.get('currency')}")

        # Validate transaction type
        valid_types = ['purchase', 'refund', 'chargeback', 'adjustment']
        if transaction.get('transaction_type') not in valid_types:
            errors.append(f"Invalid transaction_type: {transaction.get('transaction_type')}")

        if errors:
            self.invalid_count.inc()
            yield beam.pvalue.TaggedOutput('invalid', {
                'transaction': transaction,
                'validation_errors': errors,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
        else:
            self.valid_count.inc()
            yield transaction


class EnrichTransaction(beam.DoFn):
    """Enrich transaction with derived fields."""

    def __init__(self):
        self.enriched_count = Metrics.counter(self.__class__, 'enriched_transactions')

    def process(self, transaction: Dict[str, Any]):
        """
        Add computed fields to transaction.

        Args:
            transaction: Valid transaction record

        Yields:
            Enriched transaction
        """
        enriched = transaction.copy()

        # Add ingestion timestamp
        enriched['ingestion_timestamp'] = datetime.now(timezone.utc).isoformat()

        # Normalize amount to USD (simplified conversion)
        currency = transaction['currency']
        amount = float(transaction['amount'])
        conversion_rates = {
            'USD': 1.0,
            'EUR': 1.10,
            'GBP': 1.27,
            'BRL': 0.20
        }
        enriched['amount_usd'] = amount * conversion_rates.get(currency, 1.0)

        # Add transaction hour for partitioning
        timestamp = datetime.fromisoformat(
            transaction['transaction_timestamp'].replace('Z', '+00:00')
        )
        enriched['transaction_hour'] = timestamp.strftime('%Y-%m-%d %H:00:00')
        enriched['transaction_date'] = timestamp.strftime('%Y-%m-%d')

        # Flag high-value transactions
        enriched['is_high_value'] = enriched['amount_usd'] > 1000

        self.enriched_count.inc()
        yield enriched


class ComputeTransactionMetrics(beam.DoFn):
    """Compute aggregated metrics for transactions."""

    def process(self, element: tuple):
        """
        Compute metrics per user.

        Args:
            element: Tuple of (user_id, transactions_list)

        Yields:
            User transaction metrics
        """
        user_id, transactions = element
        transactions_list = list(transactions)

        total_amount = sum(float(t['amount_usd']) for t in transactions_list)
        transaction_count = len(transactions_list)
        avg_amount = total_amount / transaction_count if transaction_count > 0 else 0

        transaction_types = {}
        for t in transactions_list:
            tx_type = t['transaction_type']
            transaction_types[tx_type] = transaction_types.get(tx_type, 0) + 1

        yield {
            'user_id': user_id,
            'total_transaction_amount': total_amount,
            'transaction_count': transaction_count,
            'average_transaction_amount': avg_amount,
            'purchase_count': transaction_types.get('purchase', 0),
            'refund_count': transaction_types.get('refund', 0),
            'calculation_timestamp': datetime.now(timezone.utc).isoformat()
        }


def run_batch_pipeline(
    input_pattern: str,
    output_table: str,
    metrics_table: str,
    pipeline_options: PipelineOptions
):
    """
    Execute the batch processing pipeline.

    Args:
        input_pattern: GCS path pattern for input Parquet files
        output_table: BigQuery table for transaction records
        metrics_table: BigQuery table for aggregated metrics
        pipeline_options: Apache Beam pipeline options
    """
    with beam.Pipeline(options=pipeline_options) as pipeline:

        # Read Parquet files from GCS
        transactions = (
            pipeline
            | 'Read from Parquet' >> ReadFromParquet(input_pattern)
        )

        # Validate transactions
        validation_results = (
            transactions
            | 'Validate Transactions' >> beam.ParDo(ValidateTransaction()).with_outputs(
                'invalid', main='valid'
            )
        )

        # Enrich valid transactions
        enriched_transactions = (
            validation_results.valid
            | 'Enrich Transactions' >> beam.ParDo(EnrichTransaction())
        )

        # Write enriched transactions to BigQuery
        _ = (
            enriched_transactions
            | 'Write Transactions to BigQuery' >> WriteToBigQuery(
                table=output_table,
                schema='SCHEMA_AUTODETECT',
                create_disposition=BigQueryDisposition.CREATE_NEVER,
                write_disposition=BigQueryDisposition.WRITE_APPEND,
                additional_bq_parameters={
                    'timePartitioning': {
                        'type': 'DAY',
                        'field': 'transaction_timestamp'
                    },
                    'clustering': {
                        'fields': ['user_id', 'transaction_type']
                    }
                }
            )
        )

        # Compute user-level metrics
        user_metrics = (
            enriched_transactions
            | 'Key by User' >> beam.Map(lambda t: (t['user_id'], t))
            | 'Group by User' >> beam.GroupByKey()
            | 'Compute Metrics' >> beam.ParDo(ComputeTransactionMetrics())
        )

        # Write metrics to BigQuery
        _ = (
            user_metrics
            | 'Write Metrics to BigQuery' >> WriteToBigQuery(
                table=metrics_table,
                schema='SCHEMA_AUTODETECT',
                create_disposition=BigQueryDisposition.CREATE_IF_NEEDED,
                write_disposition=BigQueryDisposition.WRITE_TRUNCATE
            )
        )

        # Log invalid transactions
        _ = (
            validation_results.invalid
            | 'Log Invalid Transactions' >> beam.Map(
                lambda x: logging.error(f"Invalid transaction: {x}")
            )
        )


def main():
    """Main entry point for the batch pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description='Batch transaction processing pipeline')
    parser.add_argument('--project', required=True, help='GCP project ID')
    parser.add_argument('--region', default='us-central1', help='GCP region')
    parser.add_argument('--input_path', required=True, help='GCS input path pattern')
    parser.add_argument('--output_table', required=True, help='BigQuery output table')
    parser.add_argument('--metrics_table', required=True, help='BigQuery metrics table')
    parser.add_argument('--runner', default='DirectRunner', help='Pipeline runner')
    parser.add_argument('--temp_location', help='GCS temp location')
    parser.add_argument('--staging_location', help='GCS staging location')

    args, beam_args = parser.parse_known_args()

    # Configure pipeline options
    pipeline_options = PipelineOptions(
        beam_args,
        project=args.project,
        region=args.region,
        runner=args.runner,
        temp_location=args.temp_location,
        staging_location=args.staging_location,
        save_main_session=True
    )

    logging.getLogger().setLevel(logging.INFO)
    logging.info(f"Starting batch pipeline for input: {args.input_path}")

    # Run the pipeline
    run_batch_pipeline(
        input_pattern=args.input_path,
        output_table=args.output_table,
        metrics_table=args.metrics_table,
        pipeline_options=pipeline_options
    )


if __name__ == '__main__':
    main()
