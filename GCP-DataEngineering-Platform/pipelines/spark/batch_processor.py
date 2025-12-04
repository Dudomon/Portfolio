"""
PySpark batch processing job for large-scale data transformations on Dataproc.

This job demonstrates:
- Reading from BigQuery using Spark connector
- Complex transformations and aggregations
- Performance optimization techniques
- Writing results back to BigQuery with partitioning
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType, LongType


class SparkBatchProcessor:
    """
    Batch processor for transaction analytics using PySpark.

    Processes raw transaction data and generates business-ready analytics tables
    including user behavior metrics, merchant statistics, and fraud indicators.
    """

    def __init__(self, spark: SparkSession, project_id: str):
        """
        Initialize the batch processor.

        Args:
            spark: SparkSession instance
            project_id: GCP project ID
        """
        self.spark = spark
        self.project_id = project_id
        self.logger = logging.getLogger(self.__class__.__name__)

    def read_from_bigquery(self, table_name: str) -> DataFrame:
        """
        Read data from BigQuery table.

        Args:
            table_name: Fully qualified table name (project:dataset.table)

        Returns:
            DataFrame containing table data
        """
        self.logger.info(f"Reading from BigQuery table: {table_name}")

        df = self.spark.read \
            .format("bigquery") \
            .option("table", table_name) \
            .option("filter", "ingestion_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)") \
            .load()

        record_count = df.count()
        self.logger.info(f"Loaded {record_count} records from {table_name}")

        return df

    def clean_transactions(self, df: DataFrame) -> DataFrame:
        """
        Clean and standardize transaction data.

        Args:
            df: Raw transaction DataFrame

        Returns:
            Cleaned DataFrame
        """
        self.logger.info("Cleaning transaction data")

        # Remove duplicates based on transaction_id
        df = df.dropDuplicates(['transaction_id'])

        # Filter out invalid transactions
        df = df.filter(
            (F.col('amount') > 0) &
            (F.col('transaction_id').isNotNull()) &
            (F.col('user_id').isNotNull())
        )

        # Standardize currency to uppercase
        df = df.withColumn('currency', F.upper(F.col('currency')))

        # Parse transaction timestamp
        df = df.withColumn(
            'transaction_timestamp',
            F.to_timestamp(F.col('transaction_timestamp'))
        )

        # Add date column for partitioning
        df = df.withColumn('transaction_date', F.to_date(F.col('transaction_timestamp')))

        return df

    def compute_user_metrics(self, df: DataFrame) -> DataFrame:
        """
        Compute user-level transaction metrics.

        Args:
            df: Cleaned transaction DataFrame

        Returns:
            DataFrame with user metrics
        """
        self.logger.info("Computing user metrics")

        # Aggregate metrics per user
        user_metrics = df.groupBy('user_id').agg(
            F.count('transaction_id').alias('total_transactions'),
            F.sum('amount').alias('total_amount'),
            F.avg('amount').alias('average_amount'),
            F.min('transaction_timestamp').alias('first_transaction_date'),
            F.max('transaction_timestamp').alias('last_transaction_date'),
            F.countDistinct('merchant_id').alias('unique_merchants'),
            F.sum(F.when(F.col('transaction_type') == 'purchase', 1).otherwise(0)).alias('purchase_count'),
            F.sum(F.when(F.col('transaction_type') == 'refund', 1).otherwise(0)).alias('refund_count')
        )

        # Calculate days since first transaction
        user_metrics = user_metrics.withColumn(
            'days_active',
            F.datediff(F.col('last_transaction_date'), F.col('first_transaction_date'))
        )

        # Calculate refund rate
        user_metrics = user_metrics.withColumn(
            'refund_rate',
            F.when(
                F.col('total_transactions') > 0,
                F.col('refund_count') / F.col('total_transactions')
            ).otherwise(0.0)
        )

        # Add calculation timestamp
        user_metrics = user_metrics.withColumn(
            'calculation_timestamp',
            F.lit(datetime.now(timezone.utc)).cast(TimestampType())
        )

        return user_metrics

    def compute_merchant_metrics(self, df: DataFrame) -> DataFrame:
        """
        Compute merchant-level transaction metrics.

        Args:
            df: Cleaned transaction DataFrame

        Returns:
            DataFrame with merchant metrics
        """
        self.logger.info("Computing merchant metrics")

        # Filter for records with merchant_id
        df_with_merchant = df.filter(F.col('merchant_id').isNotNull())

        merchant_metrics = df_with_merchant.groupBy('merchant_id').agg(
            F.count('transaction_id').alias('total_transactions'),
            F.countDistinct('user_id').alias('unique_customers'),
            F.sum('amount').alias('total_revenue'),
            F.avg('amount').alias('average_transaction_value'),
            F.min('transaction_timestamp').alias('first_transaction_date'),
            F.max('transaction_timestamp').alias('last_transaction_date')
        )

        # Add calculation timestamp
        merchant_metrics = merchant_metrics.withColumn(
            'calculation_timestamp',
            F.lit(datetime.now(timezone.utc)).cast(TimestampType())
        )

        return merchant_metrics

    def detect_anomalies(self, df: DataFrame) -> DataFrame:
        """
        Detect potential fraudulent transactions using simple heuristics.

        Args:
            df: Cleaned transaction DataFrame

        Returns:
            DataFrame with anomaly flags
        """
        self.logger.info("Detecting transaction anomalies")

        # Calculate per-user statistics
        window_spec = Window.partitionBy('user_id').orderBy('transaction_timestamp')

        df_with_stats = df.withColumn(
            'user_avg_amount',
            F.avg('amount').over(Window.partitionBy('user_id'))
        ).withColumn(
            'user_stddev_amount',
            F.stddev('amount').over(Window.partitionBy('user_id'))
        )

        # Flag anomalies
        df_with_flags = df_with_stats.withColumn(
            'is_high_value_anomaly',
            F.when(
                (F.col('amount') > (F.col('user_avg_amount') + 3 * F.col('user_stddev_amount'))) &
                (F.col('user_stddev_amount').isNotNull()),
                True
            ).otherwise(False)
        )

        # Flag rapid transactions (multiple transactions within 1 minute)
        df_with_flags = df_with_flags.withColumn(
            'prev_transaction_time',
            F.lag('transaction_timestamp').over(window_spec)
        )

        df_with_flags = df_with_flags.withColumn(
            'time_diff_seconds',
            F.when(
                F.col('prev_transaction_time').isNotNull(),
                F.unix_timestamp('transaction_timestamp') - F.unix_timestamp('prev_transaction_time')
            ).otherwise(999999)
        )

        df_with_flags = df_with_flags.withColumn(
            'is_rapid_transaction',
            F.col('time_diff_seconds') < 60
        )

        # Combine anomaly flags
        df_with_flags = df_with_flags.withColumn(
            'anomaly_score',
            F.when(F.col('is_high_value_anomaly'), 0.5).otherwise(0.0) +
            F.when(F.col('is_rapid_transaction'), 0.3).otherwise(0.0)
        )

        # Select relevant columns
        result = df_with_flags.select(
            'transaction_id',
            'user_id',
            'amount',
            'transaction_timestamp',
            'is_high_value_anomaly',
            'is_rapid_transaction',
            'anomaly_score'
        )

        return result

    def write_to_bigquery(
        self,
        df: DataFrame,
        table_name: str,
        partition_field: Optional[str] = None,
        write_mode: str = 'overwrite'
    ):
        """
        Write DataFrame to BigQuery table.

        Args:
            df: DataFrame to write
            table_name: Fully qualified table name (project:dataset.table)
            partition_field: Field to partition by (optional)
            write_mode: Write mode (overwrite, append)
        """
        self.logger.info(f"Writing to BigQuery table: {table_name} (mode: {write_mode})")

        writer = df.write \
            .format("bigquery") \
            .option("table", table_name) \
            .option("writeMethod", "direct") \
            .mode(write_mode)

        if partition_field:
            writer = writer.option("partitionField", partition_field)
            writer = writer.option("partitionType", "DAY")

        writer.save()

        self.logger.info(f"Successfully wrote data to {table_name}")

    def run(self):
        """Execute the complete batch processing workflow."""
        self.logger.info("Starting batch processing job")

        # Read raw transactions
        raw_transactions = self.read_from_bigquery(
            f"{self.project_id}:raw_data.transactions"
        )

        # Clean data
        clean_transactions = self.clean_transactions(raw_transactions)
        clean_transactions.cache()

        # Write cleaned data to staging
        self.write_to_bigquery(
            clean_transactions,
            f"{self.project_id}:staging_data.transactions_clean",
            partition_field="transaction_date",
            write_mode="overwrite"
        )

        # Compute user metrics
        user_metrics = self.compute_user_metrics(clean_transactions)
        self.write_to_bigquery(
            user_metrics,
            f"{self.project_id}:analytics_data.user_transaction_metrics",
            write_mode="overwrite"
        )

        # Compute merchant metrics
        merchant_metrics = self.compute_merchant_metrics(clean_transactions)
        self.write_to_bigquery(
            merchant_metrics,
            f"{self.project_id}:analytics_data.merchant_transaction_metrics",
            write_mode="overwrite"
        )

        # Detect anomalies
        anomalies = self.detect_anomalies(clean_transactions)
        high_risk_anomalies = anomalies.filter(F.col('anomaly_score') >= 0.5)
        self.write_to_bigquery(
            high_risk_anomalies,
            f"{self.project_id}:analytics_data.transaction_anomalies",
            write_mode="overwrite"
        )

        clean_transactions.unpersist()

        self.logger.info("Batch processing job completed successfully")


def main():
    """Main entry point for the Spark job."""
    import argparse

    parser = argparse.ArgumentParser(description='Spark batch processing job')
    parser.add_argument('--project_id', required=True, help='GCP project ID')
    parser.add_argument('--app_name', default='transaction-batch-processor', help='Spark application name')

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Initialize Spark session
    spark = SparkSession.builder \
        .appName(args.app_name) \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.autoBroadcastJoinThreshold", "10MB") \
        .getOrCreate()

    try:
        processor = SparkBatchProcessor(spark, args.project_id)
        processor.run()
    except Exception as e:
        logging.error(f"Job failed with error: {e}", exc_info=True)
        raise
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
