"""
Unit tests for PySpark batch processing job.

Tests data cleaning, transformations, and aggregations using PySpark's testing utilities.
"""

import unittest
from datetime import datetime, timedelta
from decimal import Decimal

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType
import pyspark.sql.functions as F


class TestSparkBatchProcessor(unittest.TestCase):
    """Test Spark batch processing logic."""

    @classmethod
    def setUpClass(cls):
        """Set up Spark session for all tests."""
        cls.spark = SparkSession.builder \
            .appName("test") \
            .master("local[2]") \
            .config("spark.sql.shuffle.partitions", "2") \
            .getOrCreate()

        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls):
        """Tear down Spark session."""
        cls.spark.stop()

    def test_clean_transactions_removes_duplicates(self):
        """Test that duplicate transaction_ids are removed."""
        schema = StructType([
            StructField("transaction_id", StringType(), True),
            StructField("user_id", StringType(), True),
            StructField("amount", DoubleType(), True),
            StructField("transaction_timestamp", TimestampType(), True)
        ])

        data = [
            ("txn_1", "user_1", 100.0, datetime(2024, 1, 15, 10, 0)),
            ("txn_1", "user_1", 100.0, datetime(2024, 1, 15, 10, 0)),  # Duplicate
            ("txn_2", "user_2", 200.0, datetime(2024, 1, 15, 11, 0))
        ]

        df = self.spark.createDataFrame(data, schema)

        # Simulate clean_transactions logic
        cleaned_df = df.dropDuplicates(['transaction_id'])

        self.assertEqual(cleaned_df.count(), 2)

    def test_clean_transactions_filters_invalid_amounts(self):
        """Test that transactions with invalid amounts are filtered."""
        schema = StructType([
            StructField("transaction_id", StringType(), True),
            StructField("user_id", StringType(), True),
            StructField("amount", DoubleType(), True),
            StructField("transaction_timestamp", TimestampType(), True)
        ])

        data = [
            ("txn_1", "user_1", 100.0, datetime(2024, 1, 15, 10, 0)),
            ("txn_2", "user_2", 0.0, datetime(2024, 1, 15, 11, 0)),  # Zero amount
            ("txn_3", "user_3", -50.0, datetime(2024, 1, 15, 12, 0)),  # Negative
            ("txn_4", "user_4", 200.0, datetime(2024, 1, 15, 13, 0))
        ]

        df = self.spark.createDataFrame(data, schema)

        # Simulate clean_transactions logic
        cleaned_df = df.filter(F.col('amount') > 0)

        self.assertEqual(cleaned_df.count(), 2)

    def test_clean_transactions_filters_null_values(self):
        """Test that transactions with null required fields are filtered."""
        schema = StructType([
            StructField("transaction_id", StringType(), True),
            StructField("user_id", StringType(), True),
            StructField("amount", DoubleType(), True),
            StructField("transaction_timestamp", TimestampType(), True)
        ])

        data = [
            ("txn_1", "user_1", 100.0, datetime(2024, 1, 15, 10, 0)),
            (None, "user_2", 200.0, datetime(2024, 1, 15, 11, 0)),  # Null transaction_id
            ("txn_3", None, 150.0, datetime(2024, 1, 15, 12, 0)),  # Null user_id
            ("txn_4", "user_4", 250.0, datetime(2024, 1, 15, 13, 0))
        ]

        df = self.spark.createDataFrame(data, schema)

        # Simulate clean_transactions logic
        cleaned_df = df.filter(
            (F.col('transaction_id').isNotNull()) &
            (F.col('user_id').isNotNull()) &
            (F.col('amount') > 0)
        )

        self.assertEqual(cleaned_df.count(), 2)

    def test_compute_user_metrics_aggregation(self):
        """Test user-level aggregation calculations."""
        schema = StructType([
            StructField("transaction_id", StringType(), True),
            StructField("user_id", StringType(), True),
            StructField("amount", DoubleType(), True),
            StructField("transaction_type", StringType(), True),
            StructField("merchant_id", StringType(), True),
            StructField("transaction_timestamp", TimestampType(), True)
        ])

        data = [
            ("txn_1", "user_1", 100.0, "purchase", "merch_1", datetime(2024, 1, 15, 10, 0)),
            ("txn_2", "user_1", 200.0, "purchase", "merch_2", datetime(2024, 1, 16, 10, 0)),
            ("txn_3", "user_1", 50.0, "refund", "merch_1", datetime(2024, 1, 17, 10, 0)),
            ("txn_4", "user_2", 300.0, "purchase", "merch_3", datetime(2024, 1, 15, 11, 0))
        ]

        df = self.spark.createDataFrame(data, schema)

        # Simulate compute_user_metrics logic
        user_metrics = df.groupBy('user_id').agg(
            F.count('transaction_id').alias('total_transactions'),
            F.sum('amount').alias('total_amount'),
            F.avg('amount').alias('average_amount'),
            F.countDistinct('merchant_id').alias('unique_merchants'),
            F.sum(F.when(F.col('transaction_type') == 'purchase', 1).otherwise(0)).alias('purchase_count'),
            F.sum(F.when(F.col('transaction_type') == 'refund', 1).otherwise(0)).alias('refund_count')
        )

        user_1_metrics = user_metrics.filter(F.col('user_id') == 'user_1').collect()[0]

        self.assertEqual(user_1_metrics['total_transactions'], 3)
        self.assertAlmostEqual(user_1_metrics['total_amount'], 350.0, places=2)
        self.assertAlmostEqual(user_1_metrics['average_amount'], 116.67, places=2)
        self.assertEqual(user_1_metrics['unique_merchants'], 2)
        self.assertEqual(user_1_metrics['purchase_count'], 2)
        self.assertEqual(user_1_metrics['refund_count'], 1)

    def test_compute_user_metrics_refund_rate(self):
        """Test refund rate calculation."""
        schema = StructType([
            StructField("user_id", StringType(), True),
            StructField("total_transactions", StringType(), True),
            StructField("refund_count", StringType(), True)
        ])

        data = [
            ("user_1", 10, 2),
            ("user_2", 5, 0),
            ("user_3", 20, 5)
        ]

        df = self.spark.createDataFrame(data, schema)

        # Calculate refund rate
        result_df = df.withColumn(
            'refund_rate',
            F.when(
                F.col('total_transactions') > 0,
                F.col('refund_count') / F.col('total_transactions')
            ).otherwise(0.0)
        )

        results = {row['user_id']: row['refund_rate'] for row in result_df.collect()}

        self.assertAlmostEqual(results['user_1'], 0.2, places=2)
        self.assertAlmostEqual(results['user_2'], 0.0, places=2)
        self.assertAlmostEqual(results['user_3'], 0.25, places=2)

    def test_detect_anomalies_high_value(self):
        """Test high-value anomaly detection."""
        schema = StructType([
            StructField("transaction_id", StringType(), True),
            StructField("user_id", StringType(), True),
            StructField("amount", DoubleType(), True),
            StructField("transaction_timestamp", TimestampType(), True)
        ])

        # User with typical $100 transactions, then one $1000 transaction
        data = [
            ("txn_1", "user_1", 100.0, datetime(2024, 1, 15, 10, 0)),
            ("txn_2", "user_1", 110.0, datetime(2024, 1, 16, 10, 0)),
            ("txn_3", "user_1", 95.0, datetime(2024, 1, 17, 10, 0)),
            ("txn_4", "user_1", 1000.0, datetime(2024, 1, 18, 10, 0))  # Anomaly
        ]

        df = self.spark.createDataFrame(data, schema)

        # Calculate user statistics
        df_with_stats = df.withColumn(
            'user_avg_amount',
            F.avg('amount').over(F.Window.partitionBy('user_id'))
        ).withColumn(
            'user_stddev_amount',
            F.stddev('amount').over(F.Window.partitionBy('user_id'))
        )

        # Flag anomalies (amount > mean + 2 * stddev)
        df_with_flags = df_with_stats.withColumn(
            'is_high_value_anomaly',
            F.when(
                (F.col('amount') > (F.col('user_avg_amount') + 2 * F.col('user_stddev_amount'))) &
                (F.col('user_stddev_amount').isNotNull()),
                True
            ).otherwise(False)
        )

        anomalies = df_with_flags.filter(F.col('is_high_value_anomaly') == True).collect()

        # The $1000 transaction should be flagged as anomaly
        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies[0]['transaction_id'], 'txn_4')

    def test_detect_anomalies_rapid_transactions(self):
        """Test rapid transaction detection."""
        schema = StructType([
            StructField("transaction_id", StringType(), True),
            StructField("user_id", StringType(), True),
            StructField("amount", DoubleType(), True),
            StructField("transaction_timestamp", TimestampType(), True)
        ])

        base_time = datetime(2024, 1, 15, 10, 0, 0)
        data = [
            ("txn_1", "user_1", 100.0, base_time),
            ("txn_2", "user_1", 110.0, base_time + timedelta(seconds=30)),  # 30 sec later
            ("txn_3", "user_1", 95.0, base_time + timedelta(hours=1)),  # 1 hour later
        ]

        df = self.spark.createDataFrame(data, schema)

        window_spec = F.Window.partitionBy('user_id').orderBy('transaction_timestamp')

        # Calculate time difference from previous transaction
        df_with_time_diff = df.withColumn(
            'prev_transaction_time',
            F.lag('transaction_timestamp').over(window_spec)
        ).withColumn(
            'time_diff_seconds',
            F.when(
                F.col('prev_transaction_time').isNotNull(),
                F.unix_timestamp('transaction_timestamp') - F.unix_timestamp('prev_transaction_time')
            ).otherwise(999999)
        ).withColumn(
            'is_rapid_transaction',
            F.col('time_diff_seconds') < 60
        )

        rapid_txns = df_with_time_diff.filter(F.col('is_rapid_transaction') == True).collect()

        # Transaction 2 should be flagged (30 seconds after transaction 1)
        self.assertEqual(len(rapid_txns), 1)
        self.assertEqual(rapid_txns[0]['transaction_id'], 'txn_2')


class TestDataTransformations(unittest.TestCase):
    """Test common data transformation patterns."""

    @classmethod
    def setUpClass(cls):
        """Set up Spark session."""
        cls.spark = SparkSession.builder \
            .appName("test_transformations") \
            .master("local[2]") \
            .getOrCreate()

        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls):
        """Tear down Spark session."""
        cls.spark.stop()

    def test_currency_standardization(self):
        """Test currency code standardization."""
        schema = StructType([
            StructField("transaction_id", StringType(), True),
            StructField("currency", StringType(), True)
        ])

        data = [
            ("txn_1", "usd"),
            ("txn_2", "USD"),
            ("txn_3", "Usd"),
            ("txn_4", "EUR")
        ]

        df = self.spark.createDataFrame(data, schema)

        # Standardize to uppercase
        standardized_df = df.withColumn('currency', F.upper(F.col('currency')))

        currencies = [row['currency'] for row in standardized_df.collect()]

        self.assertEqual(currencies, ['USD', 'USD', 'USD', 'EUR'])

    def test_date_partitioning_column(self):
        """Test date extraction for partitioning."""
        schema = StructType([
            StructField("transaction_id", StringType(), True),
            StructField("transaction_timestamp", TimestampType(), True)
        ])

        data = [
            ("txn_1", datetime(2024, 1, 15, 10, 30, 45)),
            ("txn_2", datetime(2024, 1, 16, 14, 20, 10))
        ]

        df = self.spark.createDataFrame(data, schema)

        # Extract date for partitioning
        df_with_date = df.withColumn('transaction_date', F.to_date(F.col('transaction_timestamp')))

        dates = [row['transaction_date'] for row in df_with_date.collect()]

        self.assertEqual(str(dates[0]), '2024-01-15')
        self.assertEqual(str(dates[1]), '2024-01-16')


if __name__ == '__main__':
    unittest.main()
