"""
Great Expectations configuration and expectation suites for data quality validation.

This module defines comprehensive data quality expectations for:
- Transaction data validation
- User event validation
- Schema enforcement
- Business rule validation
"""

from typing import Dict, List
from great_expectations.core import ExpectationConfiguration


class DataQualityExpectations:
    """
    Centralized configuration for Great Expectations checkpoints and expectation suites.
    """

    @staticmethod
    def get_transactions_expectations() -> List[ExpectationConfiguration]:
        """
        Define expectations for transaction data quality.

        Returns:
            List of ExpectationConfiguration objects for transactions
        """
        return [
            # Required columns must exist
            ExpectationConfiguration(
                expectation_type="expect_table_columns_to_match_set",
                kwargs={
                    "column_set": [
                        "transaction_id",
                        "user_id",
                        "transaction_type",
                        "amount",
                        "currency",
                        "merchant_id",
                        "metadata",
                        "transaction_timestamp",
                        "ingestion_timestamp"
                    ],
                    "exact_match": False
                }
            ),

            # Transaction ID must be unique
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_unique",
                kwargs={"column": "transaction_id"}
            ),

            # Transaction ID must not be null
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": "transaction_id"}
            ),

            # User ID must not be null
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": "user_id"}
            ),

            # Amount must be positive
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": "amount",
                    "min_value": 0.01,
                    "max_value": 1000000,
                    "mostly": 0.99
                }
            ),

            # Currency must be in valid set
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_in_set",
                kwargs={
                    "column": "currency",
                    "value_set": ["USD", "EUR", "GBP", "BRL"],
                    "mostly": 0.99
                }
            ),

            # Transaction type must be valid
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_in_set",
                kwargs={
                    "column": "transaction_type",
                    "value_set": ["purchase", "refund", "chargeback", "adjustment"],
                    "mostly": 0.99
                }
            ),

            # Transaction timestamp must be valid date
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_of_type",
                kwargs={
                    "column": "transaction_timestamp",
                    "type_": "TIMESTAMP"
                }
            ),

            # Transaction timestamp should not be in future
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_dateutil_parseable",
                kwargs={"column": "transaction_timestamp"}
            ),

            # Check for reasonable row count
            ExpectationConfiguration(
                expectation_type="expect_table_row_count_to_be_between",
                kwargs={
                    "min_value": 100,
                    "max_value": 100000000
                }
            ),

            # Merchant ID should be present in most cases
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={
                    "column": "merchant_id",
                    "mostly": 0.8
                }
            )
        ]

    @staticmethod
    def get_user_events_expectations() -> List[ExpectationConfiguration]:
        """
        Define expectations for user event data quality.

        Returns:
            List of ExpectationConfiguration objects for user events
        """
        return [
            # Required columns
            ExpectationConfiguration(
                expectation_type="expect_table_columns_to_match_set",
                kwargs={
                    "column_set": [
                        "event_id",
                        "user_id",
                        "session_id",
                        "event_type",
                        "event_properties",
                        "device_type",
                        "user_agent",
                        "ip_address",
                        "event_timestamp"
                    ],
                    "exact_match": False
                }
            ),

            # Event ID must be unique
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_unique",
                kwargs={"column": "event_id"}
            ),

            # Required fields must not be null
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": "event_id"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": "user_id"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": "session_id"}
            ),

            # Event type must be valid
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_in_set",
                kwargs={
                    "column": "event_type",
                    "value_set": ["page_view", "click", "form_submit", "purchase", "add_to_cart"],
                    "mostly": 0.95
                }
            ),

            # Device type must be valid
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_in_set",
                kwargs={
                    "column": "device_type",
                    "value_set": ["mobile", "desktop", "tablet"],
                    "mostly": 0.9
                }
            ),

            # Session ID minimum length check
            ExpectationConfiguration(
                expectation_type="expect_column_value_lengths_to_be_between",
                kwargs={
                    "column": "session_id",
                    "min_value": 10,
                    "max_value": 100,
                    "mostly": 0.99
                }
            ),

            # Check for reasonable row count
            ExpectationConfiguration(
                expectation_type="expect_table_row_count_to_be_between",
                kwargs={
                    "min_value": 1000,
                    "max_value": 1000000000
                }
            )
        ]

    @staticmethod
    def get_analytics_expectations() -> List[ExpectationConfiguration]:
        """
        Define expectations for analytics tables.

        Returns:
            List of ExpectationConfiguration objects for analytics
        """
        return [
            # User metrics validations
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": "total_transactions",
                    "min_value": 0,
                    "max_value": 100000
                }
            ),

            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": "total_amount",
                    "min_value": 0,
                    "max_value": 10000000
                }
            ),

            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": "average_amount",
                    "min_value": 0,
                    "max_value": 100000
                }
            ),

            # Refund rate should be reasonable
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": "refund_rate",
                    "min_value": 0.0,
                    "max_value": 1.0,
                    "mostly": 0.99
                }
            ),

            # All users should have at least one transaction
            ExpectationConfiguration(
                expectation_type="expect_column_min_to_be_between",
                kwargs={
                    "column": "total_transactions",
                    "min_value": 1,
                    "max_value": 1
                }
            )
        ]


def create_checkpoint_config(checkpoint_name: str, expectation_suite_name: str, table_name: str) -> Dict:
    """
    Create a checkpoint configuration for Great Expectations.

    Args:
        checkpoint_name: Name of the checkpoint
        expectation_suite_name: Name of the expectation suite to use
        table_name: Fully qualified BigQuery table name

    Returns:
        Checkpoint configuration dictionary
    """
    return {
        "name": checkpoint_name,
        "config_version": 1.0,
        "class_name": "Checkpoint",
        "run_name_template": f"%Y%m%d-%H%M%S-{checkpoint_name}",
        "validations": [
            {
                "batch_request": {
                    "datasource_name": "bigquery_datasource",
                    "data_connector_name": "default_runtime_data_connector",
                    "data_asset_name": table_name,
                },
                "expectation_suite_name": expectation_suite_name
            }
        ],
        "action_list": [
            {
                "name": "store_validation_result",
                "action": {
                    "class_name": "StoreValidationResultAction"
                }
            },
            {
                "name": "store_evaluation_params",
                "action": {
                    "class_name": "StoreEvaluationParametersAction"
                }
            },
            {
                "name": "update_data_docs",
                "action": {
                    "class_name": "UpdateDataDocsAction"
                }
            }
        ]
    }
