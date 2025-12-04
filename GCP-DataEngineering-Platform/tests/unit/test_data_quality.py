"""
Unit tests for data quality validation configuration.

Tests Great Expectations expectation suite definitions and checkpoint configurations.
"""

import unittest
from unittest.mock import Mock, patch

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../monitoring/data-quality')))

from great_expectations_config import DataQualityExpectations, create_checkpoint_config


class TestDataQualityExpectations(unittest.TestCase):
    """Test expectation suite configurations."""

    def test_transactions_expectations_structure(self):
        """Test that transaction expectations are properly structured."""
        expectations = DataQualityExpectations.get_transactions_expectations()

        self.assertIsInstance(expectations, list)
        self.assertGreater(len(expectations), 0)

        # Verify all expectations have required attributes
        for exp in expectations:
            self.assertIsNotNone(exp.expectation_type)
            self.assertIsNotNone(exp.kwargs)

    def test_transactions_has_required_column_expectations(self):
        """Test that required columns are validated."""
        expectations = DataQualityExpectations.get_transactions_expectations()

        expectation_types = [exp.expectation_type for exp in expectations]

        # Should have column set validation
        self.assertIn('expect_table_columns_to_match_set', expectation_types)

        # Should have uniqueness checks
        self.assertIn('expect_column_values_to_be_unique', expectation_types)

        # Should have null checks
        self.assertIn('expect_column_values_to_not_be_null', expectation_types)

    def test_transactions_has_value_range_expectations(self):
        """Test that value ranges are validated."""
        expectations = DataQualityExpectations.get_transactions_expectations()

        expectation_types = [exp.expectation_type for exp in expectations]

        # Should have range validation
        self.assertIn('expect_column_values_to_be_between', expectation_types)

        # Should have set membership validation
        self.assertIn('expect_column_values_to_be_in_set', expectation_types)

    def test_transactions_amount_validation(self):
        """Test that amount field has proper validation."""
        expectations = DataQualityExpectations.get_transactions_expectations()

        amount_expectations = [
            exp for exp in expectations
            if exp.kwargs.get('column') == 'amount'
        ]

        self.assertGreater(len(amount_expectations), 0)

        # Check for positive amount validation
        range_expectations = [
            exp for exp in amount_expectations
            if exp.expectation_type == 'expect_column_values_to_be_between'
        ]

        self.assertEqual(len(range_expectations), 1)
        self.assertGreater(range_expectations[0].kwargs['min_value'], 0)

    def test_transactions_currency_validation(self):
        """Test that currency field has valid value set."""
        expectations = DataQualityExpectations.get_transactions_expectations()

        currency_expectations = [
            exp for exp in expectations
            if exp.kwargs.get('column') == 'currency'
            and exp.expectation_type == 'expect_column_values_to_be_in_set'
        ]

        self.assertEqual(len(currency_expectations), 1)

        valid_currencies = currency_expectations[0].kwargs['value_set']
        self.assertIn('USD', valid_currencies)
        self.assertIn('EUR', valid_currencies)
        self.assertIn('GBP', valid_currencies)
        self.assertIn('BRL', valid_currencies)

    def test_user_events_expectations_structure(self):
        """Test that user event expectations are properly structured."""
        expectations = DataQualityExpectations.get_user_events_expectations()

        self.assertIsInstance(expectations, list)
        self.assertGreater(len(expectations), 0)

        for exp in expectations:
            self.assertIsNotNone(exp.expectation_type)
            self.assertIsNotNone(exp.kwargs)

    def test_user_events_has_uniqueness_check(self):
        """Test that event_id uniqueness is validated."""
        expectations = DataQualityExpectations.get_user_events_expectations()

        uniqueness_expectations = [
            exp for exp in expectations
            if exp.kwargs.get('column') == 'event_id'
            and exp.expectation_type == 'expect_column_values_to_be_unique'
        ]

        self.assertEqual(len(uniqueness_expectations), 1)

    def test_user_events_device_type_validation(self):
        """Test that device_type has valid value set."""
        expectations = DataQualityExpectations.get_user_events_expectations()

        device_expectations = [
            exp for exp in expectations
            if exp.kwargs.get('column') == 'device_type'
            and exp.expectation_type == 'expect_column_values_to_be_in_set'
        ]

        self.assertEqual(len(device_expectations), 1)

        valid_devices = device_expectations[0].kwargs['value_set']
        self.assertIn('mobile', valid_devices)
        self.assertIn('desktop', valid_devices)
        self.assertIn('tablet', valid_devices)

    def test_user_events_session_id_length_validation(self):
        """Test that session_id has length constraints."""
        expectations = DataQualityExpectations.get_user_events_expectations()

        length_expectations = [
            exp for exp in expectations
            if exp.kwargs.get('column') == 'session_id'
            and exp.expectation_type == 'expect_column_value_lengths_to_be_between'
        ]

        self.assertEqual(len(length_expectations), 1)
        self.assertGreaterEqual(length_expectations[0].kwargs['min_value'], 10)

    def test_analytics_expectations_structure(self):
        """Test that analytics expectations are properly structured."""
        expectations = DataQualityExpectations.get_analytics_expectations()

        self.assertIsInstance(expectations, list)
        self.assertGreater(len(expectations), 0)

    def test_analytics_has_numeric_range_validations(self):
        """Test that numeric fields have range validations."""
        expectations = DataQualityExpectations.get_analytics_expectations()

        range_expectations = [
            exp for exp in expectations
            if exp.expectation_type == 'expect_column_values_to_be_between'
        ]

        self.assertGreater(len(range_expectations), 0)

        # Check that ranges are reasonable
        for exp in range_expectations:
            self.assertGreaterEqual(exp.kwargs['min_value'], 0)
            self.assertIsNotNone(exp.kwargs['max_value'])

    def test_analytics_refund_rate_validation(self):
        """Test that refund_rate is validated as percentage."""
        expectations = DataQualityExpectations.get_analytics_expectations()

        refund_rate_expectations = [
            exp for exp in expectations
            if exp.kwargs.get('column') == 'refund_rate'
        ]

        self.assertEqual(len(refund_rate_expectations), 1)

        exp = refund_rate_expectations[0]
        self.assertEqual(exp.kwargs['min_value'], 0.0)
        self.assertEqual(exp.kwargs['max_value'], 1.0)


class TestCheckpointConfiguration(unittest.TestCase):
    """Test checkpoint configuration generation."""

    def test_create_checkpoint_basic_structure(self):
        """Test that checkpoint config has required structure."""
        config = create_checkpoint_config(
            checkpoint_name='test_checkpoint',
            expectation_suite_name='test_suite',
            table_name='project.dataset.table'
        )

        self.assertEqual(config['name'], 'test_checkpoint')
        self.assertEqual(config['config_version'], 1.0)
        self.assertEqual(config['class_name'], 'Checkpoint')

    def test_create_checkpoint_has_validations(self):
        """Test that checkpoint config includes validations."""
        config = create_checkpoint_config(
            checkpoint_name='test_checkpoint',
            expectation_suite_name='test_suite',
            table_name='project.dataset.table'
        )

        self.assertIn('validations', config)
        self.assertIsInstance(config['validations'], list)
        self.assertEqual(len(config['validations']), 1)

        validation = config['validations'][0]
        self.assertIn('batch_request', validation)
        self.assertIn('expectation_suite_name', validation)
        self.assertEqual(validation['expectation_suite_name'], 'test_suite')

    def test_create_checkpoint_has_batch_request(self):
        """Test that checkpoint includes proper batch request."""
        table_name = 'my-project.my_dataset.my_table'

        config = create_checkpoint_config(
            checkpoint_name='test_checkpoint',
            expectation_suite_name='test_suite',
            table_name=table_name
        )

        batch_request = config['validations'][0]['batch_request']

        self.assertEqual(batch_request['datasource_name'], 'bigquery_datasource')
        self.assertEqual(batch_request['data_asset_name'], table_name)

    def test_create_checkpoint_has_action_list(self):
        """Test that checkpoint includes required actions."""
        config = create_checkpoint_config(
            checkpoint_name='test_checkpoint',
            expectation_suite_name='test_suite',
            table_name='project.dataset.table'
        )

        self.assertIn('action_list', config)
        self.assertIsInstance(config['action_list'], list)
        self.assertGreater(len(config['action_list']), 0)

        action_names = [action['name'] for action in config['action_list']]

        # Should have result storage
        self.assertIn('store_validation_result', action_names)

        # Should have parameter storage
        self.assertIn('store_evaluation_params', action_names)

        # Should update data docs
        self.assertIn('update_data_docs', action_names)

    def test_create_checkpoint_run_name_template(self):
        """Test that run name template includes checkpoint name."""
        checkpoint_name = 'my_checkpoint'

        config = create_checkpoint_config(
            checkpoint_name=checkpoint_name,
            expectation_suite_name='test_suite',
            table_name='project.dataset.table'
        )

        self.assertIn('run_name_template', config)
        self.assertIn(checkpoint_name, config['run_name_template'])


class TestExpectationCoverage(unittest.TestCase):
    """Test that expectations cover all critical validation scenarios."""

    def test_transactions_covers_all_required_fields(self):
        """Test that all required transaction fields are validated."""
        expectations = DataQualityExpectations.get_transactions_expectations()

        required_fields = [
            'transaction_id',
            'user_id',
            'amount',
            'currency',
            'transaction_type',
            'transaction_timestamp'
        ]

        validated_columns = set()
        for exp in expectations:
            if 'column' in exp.kwargs:
                validated_columns.add(exp.kwargs['column'])

        for field in required_fields:
            self.assertIn(
                field,
                validated_columns,
                f"Required field '{field}' not validated in expectations"
            )

    def test_user_events_covers_all_required_fields(self):
        """Test that all required event fields are validated."""
        expectations = DataQualityExpectations.get_user_events_expectations()

        required_fields = [
            'event_id',
            'user_id',
            'session_id',
            'event_type'
        ]

        validated_columns = set()
        for exp in expectations:
            if 'column' in exp.kwargs:
                validated_columns.add(exp.kwargs['column'])

        for field in required_fields:
            self.assertIn(
                field,
                validated_columns,
                f"Required field '{field}' not validated in expectations"
            )

    def test_transactions_has_row_count_validation(self):
        """Test that table row count is validated."""
        expectations = DataQualityExpectations.get_transactions_expectations()

        row_count_expectations = [
            exp for exp in expectations
            if exp.expectation_type == 'expect_table_row_count_to_be_between'
        ]

        self.assertGreater(len(row_count_expectations), 0)


if __name__ == '__main__':
    unittest.main()
