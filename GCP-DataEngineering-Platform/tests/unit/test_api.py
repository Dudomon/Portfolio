"""
Unit tests for FastAPI REST API endpoints.

Tests API request/response handling, validation, and error cases using FastAPI's TestClient.
"""

import unittest
from datetime import datetime, date
from unittest.mock import Mock, patch, MagicMock

from fastapi.testclient import TestClient
from google.cloud import bigquery

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../api/src')))

from main import app, get_bigquery_client


class TestAPIEndpoints(unittest.TestCase):
    """Test API endpoint functionality."""

    def setUp(self):
        """Set up test client and mocks."""
        self.client = TestClient(app)
        self.mock_bq_client = Mock(spec=bigquery.Client)

        # Override BigQuery client dependency
        def override_get_bigquery_client():
            return self.mock_bq_client

        app.dependency_overrides[get_bigquery_client] = override_get_bigquery_client

    def tearDown(self):
        """Clean up dependency overrides."""
        app.dependency_overrides.clear()

    def test_root_endpoint(self):
        """Test root endpoint returns service information."""
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['service'], 'Data Engineering Platform API')
        self.assertEqual(data['version'], '1.0.0')
        self.assertEqual(data['status'], 'operational')

    def test_health_check_success(self):
        """Test health check endpoint when BigQuery is accessible."""
        # Mock successful BigQuery query
        mock_query_job = Mock()
        mock_query_job.result.return_value = [{'health_check': 1}]
        self.mock_bq_client.query.return_value = mock_query_job

        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], 'healthy')
        self.assertEqual(data['bigquery'], 'connected')

    def test_health_check_failure(self):
        """Test health check endpoint when BigQuery is not accessible."""
        # Mock BigQuery failure
        self.mock_bq_client.query.side_effect = Exception("Connection failed")

        response = self.client.get("/health")

        self.assertEqual(response.status_code, 503)
        data = response.json()
        self.assertEqual(data['detail'], 'Service unhealthy')

    def test_get_user_metrics_success(self):
        """Test retrieving user metrics for existing user."""
        # Mock BigQuery response
        mock_row = {
            'user_id': 'user_123',
            'total_transactions': 50,
            'total_amount': 5000.0,
            'average_amount': 100.0,
            'first_transaction_date': datetime(2024, 1, 1, 10, 0),
            'last_transaction_date': datetime(2024, 1, 15, 10, 0),
            'unique_merchants': 10,
            'purchase_count': 45,
            'refund_count': 5,
            'refund_rate': 0.1,
            'days_active': 14
        }

        mock_result = [mock_row]
        mock_query_job = Mock()
        mock_query_job.result.return_value = mock_result
        self.mock_bq_client.query.return_value = mock_query_job

        response = self.client.get("/api/v1/users/user_123/metrics")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['user_id'], 'user_123')
        self.assertEqual(data['total_transactions'], 50)
        self.assertEqual(data['total_amount'], 5000.0)

    def test_get_user_metrics_not_found(self):
        """Test retrieving user metrics for non-existent user."""
        # Mock empty BigQuery response
        mock_query_job = Mock()
        mock_query_job.result.return_value = []
        self.mock_bq_client.query.return_value = mock_query_job

        response = self.client.get("/api/v1/users/nonexistent_user/metrics")

        self.assertEqual(response.status_code, 404)
        data = response.json()
        self.assertIn('not found', data['detail'])

    def test_get_top_users_default_parameters(self):
        """Test retrieving top users with default parameters."""
        # Mock BigQuery response
        mock_rows = [
            {
                'user_id': 'user_1',
                'total_transactions': 100,
                'total_amount': 10000.0,
                'average_amount': 100.0,
                'first_transaction_date': datetime(2024, 1, 1),
                'last_transaction_date': datetime(2024, 1, 15),
                'unique_merchants': 20,
                'purchase_count': 90,
                'refund_count': 10,
                'refund_rate': 0.1,
                'days_active': 14
            },
            {
                'user_id': 'user_2',
                'total_transactions': 80,
                'total_amount': 8000.0,
                'average_amount': 100.0,
                'first_transaction_date': datetime(2024, 1, 1),
                'last_transaction_date': datetime(2024, 1, 15),
                'unique_merchants': 15,
                'purchase_count': 75,
                'refund_count': 5,
                'refund_rate': 0.0625,
                'days_active': 14
            }
        ]

        mock_query_job = Mock()
        mock_query_job.result.return_value = mock_rows
        self.mock_bq_client.query.return_value = mock_query_job

        response = self.client.get("/api/v1/users/top")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]['user_id'], 'user_1')
        self.assertEqual(data[1]['user_id'], 'user_2')

    def test_get_top_users_custom_parameters(self):
        """Test retrieving top users with custom limit and order_by."""
        mock_query_job = Mock()
        mock_query_job.result.return_value = []
        self.mock_bq_client.query.return_value = mock_query_job

        response = self.client.get("/api/v1/users/top?limit=5&order_by=total_transactions")

        self.assertEqual(response.status_code, 200)

    def test_get_top_users_invalid_order_by(self):
        """Test that invalid order_by parameter is rejected."""
        response = self.client.get("/api/v1/users/top?order_by=invalid_column")

        self.assertEqual(response.status_code, 422)  # Validation error

    def test_get_top_users_invalid_limit(self):
        """Test that invalid limit parameter is rejected."""
        response = self.client.get("/api/v1/users/top?limit=200")  # Max is 100

        self.assertEqual(response.status_code, 422)  # Validation error

    def test_get_merchant_metrics_success(self):
        """Test retrieving merchant metrics."""
        mock_row = {
            'merchant_id': 'merch_123',
            'total_transactions': 500,
            'unique_customers': 100,
            'total_revenue': 50000.0,
            'average_transaction_value': 100.0,
            'first_transaction_date': datetime(2024, 1, 1),
            'last_transaction_date': datetime(2024, 1, 15)
        }

        mock_query_job = Mock()
        mock_query_job.result.return_value = [mock_row]
        self.mock_bq_client.query.return_value = mock_query_job

        response = self.client.get("/api/v1/merchants/merch_123/metrics")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['merchant_id'], 'merch_123')
        self.assertEqual(data['total_transactions'], 500)

    def test_get_data_quality_status(self):
        """Test retrieving data quality validation status."""
        mock_rows = [
            {
                'checkpoint_name': 'transactions_checkpoint',
                'success': True,
                'success_percent': 98.5,
                'evaluated_expectations': 20,
                'successful_expectations': 19,
                'unsuccessful_expectations': 1,
                'validation_timestamp': datetime(2024, 1, 15, 10, 0)
            }
        ]

        mock_query_job = Mock()
        mock_query_job.result.return_value = mock_rows
        self.mock_bq_client.query.return_value = mock_query_job

        response = self.client.get("/api/v1/data-quality/status")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]['checkpoint_name'], 'transactions_checkpoint')
        self.assertTrue(data[0]['success'])

    def test_get_data_quality_status_custom_hours(self):
        """Test retrieving data quality status with custom time range."""
        mock_query_job = Mock()
        mock_query_job.result.return_value = []
        self.mock_bq_client.query.return_value = mock_query_job

        response = self.client.get("/api/v1/data-quality/status?hours=48")

        self.assertEqual(response.status_code, 200)

    def test_get_pipeline_metrics(self):
        """Test retrieving pipeline performance metrics."""
        mock_rows = [
            {
                'pipeline_name': 'streaming-pipeline',
                'metric_type': 'latency',
                'metric_value': 45.5,
                'metric_unit': 'seconds',
                'metric_timestamp': datetime(2024, 1, 15, 10, 0)
            }
        ]

        mock_query_job = Mock()
        mock_query_job.result.return_value = mock_rows
        self.mock_bq_client.query.return_value = mock_query_job

        response = self.client.get("/api/v1/pipelines/metrics")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]['pipeline_name'], 'streaming-pipeline')

    def test_get_pipeline_metrics_with_filters(self):
        """Test retrieving pipeline metrics with filters."""
        mock_query_job = Mock()
        mock_query_job.result.return_value = []
        self.mock_bq_client.query.return_value = mock_query_job

        response = self.client.get(
            "/api/v1/pipelines/metrics?pipeline_name=streaming-pipeline&metric_type=latency&hours=12"
        )

        self.assertEqual(response.status_code, 200)

    def test_get_daily_summary(self):
        """Test retrieving daily transaction summary."""
        mock_rows = [
            {
                'transaction_date': date(2024, 1, 15),
                'active_users': 1000,
                'total_transactions': 5000,
                'total_volume': 500000.0,
                'avg_transaction_value': 100.0
            }
        ]

        mock_query_job = Mock()
        mock_query_job.result.return_value = mock_rows
        self.mock_bq_client.query.return_value = mock_query_job

        response = self.client.get("/api/v1/analytics/daily-summary")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]['active_users'], 1000)

    def test_get_daily_summary_with_date_range(self):
        """Test retrieving daily summary with custom date range."""
        mock_query_job = Mock()
        mock_query_job.result.return_value = []
        self.mock_bq_client.query.return_value = mock_query_job

        response = self.client.get(
            "/api/v1/analytics/daily-summary?start_date=2024-01-01&end_date=2024-01-31"
        )

        self.assertEqual(response.status_code, 200)


class TestAPIErrorHandling(unittest.TestCase):
    """Test API error handling."""

    def setUp(self):
        """Set up test client."""
        self.client = TestClient(app)
        self.mock_bq_client = Mock(spec=bigquery.Client)

        def override_get_bigquery_client():
            return self.mock_bq_client

        app.dependency_overrides[get_bigquery_client] = override_get_bigquery_client

    def tearDown(self):
        """Clean up dependency overrides."""
        app.dependency_overrides.clear()

    def test_internal_server_error_handling(self):
        """Test that internal errors return 500 status code."""
        # Mock BigQuery to raise unexpected exception
        self.mock_bq_client.query.side_effect = Exception("Unexpected error")

        response = self.client.get("/api/v1/users/user_123/metrics")

        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertEqual(data['detail'], 'Internal server error')

    def test_not_found_endpoint(self):
        """Test that invalid endpoints return 404."""
        response = self.client.get("/api/v1/nonexistent/endpoint")

        self.assertEqual(response.status_code, 404)


if __name__ == '__main__':
    unittest.main()
