## Testing Guide

## Overview

This document describes the testing strategy, test organization, and procedures for the GCP Data Engineering Platform. The test suite ensures code quality, reliability, and maintainability through comprehensive unit, integration, and end-to-end tests.

## Testing Philosophy

The testing approach follows these principles:

1. **Test Pyramid**: Majority of tests are fast unit tests, with fewer integration tests and minimal end-to-end tests
2. **Test Independence**: Tests can run in any order and don't depend on external state
3. **Clarity**: Test names clearly describe what is being tested and expected behavior
4. **Coverage**: Aim for 75%+ code coverage, focusing on critical paths
5. **Maintainability**: Tests are easy to understand and update as code evolves

## Test Organization

```
tests/
├── __init__.py
├── unit/                           # Unit tests (fast, isolated)
│   ├── test_beam_pipeline.py      # Apache Beam pipeline components
│   ├── test_spark_processor.py    # PySpark transformations
│   ├── test_api.py                # API endpoints and logic
│   └── test_data_quality.py       # Data quality configurations
├── integration/                    # Integration tests (slower, may use mocks)
│   └── test_end_to_end.py         # End-to-end flow testing
└── fixtures/                       # Test data and fixtures
```

## Running Tests

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Run All Tests

```bash
# Run complete test suite with coverage
make test

# Or using pytest directly
pytest tests/ -v --cov --cov-report=html
```

### Run Specific Test Categories

```bash
# Unit tests only (fast)
make test-unit
# or
pytest tests/unit/ -v

# Integration tests only
make test-integration
# or
pytest tests/integration/ -v

# Tests for specific component
pytest tests/unit/test_beam_pipeline.py -v
pytest tests/unit/test_api.py -v
```

### Run Tests with Markers

```bash
# Run only API tests
pytest -m api

# Run only tests that don't require credentials
pytest -m "not bigquery and not dataflow"

# Run fast tests only (skip slow tests)
pytest -m "not slow"
```

## Test Coverage

### View Coverage Report

After running tests with coverage:

```bash
# Terminal summary
pytest --cov --cov-report=term-missing

# HTML report (detailed)
pytest --cov --cov-report=html
# Open htmlcov/index.html in browser
```

### Coverage Requirements

- **Overall**: 75% minimum
- **Critical paths**: 90%+ (authentication, data validation, error handling)
- **New code**: 80%+ for all new features

### Coverage Exemptions

The following are excluded from coverage requirements:
- Configuration files
- Test files themselves
- Generated code
- Example/demo scripts

## Unit Tests

### Apache Beam Pipeline Tests

Location: `tests/unit/test_beam_pipeline.py`

Tests cover:
- Message parsing and JSON validation
- Required field validation
- Data enrichment logic
- Device type detection from user agent
- Data quality validation rules
- BigQuery output formatting

Example:
```python
def test_parse_valid_message(self):
    """Test parsing a valid JSON message."""
    valid_message = {
        'event_id': 'evt_123',
        'user_id': 'user_456',
        'session_id': 'sess_789',
        'event_type': 'page_view',
        'event_timestamp': '2024-01-15T10:30:00Z'
    }

    # Test implementation...
```

Run:
```bash
pytest tests/unit/test_beam_pipeline.py -v
```

### PySpark Processor Tests

Location: `tests/unit/test_spark_processor.py`

Tests cover:
- Transaction deduplication
- Invalid data filtering (negative amounts, nulls)
- User-level aggregations
- Merchant metrics calculations
- Anomaly detection (high-value, rapid transactions)
- Currency standardization
- Date partitioning

Example:
```python
def test_compute_user_metrics_aggregation(self):
    """Test user-level aggregation calculations."""
    # Uses Spark local mode for testing
    df = self.spark.createDataFrame(data, schema)
    user_metrics = df.groupBy('user_id').agg(...)
    # Assertions...
```

Run:
```bash
pytest tests/unit/test_spark_processor.py -v
```

### API Tests

Location: `tests/unit/test_api.py`

Tests cover:
- Endpoint response formats
- Query parameter validation
- Error handling (404, 500, 422)
- BigQuery integration mocking
- Health check functionality

Example:
```python
def test_get_user_metrics_success(self):
    """Test retrieving user metrics for existing user."""
    # Mock BigQuery response
    self.mock_bq_client.query.return_value = mock_result

    response = self.client.get("/api/v1/users/user_123/metrics")
    self.assertEqual(response.status_code, 200)
```

Run:
```bash
pytest tests/unit/test_api.py -v
```

### Data Quality Tests

Location: `tests/unit/test_data_quality.py`

Tests cover:
- Expectation suite completeness
- Validation rule correctness
- Checkpoint configuration structure
- Coverage of all required fields

Example:
```python
def test_transactions_amount_validation(self):
    """Test that amount field has proper validation."""
    expectations = DataQualityExpectations.get_transactions_expectations()
    # Verify positive amount validation exists
```

Run:
```bash
pytest tests/unit/test_data_quality.py -v
```

## Integration Tests

Location: `tests/integration/test_end_to_end.py`

Integration tests validate complete workflows:
- Full pipeline flow (ingestion to storage)
- Data quality integration with pipelines
- Monitoring metrics collection
- API to BigQuery flow
- Error routing and handling

Example:
```python
def test_valid_message_full_pipeline(self):
    """Test that valid messages flow through entire pipeline successfully."""
    test_message = {...}
    # Validate end-to-end processing
```

Run:
```bash
pytest tests/integration/ -v
```

## Writing New Tests

### Test Structure Template

```python
import unittest
from unittest.mock import Mock, patch

class TestFeatureName(unittest.TestCase):
    """Test description."""

    def setUp(self):
        """Set up test fixtures before each test."""
        # Initialize test data, mocks, etc.
        pass

    def tearDown(self):
        """Clean up after each test."""
        # Clean up resources if needed
        pass

    def test_specific_behavior(self):
        """Test that specific behavior works correctly."""
        # Arrange: Set up test data
        test_input = {...}

        # Act: Execute the code under test
        result = function_under_test(test_input)

        # Assert: Verify expected behavior
        self.assertEqual(result, expected_value)
        self.assertIsNotNone(result)

    def test_error_handling(self):
        """Test that errors are handled correctly."""
        with self.assertRaises(ExpectedException):
            function_that_should_fail(invalid_input)
```

### Naming Conventions

- Test files: `test_<module_name>.py`
- Test classes: `Test<FeatureName>`
- Test methods: `test_<specific_behavior>`

### Best Practices

1. **One assertion per test** (or closely related assertions)
2. **Use descriptive test names** that explain what is being tested
3. **Arrange-Act-Assert pattern** for clarity
4. **Mock external dependencies** (GCP services, databases, APIs)
5. **Test both success and failure paths**
6. **Keep tests independent** - no shared state between tests
7. **Use fixtures for common test data**

## Mocking GCP Services

### BigQuery Mock Example

```python
from unittest.mock import Mock
from google.cloud import bigquery

# Create mock client
mock_bq_client = Mock(spec=bigquery.Client)

# Mock query result
mock_result = [{'user_id': 'test', 'count': 10}]
mock_query_job = Mock()
mock_query_job.result.return_value = mock_result
mock_bq_client.query.return_value = mock_query_job

# Use in test
result = mock_bq_client.query("SELECT ...").result()
```

### Pub/Sub Mock Example

```python
# For Beam pipelines, use TestPipeline
from apache_beam.testing.test_pipeline import TestPipeline

with TestPipeline() as p:
    output = (
        p
        | beam.Create([test_message])
        | beam.ParDo(YourTransform())
    )
    assert_that(output, equal_to(expected_output))
```

## Continuous Integration

The project uses GitHub Actions for CI/CD.

### CI Pipeline Steps

1. **Linting**: Flake8 code quality checks
2. **Unit Tests**: Full test suite with coverage
3. **Security**: Bandit and Safety checks
4. **Terraform**: Format and validation checks

Configuration: `.github/workflows/ci.yml`

### Running CI Checks Locally

```bash
# Lint check
make lint

# Format code
make format

# Run tests
make test
```

## Test Data and Fixtures

### Generating Test Data

```bash
# Generate sample data for local testing
python scripts/generate-sample-data.py --num-transactions 100 --num-events 500
```

### Using Test Fixtures

Create fixtures in `tests/fixtures/`:

```python
# tests/fixtures/sample_transactions.py
SAMPLE_TRANSACTIONS = [
    {
        'transaction_id': 'txn_001',
        'user_id': 'user_001',
        'amount': 100.00,
        'currency': 'USD'
    },
    # More test data...
]
```

Import in tests:
```python
from tests.fixtures.sample_transactions import SAMPLE_TRANSACTIONS
```

## Performance Testing

For performance-critical code:

```python
import time

def test_performance_requirement(self):
    """Test that processing completes within time limit."""
    start = time.time()

    # Execute operation
    process_large_dataset(test_data)

    duration = time.time() - start
    self.assertLess(duration, 5.0, "Processing took too long")
```

## Debugging Tests

### Run with Verbose Output

```bash
pytest -vv --tb=long
```

### Run Single Test

```bash
pytest tests/unit/test_api.py::TestAPIEndpoints::test_get_user_metrics_success -v
```

### Print Debug Information

```python
def test_with_debug(self):
    result = function_under_test()
    print(f"Debug: result = {result}")  # Shows with -s flag
    self.assertEqual(result, expected)
```

Run with stdout:
```bash
pytest tests/unit/test_api.py -v -s
```

### Use Python Debugger

```python
def test_with_breakpoint(self):
    import pdb; pdb.set_trace()
    result = function_under_test()
    self.assertEqual(result, expected)
```

## Test Maintenance

### When to Update Tests

- When fixing bugs (add test that reproduces bug)
- When adding features (add tests for new functionality)
- When refactoring (ensure tests still pass)
- When tests become flaky (fix root cause)

### Identifying Flaky Tests

```bash
# Run tests multiple times to identify flakiness
pytest tests/ --count=10
```

### Removing Obsolete Tests

- Remove tests for deleted features
- Update tests when requirements change
- Archive tests for deprecated functionality

## Troubleshooting

### Common Issues

**Import Errors**
```bash
# Fix path issues
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Mock Not Working**
```python
# Ensure you're patching the right location
# Patch where it's imported, not where it's defined
@patch('api.src.main.bigquery.Client')  # Correct
@patch('google.cloud.bigquery.Client')   # May not work
```

**Test Hangs**
- Check for infinite loops
- Verify timeout settings
- Look for deadlocks in concurrent code

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [unittest documentation](https://docs.python.org/3/library/unittest.html)
- [Apache Beam testing](https://beam.apache.org/documentation/pipelines/test-your-pipeline/)
- [FastAPI testing](https://fastapi.tiangolo.com/tutorial/testing/)

## Questions and Support

For questions about testing:
- Review existing tests as examples
- Check this documentation
- Ask in team Slack channel: #data-platform
- Create an issue in the repository

---

**Last Updated**: January 2025
