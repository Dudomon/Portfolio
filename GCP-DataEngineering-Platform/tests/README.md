# Test Suite Documentation

## Overview

This directory contains comprehensive tests for the GCP Data Engineering Platform. The test suite ensures code quality, reliability, and maintainability through unit tests, integration tests, and automated CI/CD validation.

## Test Statistics

- **Total Test Files**: 5
- **Test Coverage**: 75%+ target
- **Test Categories**: Unit, Integration
- **CI/CD Integration**: GitHub Actions

## Directory Structure

```
tests/
├── README.md                      # This file
├── __init__.py
├── unit/                          # Unit tests (fast, isolated)
│   ├── test_beam_pipeline.py     # Apache Beam components (10 test classes, 30+ tests)
│   ├── test_spark_processor.py   # PySpark transformations (15+ tests)
│   ├── test_api.py               # FastAPI endpoints (20+ tests)
│   └── test_data_quality.py      # Great Expectations config (15+ tests)
├── integration/                   # Integration tests
│   └── test_end_to_end.py        # End-to-end workflows (10+ tests)
└── fixtures/                      # Test data and fixtures
```

## Quick Start

### Run All Tests

```bash
# From project root
make test
```

### Run Specific Test Suite

```bash
# Unit tests only (fastest)
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# Specific test file
pytest tests/unit/test_beam_pipeline.py -v

# Specific test function
pytest tests/unit/test_api.py::TestAPIEndpoints::test_get_user_metrics_success -v
```

## Test Coverage by Component

### 1. Apache Beam Pipeline Tests (`test_beam_pipeline.py`)

**Coverage**: Message parsing, validation, enrichment, data quality

**Test Classes**:
- `TestParsePubSubMessage`: JSON parsing, field validation, error handling
- `TestEnrichEvent`: Metadata enrichment, device detection
- `TestValidateDataQuality`: Business rule validation, quality checks
- `TestFormatForBigQuery`: Output formatting, schema compliance

**Key Tests**:
- Valid message parsing and validation
- Invalid JSON handling
- Missing required field detection
- Timestamp format validation
- Device type detection (mobile, tablet, desktop)
- Data quality rule enforcement
- BigQuery schema formatting

**Run**:
```bash
pytest tests/unit/test_beam_pipeline.py -v
```

### 2. PySpark Processor Tests (`test_spark_processor.py`)

**Coverage**: Data cleaning, aggregations, anomaly detection

**Test Classes**:
- `TestSparkBatchProcessor`: Transaction processing, user metrics
- `TestDataTransformations`: Common transformation patterns

**Key Tests**:
- Duplicate removal
- Invalid data filtering (negatives, nulls)
- User-level aggregations
- Merchant metrics calculation
- Refund rate computation
- High-value anomaly detection
- Rapid transaction detection
- Currency standardization
- Date partitioning

**Run**:
```bash
pytest tests/unit/test_spark_processor.py -v
```

### 3. API Endpoint Tests (`test_api.py`)

**Coverage**: REST API functionality, error handling, validation

**Test Classes**:
- `TestAPIEndpoints`: Endpoint functionality and responses
- `TestAPIErrorHandling`: Error scenarios and status codes

**Key Tests**:
- Root endpoint information
- Health check (success/failure)
- User metrics retrieval (found/not found)
- Top users with pagination and sorting
- Merchant metrics
- Data quality status
- Pipeline metrics with filters
- Daily summary with date ranges
- Query parameter validation
- Internal error handling

**Run**:
```bash
pytest tests/unit/test_api.py -v
```

### 4. Data Quality Tests (`test_data_quality.py`)

**Coverage**: Great Expectations configuration validation

**Test Classes**:
- `TestDataQualityExpectations`: Expectation suite validation
- `TestCheckpointConfiguration`: Checkpoint structure
- `TestExpectationCoverage`: Validation completeness

**Key Tests**:
- Expectation suite structure
- Required column validation
- Value range checks
- Currency/event type validation
- Uniqueness constraints
- Length validations
- Checkpoint configuration
- Action list completeness
- Field coverage verification

**Run**:
```bash
pytest tests/unit/test_data_quality.py -v
```

### 5. Integration Tests (`test_end_to_end.py`)

**Coverage**: Complete workflow validation

**Test Classes**:
- `TestEndToEndPipelineFlow`: Full pipeline flows
- `TestDataQualityIntegration`: Quality check integration
- `TestMonitoringIntegration`: Metrics collection
- `TestAPIIntegration`: API to backend flow

**Key Tests**:
- Valid message complete flow
- Invalid message error routing
- Transaction processing pipeline
- Data quality validation integration
- Quality failure routing
- Metrics collection
- Alert threshold detection
- API query execution flow
- Error response formatting

**Run**:
```bash
pytest tests/integration/ -v
```

## Test Execution Patterns

### Development Workflow

```bash
# 1. Make code changes
# 2. Run relevant unit tests
pytest tests/unit/test_beam_pipeline.py -v

# 3. Run all unit tests
make test-unit

# 4. Run full test suite with coverage
make test

# 5. Check coverage report
open htmlcov/index.html
```

### Pre-Commit Checks

```bash
# Format code
make format

# Lint check
make lint

# Run tests
make test
```

### Continuous Integration

Tests run automatically on:
- Every commit to main/develop branches
- Every pull request
- Configured in `.github/workflows/ci.yml`

CI includes:
- Linting (flake8)
- Unit tests with coverage
- Security checks (bandit, safety)
- Terraform validation

## Writing New Tests

### Unit Test Template

```python
import unittest
from unittest.mock import Mock

class TestNewFeature(unittest.TestCase):
    """Test description."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_data = {...}

    def test_normal_case(self):
        """Test normal operation."""
        result = function_under_test(self.test_data)
        self.assertEqual(result, expected_value)

    def test_edge_case(self):
        """Test edge case handling."""
        result = function_under_test(edge_case_input)
        self.assertIsNotNone(result)

    def test_error_handling(self):
        """Test error handling."""
        with self.assertRaises(ExpectedException):
            function_under_test(invalid_input)
```

### Integration Test Template

```python
class TestNewIntegration(unittest.TestCase):
    """Test integration between components."""

    def test_component_integration(self):
        """Test that components work together."""
        # Arrange: Set up multiple components
        # Act: Execute workflow
        # Assert: Verify end-to-end behavior
        pass
```

## Test Configuration

### pytest.ini

Configuration file for pytest:
- Test discovery patterns
- Coverage settings (75% minimum)
- Test markers (unit, integration, slow)
- Warning filters

### Makefile

Common test commands:
- `make test`: Run all tests with coverage
- `make test-unit`: Unit tests only
- `make test-integration`: Integration tests only
- `make lint`: Code quality checks
- `make format`: Code formatting
- `make clean`: Clean test artifacts

## Mocking Strategy

### BigQuery Mocking

```python
from unittest.mock import Mock
from google.cloud import bigquery

mock_client = Mock(spec=bigquery.Client)
mock_result = [{'field': 'value'}]
mock_query_job = Mock()
mock_query_job.result.return_value = mock_result
mock_client.query.return_value = mock_query_job
```

### Beam Pipeline Testing

```python
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that, equal_to

with TestPipeline() as p:
    output = (
        p
        | beam.Create([input_data])
        | beam.ParDo(Transform())
    )
    assert_that(output, equal_to(expected_output))
```

### FastAPI Testing

```python
from fastapi.testclient import TestClient

client = TestClient(app)
response = client.get("/endpoint")
assert response.status_code == 200
```

## Common Issues and Solutions

### Import Errors

```bash
# Solution: Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Mock Not Working

```python
# Patch where it's used, not where it's defined
@patch('api.src.main.bigquery.Client')  # Correct
```

### Slow Tests

```bash
# Run unit tests only (faster)
pytest tests/unit/ -v

# Skip slow tests
pytest -m "not slow"
```

## Coverage Reports

### Viewing Coverage

```bash
# Terminal report
pytest --cov --cov-report=term-missing

# HTML report
pytest --cov --cov-report=html
open htmlcov/index.html
```

### Coverage Goals

- **Overall**: 75%+ (enforced in CI)
- **Critical paths**: 90%+
- **New features**: 80%+

## Best Practices

1. **Test Independence**: Tests should run in any order
2. **Clear Names**: Test name describes what is being tested
3. **Single Responsibility**: One test validates one behavior
4. **Mock External Dependencies**: Don't call real GCP services
5. **Fast Execution**: Unit tests should complete in milliseconds
6. **Comprehensive**: Test success paths and error paths
7. **Maintainable**: Easy to understand and update

## Resources

- [Full Testing Guide](../docs/testing.md)
- [pytest Documentation](https://docs.pytest.org/)
- [Apache Beam Testing](https://beam.apache.org/documentation/pipelines/test-your-pipeline/)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)

## Contributing

When adding new code:
1. Write tests first (TDD) or alongside implementation
2. Ensure tests pass: `make test`
3. Check coverage: `pytest --cov`
4. Run linting: `make lint`
5. Submit PR with tests included

## Questions

For questions about testing:
- Review existing tests as examples
- Check [Testing Guide](../docs/testing.md)
- Ask in team Slack: #data-platform

---

**Last Updated**: January 2025
