.PHONY: help install test test-unit test-integration lint format clean docs

help:
	@echo "Available commands:"
	@echo "  make install          Install all dependencies"
	@echo "  make test             Run all tests with coverage"
	@echo "  make test-unit        Run unit tests only"
	@echo "  make test-integration Run integration tests only"
	@echo "  make lint             Run code linting"
	@echo "  make format           Format code with black and isort"
	@echo "  make clean            Clean temporary files and caches"
	@echo "  make docs             Generate documentation"

install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

test:
	pytest tests/ -v --cov --cov-report=html --cov-report=term

test-unit:
	pytest tests/unit/ -v -m unit

test-integration:
	pytest tests/integration/ -v -m integration

lint:
	flake8 pipelines/ api/src/ monitoring/data-quality/ --max-line-length=120
	pylint pipelines/ api/src/ --max-line-length=120 --disable=C0111,R0913

format:
	black pipelines/ api/src/ monitoring/data-quality/ tests/ --line-length=120
	isort pipelines/ api/src/ monitoring/data-quality/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete

docs:
	@echo "Generating project documentation..."
	@echo "See docs/ directory for architecture and runbook documentation"
