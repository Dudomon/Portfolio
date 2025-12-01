# Contributing to Real-Time Analytics Pipeline

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/real-time-analytics-pipeline.git`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Test your changes
6. Commit: `git commit -m "Add your feature"`
7. Push: `git push origin feature/your-feature-name`
8. Create a Pull Request

## Development Setup

```bash
# Start all services
docker-compose up -d

# Run tests
pytest tests/

# Check code style
flake8 api/
eslint dashboard/src/
```

## Code Style

- Python: Follow PEP 8
- JavaScript: Use ESLint configuration
- Write clear commit messages
- Add comments for complex logic

## Pull Request Process

1. Update documentation if needed
2. Add tests for new features
3. Ensure all tests pass
4. Update CHANGELOG.md
5. Request review from maintainers

## Reporting Issues

- Use GitHub Issues
- Include steps to reproduce
- Provide logs and error messages
- Specify your environment

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
