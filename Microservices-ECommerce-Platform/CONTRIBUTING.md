# Contributing Guidelines

## Getting Started

Thank you for considering contributing to the E-Commerce Microservices Platform. This document provides guidelines for contributing to the project.

## Development Setup

1. Fork the repository
2. Clone your fork
3. Create a feature branch
4. Make your changes
5. Test your changes
6. Submit a pull request

## Code Standards

### Python Code
- Follow PEP 8 style guide
- Use type hints
- Write docstrings for functions and classes
- Maximum line length: 100 characters

### Go Code
- Follow Go standard formatting (gofmt)
- Write tests for new functionality
- Use meaningful variable names
- Add comments for complex logic

### Commit Messages
Format: `type(scope): description`

Types:
- feat: New feature
- fix: Bug fix
- docs: Documentation changes
- refactor: Code refactoring
- test: Adding tests
- chore: Maintenance tasks

Example: `feat(order-service): add order cancellation endpoint`

## Testing

### Unit Tests
```bash
# Python services
cd services/order-service
pytest tests/unit -v

# Go services
cd services/product-service
go test ./... -v
```

### Integration Tests
```bash
pytest tests/integration -v
```

## Pull Request Process

1. Update documentation if needed
2. Add tests for new features
3. Ensure all tests pass
4. Update CHANGELOG.md
5. Request review from maintainers

## Code Review

All submissions require review. We use GitHub pull requests for this purpose.

## Reporting Issues

Use GitHub Issues to report bugs or request features. Include:
- Clear description
- Steps to reproduce
- Expected vs actual behavior
- Environment details

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
