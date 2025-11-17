# Contributing to Person Detector

Thank you for your interest in contributing to Person Detector! This document provides guidelines and instructions for contributing.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- A clear, descriptive title
- Steps to reproduce the bug
- Expected behavior
- Actual behavior
- Screenshots (if applicable)
- Your environment (OS, Python version, package versions)

### Suggesting Enhancements

Enhancement suggestions are welcome! Please create an issue with:
- A clear, descriptive title
- Detailed description of the proposed feature
- Use cases and benefits
- Possible implementation approach (if you have ideas)

### Pull Requests

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for your changes
5. Ensure all tests pass
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/Sehla-Yesser/Person_detector.git
cd Person_detector
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies (including development dependencies):
```bash
pip install -r requirements.txt
pip install -e .[dev]  # If using setup.py
```

4. Run tests:
```bash
python -m pytest tests/
```

## Coding Standards

- Follow PEP 8 style guide for Python code
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and small
- Write unit tests for new features
- Comment complex logic

## Testing

- Write unit tests for new features
- Ensure all tests pass before submitting PR
- Aim for high code coverage
- Test with different Python versions (3.8+)

## Documentation

- Update README.md if adding new features
- Add docstrings to new functions/classes
- Update CHANGELOG.md (if exists)
- Include examples for new features

## Code Review Process

1. Submit a PR with clear description
2. Address reviewer feedback
3. Ensure CI/CD checks pass
4. Maintainers will review and merge

## Areas for Contribution

- Adding specialized FER models
- Improving emotion detection accuracy
- Adding real-time webcam support
- Performance optimizations
- Adding more unit tests
- Improving documentation
- Adding example notebooks
- Supporting more input formats
- Creating a web interface

## Questions?

Feel free to open an issue for any questions or clarifications.

Thank you for contributing!
