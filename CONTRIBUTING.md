# Contributing to SCENIC+ and PINNACLE Integration Framework

We welcome contributions to the SCENIC+ and PINNACLE Integration Framework! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

This project adheres to a code of conduct that we expect all contributors to follow. Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) to help maintain a welcoming and inclusive community.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/scenic-pinnacle-integration.git
   cd scenic-pinnacle-integration
   ```
3. **Set up the development environment** (see Development Setup below)
4. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Installation

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install the package in development mode**:
   ```bash
   pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks** (optional but recommended):
   ```bash
   pre-commit install
   ```

### Verify Installation

Run the test suite to ensure everything is working:

```bash
pytest tests/
```

## Contributing Guidelines

### Types of Contributions

We welcome several types of contributions:

- **Bug fixes**: Fix issues in the existing codebase
- **Feature additions**: Add new functionality
- **Documentation improvements**: Enhance or fix documentation
- **Performance optimizations**: Improve code efficiency
- **Test coverage**: Add or improve tests
- **Examples**: Create new examples or tutorials

### Before You Start

1. **Check existing issues** to see if your idea is already being worked on
2. **Open an issue** to discuss major changes before implementing them
3. **Search existing pull requests** to avoid duplicate work

## Pull Request Process

### 1. Prepare Your Changes

- Ensure your code follows our [coding standards](#coding-standards)
- Add or update tests as needed
- Update documentation if necessary
- Run the full test suite locally

### 2. Commit Your Changes

Use clear, descriptive commit messages:

```bash
git add .
git commit -m "Add feature: brief description of what you added"
```

Follow conventional commit format when possible:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for test additions/changes
- `refactor:` for code refactoring
- `style:` for formatting changes

### 3. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub with:
- **Clear title** describing the change
- **Detailed description** explaining what and why
- **Reference to related issues** (if applicable)
- **Screenshots or examples** (if relevant)

### 4. Review Process

- Maintainers will review your pull request
- Address any feedback or requested changes
- Once approved, your changes will be merged

## Issue Reporting

### Before Reporting

1. **Search existing issues** to avoid duplicates
2. **Try the latest version** to see if the issue persists
3. **Gather relevant information** (Python version, OS, error messages)

### Creating Good Issues

Use our issue templates and include:

- **Clear, descriptive title**
- **Steps to reproduce** the problem
- **Expected vs. actual behavior**
- **Environment details** (Python version, OS, package versions)
- **Code examples** or error messages
- **Possible solutions** (if you have ideas)

### Issue Labels

We use labels to categorize issues:

- `bug`: Something isn't working
- `enhancement`: New feature or improvement
- `documentation`: Documentation needs improvement
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention needed
- `question`: Further information requested

## Coding Standards

### Python Style

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

- **Line length**: 88 characters (Black default)
- **Import organization**: Use `isort`
- **Code formatting**: Use `black`
- **Type hints**: Use type hints for function signatures
- **Docstrings**: Use Google-style docstrings

### Code Quality Tools

We use several tools to maintain code quality:

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/

# Run all checks
pre-commit run --all-files
```

### Naming Conventions

- **Classes**: PascalCase (`NetworkIntegrator`)
- **Functions/methods**: snake_case (`integrate_networks`)
- **Variables**: snake_case (`scenic_networks`)
- **Constants**: UPPER_SNAKE_CASE (`DEFAULT_CONFIG`)
- **Private methods**: Leading underscore (`_process_data`)

### Documentation Style

- **Docstrings**: Google style for all public functions and classes
- **Comments**: Explain why, not what
- **Type hints**: Use for all function parameters and return values

Example:

```python
def integrate_networks(
    self,
    scenic_networks: Dict[str, nx.DiGraph],
    pinnacle_embeddings: Dict[str, Dict],
    gene_protein_mapping: Optional[Dict[str, str]] = None
) -> Dict[str, nx.Graph]:
    """
    Integrate SCENIC+ networks with PINNACLE embeddings.
    
    Args:
        scenic_networks: Dictionary of SCENIC+ regulatory networks by condition
        pinnacle_embeddings: Dictionary of PINNACLE protein embeddings by condition
        gene_protein_mapping: Optional mapping from gene IDs to protein IDs
        
    Returns:
        Dictionary of integrated networks by condition
        
    Raises:
        ValueError: If input data formats are incompatible
        
    Examples:
        >>> integrator = ScenicPinnacleIntegrator()
        >>> integrated = integrator.integrate_networks(scenic_data, pinnacle_data)
    """
```

## Testing

### Test Structure

Tests are organized in the `tests/` directory:

```
tests/
├── test_core.py              # Test main integrator
├── test_data_processing.py   # Test data loading/processing
├── test_network_analysis.py  # Test integration and analysis
├── test_utils.py            # Test utilities
├── test_visualization.py    # Test plotting functions
└── fixtures/                # Test data and fixtures
```

### Writing Tests

- Use `pytest` for all tests
- Follow the Arrange-Act-Assert pattern
- Use descriptive test names
- Include both positive and negative test cases
- Mock external dependencies when appropriate

Example:

```python
def test_integrate_networks_with_valid_data():
    """Test network integration with valid input data."""
    # Arrange
    integrator = ScenicPinnacleIntegrator()
    scenic_networks = create_test_scenic_networks()
    pinnacle_embeddings = create_test_pinnacle_embeddings()
    
    # Act
    result = integrator.integrate_networks(scenic_networks, pinnacle_embeddings)
    
    # Assert
    assert len(result) == 2
    assert all(isinstance(network, nx.Graph) for network in result.values())
    assert result['condition1'].number_of_nodes() > 0
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_core.py

# Run with coverage
pytest --cov=scenic_pinnacle

# Run with verbose output
pytest -v

# Run specific test
pytest tests/test_core.py::test_integrate_networks_with_valid_data
```

### Test Coverage

We aim for high test coverage:

- **Minimum**: 80% overall coverage
- **Target**: 90%+ for core functionality
- **Critical paths**: 100% coverage for main workflows

## Documentation

### Types of Documentation

1. **API Documentation**: Docstrings in code
2. **User Guide**: High-level usage documentation
3. **Examples**: Jupyter notebooks and Python scripts
4. **README**: Project overview and quick start

### Building Documentation

We use Sphinx for documentation:

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation
cd docs/
make html

# View documentation
open _build/html/index.html
```

### Documentation Guidelines

- **Keep it current**: Update docs with code changes
- **Be comprehensive**: Cover all public APIs
- **Include examples**: Show real usage patterns
- **Use clear language**: Write for your audience
- **Add diagrams**: Visual aids help understanding

## Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

1. Update version numbers
2. Update CHANGELOG.md
3. Run full test suite
4. Build and test documentation
5. Create release notes
6. Tag release in Git
7. Deploy to PyPI (maintainers only)

## Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Email**: contact@manus.ai for sensitive issues

### Resources

- **Documentation**: [Read the Docs](https://scenic-pinnacle-integration.readthedocs.io/)
- **Examples**: Check the `examples/` directory
- **Tests**: Look at test files for usage patterns

## Recognition

Contributors will be recognized in:

- **README.md**: Contributors section
- **CHANGELOG.md**: Release notes
- **Documentation**: Acknowledgments page
- **GitHub**: Contributor graphs and statistics

## Questions?

If you have questions about contributing, please:

1. Check this document first
2. Search existing issues and discussions
3. Open a new discussion or issue
4. Contact maintainers directly if needed

Thank you for contributing to the SCENIC+ and PINNACLE Integration Framework!

