# Changelog

All notable changes to the SCENIC+ and PINNACLE Integration Framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Planned features for future releases

### Changed
- Planned improvements for future releases

### Fixed
- Planned bug fixes for future releases

## [1.0.0] - 2025-07-21

### Added
- Initial release of the SCENIC+ and PINNACLE Integration Framework
- Core integration functionality combining regulatory networks with protein embeddings
- Support for multiple data formats (pickle, CSV, HDF5, NPZ)
- Comprehensive differential analysis capabilities
- Rich visualization suite with matplotlib and plotly backends
- Command-line interface for non-programmatic usage
- Automated identifier mapping between genes and proteins
- Quality control and validation systems
- Modular architecture for easy extension
- Complete documentation and examples
- Jupyter notebook tutorials
- Unit test suite with pytest
- CI/CD pipeline with GitHub Actions

### Core Components
- `ScenicPinnacleIntegrator`: Main integration class
- `ScenicProcessor`: SCENIC+ data handling
- `PinnacleProcessor`: PINNACLE data handling  
- `NetworkIntegrator`: Network combination logic
- `DifferentialAnalyzer`: Cross-condition analysis
- `NetworkVisualizer`: Plotting and visualization
- `IdentifierMapper`: Gene-protein ID mapping
- `QualityController`: Data validation

### Features
- Multi-omic network integration
- Statistical differential analysis
- Interactive and static visualizations
- Scalable processing for large datasets
- Memory-efficient data structures
- Parallel processing capabilities
- Comprehensive error handling
- Extensive configuration options

### Documentation
- Complete API reference
- User guide and tutorials
- Installation instructions
- Contributing guidelines
- Example workflows
- Jupyter notebook demonstrations

### Testing
- Unit tests for all core components
- Integration tests for workflows
- Test data fixtures and utilities
- Continuous integration setup
- Code coverage reporting

### Infrastructure
- Python package structure
- PyPI-ready setup configuration
- GitHub Actions CI/CD
- Documentation building
- Code quality tools (black, flake8, mypy)
- Security scanning
- Dependency management

## [0.1.0] - 2025-07-20

### Added
- Initial project structure and planning
- Research into SCENIC+ and PINNACLE integration approaches
- Architecture design and component specification
- Development environment setup

---

## Release Notes

### Version 1.0.0

This is the initial stable release of the SCENIC+ and PINNACLE Integration Framework. The framework provides a comprehensive solution for combining enhancer-driven gene regulatory networks from SCENIC+ with context-specific protein embeddings from PINNACLE.

**Key Highlights:**
- **Multi-scale Integration**: Seamlessly combines regulatory and protein network information
- **Quantitative Analysis**: Statistical methods for comparing networks across conditions
- **Rich Visualizations**: Publication-ready plots and interactive visualizations
- **User-Friendly**: Both programmatic API and command-line interface
- **Extensible**: Modular design allows for easy customization and extension
- **Well-Tested**: Comprehensive test suite ensures reliability
- **Documented**: Complete documentation with examples and tutorials

**Supported Data Formats:**
- SCENIC+: Pickle, CSV, AnnData (H5AD)
- PINNACLE: Pickle, CSV, NumPy (NPZ)

**Analysis Capabilities:**
- Network integration with multiple edge types
- Differential analysis between conditions
- Module detection and community analysis
- Quality control and validation
- Statistical significance testing

**Visualization Options:**
- Network plots (static and interactive)
- Heatmaps and correlation matrices
- Volcano plots for differential analysis
- Embedding projections (PCA, t-SNE, UMAP)
- Summary statistics plots

**Installation:**
```bash
pip install scenic-pinnacle-integration
```

**Quick Start:**
```python
from scenic_pinnacle import ScenicPinnacleIntegrator

integrator = ScenicPinnacleIntegrator()
integrator.load_scenic_data('scenic_networks.pkl')
integrator.load_pinnacle_data('pinnacle_embeddings.pkl')
integrated_networks = integrator.integrate_networks()
```

For detailed usage instructions, please refer to the documentation and examples.

### Breaking Changes
- None (initial release)

### Migration Guide
- None (initial release)

### Known Issues
- Large datasets (>10,000 genes) may require memory optimization settings
- Online identifier mapping depends on external database availability
- Some visualization features require optional dependencies

### Acknowledgments
- SCENIC+ development team for the regulatory network inference method
- PINNACLE development team for the protein embedding approach
- Single-cell genomics community for feedback and testing
- Contributors and early adopters

---

For more information about releases, see the [GitHub Releases](https://github.com/manus-ai/scenic-pinnacle-integration/releases) page.

