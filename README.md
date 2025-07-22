# BRIDGE: Biological Regulatory Integration for Dynamic Gene Expression

A comprehensive Python framework for integrating SCENIC+ enhancer-driven gene regulatory networks with PINNACLE context-specific protein embeddings to enable multi-omic network analysis quantifiable across conditions.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/bridge-omics/badge/?version=latest)](https://bridge-omics.readthedocs.io/en/latest/?badge=latest)

## Overview

This framework bridges two powerful single-cell analysis methods:

- **SCENIC+**: Reconstructs enhancer-driven gene regulatory networks (eGRNs) by integrating single-cell chromatin accessibility and gene expression data
- **PINNACLE**: Generates context-specific protein embeddings by integrating transcriptomics, protein-protein interaction networks, and cellular context

By combining these approaches, researchers can:
- Map mechanistic regulatory pathways from enhancers to proteins
- Quantify network changes across biological conditions
- Identify condition-specific regulatory and protein interaction patterns
- Generate comprehensive multi-scale biological insights

## 🚀 Enhanced Capabilities

### **scVI Integration**
- **Denoised Expression**: Improved input quality for SCENIC+ regulatory network inference
- **Robust Latent Space**: Enhanced protein embedding context for PINNACLE
- **Batch Correction**: Handle technical variation across experimental conditions
- **Multi-modal Analysis**: Joint RNA+ATAC modeling with totalVI/MultiVI
- **Better Peak-Gene Links**: Enhanced enhancer-gene associations

### **RNA Velocity Integration**
- **Dynamic Network Analysis**: Track regulatory network evolution with RNA velocity
- **Drug Response Trajectories**: Map how networks change in response to drug treatments
- **Perturbed Pathway Identification**: Detect regulatory modules affected by perturbations
- **Pharmacological Profiling**: Comprehensive drug target discovery and prioritization
- **Temporal Analysis**: Monitor network dynamics across time points and conditions
- **Dose-Response Modeling**: Quantify network changes across drug concentration gradients

### **Optimal Transport (OT)**
- **Cross-Modal Alignment**: Principled alignment between regulatory networks and protein embeddings
- **Condition Comparison**: Quantify network changes using Wasserstein distances
- **Batch Integration**: Remove technical effects while preserving biological signal
- **Trajectory Analysis**: Track regulatory network evolution across conditions/time
- **Structure-Aware Matching**: Gromov-Wasserstein for topology-preserving alignment

### 🔬 **Multi-Omic Integration**
- Seamless integration of regulatory networks and protein embeddings
- Support for multiple data formats (pickle, CSV, HDF5, NPZ)
- Automated identifier mapping between genes and proteins
- Quality control and validation at each integration step

### 📊 **Differential Analysis**
- Quantitative comparison of networks across conditions
- Statistical testing for regulatory and protein changes
- Module-based analysis for functional interpretation
- Comprehensive summary statistics and effect sizes

### 🎨 **Rich Visualizations**
- Interactive and static network visualizations
- Heatmaps, volcano plots, and embedding projections
- Publication-ready figures with customizable styling
- Support for both matplotlib and plotly backends

### ⚡ **Scalable Architecture**
- Modular design for easy extension and customization
- Efficient processing of large-scale single-cell datasets
- Parallel processing capabilities for multiple conditions
- Memory-optimized data structures

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install from Source

```bash
# Clone the repository
git clone https://github.com/manus-ai/bridge.git
cd bridge

# Install in development mode
pip install -e .

# Or install with all optional dependencies
pip install -e ".[all]"
```

### Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# For enhanced visualizations
pip install plotly umap-learn

# For development
pip install pytest black flake8 mypy
```

## Quick Start

### Basic Usage

```python
from bridge import BridgeIntegrator

# Initialize integrator with configuration
config = {
    'scenic': {'min_regulon_size': 10, 'importance_threshold': 0.1},
    'pinnacle': {'normalize_embeddings': True},
    'integration': {'similarity_threshold': 0.5}
}

integrator = BridgeIntegrator(config=config)

# Load data
integrator.load_scenic_data('scenic_networks.pkl')
integrator.load_pinnacle_data('pinnacle_embeddings.pkl')

# Integrate networks
integrated_networks = integrator.integrate_networks()

# Perform differential analysis
diff_results = integrator.differential_analysis('condition1', 'condition2')

# Generate visualizations
integrator.visualize_networks('condition1', 'output_dir/')

# Export results
integrator.export_results('results/### Enhanced Usage with scVI and Optimal Transport

```python
from bridge import ScviEnhancedBridge, OTEnhancedBridge
import scanpy as sc
import anndata

# Load your single-cell data
rna_adata = sc.read_h5ad('rna_data.h5ad')
atac_adata = sc.read_h5ad('atac_data.h5ad')

# Option 1: scVI-Enhanced BRIDGE
scvi_config = {
    'scvi': {
        'n_latent': 30,
        'max_epochs': 400,
        'use_gpu': True
    },
    'bridge': {
        'scenic': {'min_regulon_size': 10},
        'pinnacle': {'normalize_embeddings': True}
    }
}

scvi_bridge = ScviEnhancedBridge(config=scvi_config)
results = scvi_bridge.preprocess_and_integrate(
    rna_adata=rna_adata,
    atac_adata=atac_adata,
    condition_key='condition',
    batch_key='batch',
    use_multimodal=True
)

# Option 2: Optimal Transport Enhanced BRIDGE
ot_config = {
    'optimal_transport': {
        'reg': 0.1,
        'method': 'fused_gromov_wasserstein',
        'normalize': True
    }
}

ot_bridge = OTEnhancedBridge(config=ot_config)
ot_results = ot_bridge.integrate_with_ot_alignment(
    scenic_networks=scenic_networks,
    pinnacle_embeddings=pinnacle_embeddings,
    gene_protein_mapping=gene_protein_mapping
)

# Trajectory analysis with OT
trajectory_results = ot_bridge.trajectory_analysis_with_ot(
    scenic_networks=scenic_networks,
    pinnacle_embeddings=pinnacle_embeddings,
    condition_order=['healthy', 'disease', 'treatment'],
    time_points=[0, 24, 48]
)
```a Requirements

### SCENIC+ Data

The framework accepts SCENIC+ regulatory networks in multiple formats:

**Pickle Format** (Recommended):
```python
# Dictionary structure
{
    'condition1': networkx.DiGraph,
    'condition2': networkx.DiGraph,
    # ...
}
```

**CSV Format**:
```csv
TF,target,importance,enhancer
TF1,Gene1,0.85,Enhancer_123
TF1,Gene2,0.72,Enhancer_456
# ...
```

**AnnData Format**:
- Regulatory networks stored in `adata.uns['regulons']`
- Cell type information in `adata.obs['cell_type']`

### PINNACLE Data

PINNACLE protein embeddings in supported formats:

**Pickle Format** (Recommended):
```python
# Dictionary structure
{
    'condition1': {
        'embeddings': numpy.ndarray,  # Shape: (n_proteins, embedding_dim)
        'protein_ids': list,          # List of protein identifiers
        'embedding_dim': int          # Embedding dimensionality
    },
    # ...
}
```

**NPZ Format**:
```python
# NumPy archive with keys:
# - 'embeddings': protein embedding matrix
# - 'protein_ids': protein identifier array
# - 'contexts': condition labels (optional)
```

## Examples

### 1. Basic Integration

See [`examples/basic_integration_example.py`](examples/basic_integration_example.py) for a complete standalone example.

### 2. Interactive Analysis

Explore [`examples/interactive_analysis.ipynb`](examples/interactive_analysis.ipynb) for a comprehensive Jupyter notebook tutorial.

### 3. Custom Analysis

```python
from scenic_pinnacle import (
    ScenicProcessor, PinnacleProcessor, 
    NetworkIntegrator, DifferentialAnalyzer
)

# Custom processing pipeline
scenic_processor = ScenicProcessor(config={'min_regulon_size': 15})
pinnacle_processor = PinnacleProcessor(config={'normalize_embeddings': True})

# Load and process data separately
scenic_networks = scenic_processor.load_data('scenic_data.pkl')
pinnacle_embeddings = pinnacle_processor.load_data('pinnacle_data.pkl')

# Custom integration
integrator = NetworkIntegrator(config={'similarity_threshold': 0.6})
integrated_network = integrator.integrate(
    scenic_networks['condition1'],
    pinnacle_embeddings['condition1'],
    gene_protein_mapping
)

# Custom differential analysis
analyzer = DifferentialAnalyzer()
diff_results = analyzer.analyze(
    integrated_networks['condition1'],
    integrated_networks['condition2']
)
```

## Configuration

The framework uses hierarchical configuration for fine-tuned control:

```python
config = {
    'scenic': {
        'min_regulon_size': 10,        # Minimum targets per TF
        'importance_threshold': 0.1,    # Minimum edge importance
        'min_target_genes': 5          # Minimum targets for analysis
    },
    'pinnacle': {
        'embedding_dim': 256,          # Expected embedding dimension
        'normalize_embeddings': True,   # L2 normalize embeddings
        'similarity_threshold': 0.5    # Protein similarity cutoff
    },
    'integration': {
        'similarity_threshold': 0.5,   # Edge creation threshold
        'min_overlap': 3,              # Minimum gene-protein overlap
        'use_cross_layer_edges': True  # Connect regulatory and protein layers
    },
    'differential': {
        'pvalue_threshold': 0.05,      # Statistical significance
        'fold_change_threshold': 1.5,  # Effect size threshold
        'min_observations': 10         # Minimum data points
    },
    'visualization': {
        'layout_algorithm': 'spring',  # Network layout method
        'node_size_range': (20, 200),  # Node size scaling
        'figure_size': (12, 8),        # Default figure dimensions
        'dpi': 300                     # Figure resolution
    }
}
```

## Architecture

The framework consists of several modular components:

```
scenic_pinnacle/
├── core.py              # Main ScenicPinnacleIntegrator class
├── data_processing.py   # Data loading and preprocessing
├── network_analysis.py  # Integration and differential analysis
├── utils.py            # Utilities and identifier mapping
└── visualization.py    # Plotting and visualization tools
```

### Core Components

1. **ScenicProcessor**: Handles SCENIC+ regulatory network data
2. **PinnacleProcessor**: Manages PINNACLE protein embeddings
3. **NetworkIntegrator**: Combines regulatory and protein networks
4. **DifferentialAnalyzer**: Performs cross-condition comparisons
5. **NetworkVisualizer**: Creates publication-ready visualizations
6. **IdentifierMapper**: Maps between gene and protein identifiers

## API Reference

### Main Classes

#### `ScenicPinnacleIntegrator`

The primary interface for the integration framework.

**Methods:**
- `load_scenic_data(path, format)`: Load SCENIC+ networks
- `load_pinnacle_data(path, format)`: Load PINNACLE embeddings
- `integrate_networks()`: Perform network integration
- `differential_analysis(cond1, cond2, type)`: Compare conditions
- `visualize_networks(condition, output_dir)`: Generate plots
- `export_results(output_dir, formats)`: Save results
- `run_complete_workflow(...)`: Execute full pipeline

#### `NetworkIntegrator`

Handles the core integration logic.

**Methods:**
- `integrate(scenic_network, protein_embeddings, mapping)`: Integrate single condition
- `identify_network_modules(network, method)`: Find network communities
- `_add_regulatory_edges()`: Add SCENIC+ edges
- `_add_protein_edges()`: Add PINNACLE edges
- `_add_cross_layer_edges()`: Connect layers

#### `DifferentialAnalyzer`

Performs statistical comparisons between conditions.

**Methods:**
- `analyze(network1, network2, type)`: Compare two networks
- `identify_differential_modules()`: Find changed modules
- `_analyze_regulatory_differences()`: Compare regulatory edges
- `_analyze_protein_differences()`: Compare protein similarities

## Advanced Usage

### Custom Identifier Mapping

```python
from scenic_pinnacle.utils import IdentifierMapper

# Initialize mapper with custom databases
mapper = IdentifierMapper(config={
    'mapping_dir': 'custom_mappings/',
    'use_online_mapping': True,
    'use_mygene': True,
    'use_ensembl': True
})

# Map gene symbols to UniProt IDs
gene_protein_mapping = mapper.map_genes_to_proteins(['TP53', 'BRCA1', 'MYC'])
```

### Quality Control

```python
from scenic_pinnacle.utils import QualityController

qc = QualityController()

# Validate input data
validated_scenic = qc.validate_scenic_networks(scenic_networks)
validated_pinnacle = qc.validate_pinnacle_embeddings(pinnacle_embeddings)

# Assess integration quality
quality_metrics = qc.assess_integration_quality(integrated_networks)
```

### Custom Visualizations

```python
from scenic_pinnacle.visualization import NetworkVisualizer

visualizer = NetworkVisualizer(config={
    'style': {
        'figure_size': (15, 10),
        'node_size_range': (30, 300),
        'color_palette': 'Set2'
    }
})

# Create custom network plot
visualizer.plot_network(
    integrated_networks['condition1'],
    output_path='custom_network.png',
    layout='kamada_kawai',
    color_by='node_type',
    size_by='degree',
    interactive=True
)

# Plot protein embeddings
visualizer.plot_embeddings(
    integrated_networks['condition1'],
    output_path='embeddings_umap.png',
    method='umap',
    color_by='node_type'
)
```

## Performance Optimization

### Memory Management

For large datasets, consider these optimization strategies:

```python
# Process conditions separately to reduce memory usage
config['processing'] = {
    'batch_size': 1000,
    'use_sparse_matrices': True,
    'memory_efficient': True
}

# Use data generators for very large datasets
def data_generator(file_paths):
    for path in file_paths:
        yield load_data(path)

# Process in chunks
for chunk in data_generator(file_paths):
    results = integrator.process_chunk(chunk)
```

### Parallel Processing

```python
from multiprocessing import Pool
from functools import partial

# Parallel network integration
def integrate_condition(condition, scenic_data, pinnacle_data, config):
    integrator = NetworkIntegrator(config)
    return integrator.integrate(scenic_data[condition], pinnacle_data[condition])

# Use multiprocessing for multiple conditions
with Pool(processes=4) as pool:
    integrate_func = partial(integrate_condition, 
                           scenic_data=scenic_networks,
                           pinnacle_data=pinnacle_embeddings,
                           config=config)
    
    results = pool.map(integrate_func, conditions)
```

## Troubleshooting

### Common Issues

1. **Memory Errors**:
   - Reduce batch size in configuration
   - Use sparse matrix representations
   - Process conditions separately

2. **Identifier Mapping Failures**:
   - Check gene/protein identifier formats
   - Verify online database connectivity
   - Use custom mapping files

3. **Empty Integration Results**:
   - Lower similarity thresholds
   - Check data format compatibility
   - Verify identifier overlap between datasets

4. **Visualization Errors**:
   - Install optional visualization dependencies
   - Check output directory permissions
   - Reduce network size for complex layouts

### Debug Mode

Enable detailed logging for troubleshooting:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('scenic_pinnacle')

# Run with detailed logging
integrator = ScenicPinnacleIntegrator(config=config)
# ... analysis code ...
```

## Contributing

We welcome contributions to improve the framework! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone and install in development mode
git clone https://github.com/manus-ai/scenic-pinnacle-integration.git
cd scenic-pinnacle-integration
pip install -e ".[dev]"

# Run tests
pytest tests/

# Code formatting
black src/
flake8 src/

# Type checking
mypy src/
```

### Submitting Issues

Please use our [Issue Template](.github/ISSUE_TEMPLATE.md) when reporting bugs or requesting features.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{scenic_pinnacle_integration,
  title={SCENIC+ and PINNACLE Integration Framework},
  author={Manus AI},
  year={2025},
  url={https://github.com/manus-ai/scenic-pinnacle-integration},
  version={1.0.0}
}
```

Also cite the original methods:

- **SCENIC+**: Bravo González-Blas, C. et al. SCENIC+: single-cell multiomic inference of enhancers and gene regulatory networks. Nat Methods (2023).
- **PINNACLE**: Zitnik, M. et al. Machine learning for integrating data in biology and medicine: Principles, practice, and opportunities. Inf Fusion 50, 71-91 (2019).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [Read the Docs](https://scenic-pinnacle-integration.readthedocs.io/)
- **Issues**: [GitHub Issues](https://github.com/manus-ai/scenic-pinnacle-integration/issues)
- **Discussions**: [GitHub Discussions](https://github.com/manus-ai/scenic-pinnacle-integration/discussions)
- **Email**: contact@manus.ai

## Acknowledgments

- The SCENIC+ team for developing the enhancer-driven regulatory network inference method
- The PINNACLE team for creating context-specific protein embedding approaches
- The single-cell genomics community for valuable feedback and testing
- Contributors and users who help improve this framework

---

**Developed by [Manus AI](https://manus.ai) - Advancing AI for Scientific Discovery**

