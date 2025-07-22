# BRIDGE: Biological Regulatory Integration for Dynamic Gene Expression

**BRIDGE** is a cutting-edge computational framework that integrates SCENIC+ enhancer-driven gene regulatory networks with PINNACLE context-specific protein embeddings, enhanced with **RNA velocity analysis** to detect and quantify network perturbations in drug response and experimental conditions through splicing dynamics.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/bridge-omics/badge/?version=latest)](https://bridge-omics.readthedocs.io/en/latest/?badge=latest)

## **Core Innovation: Velocity-Guided Perturbation Detection**

BRIDGE leverages **RNA velocity** (velocyto/scVelo) to detect network perturbations by analyzing splicing dynamics and transcriptional kinetics. This enables unprecedented sensitivity in identifying how drugs, treatments, or experimental perturbations alter regulatory networks at the molecular level.

### **Key Breakthrough: Splicing-Based Perturbation Mapping**
- **Detect Early Perturbations**: Identify network changes before steady-state expression changes
- **Splicing Event Analysis**: Use unspliced/spliced RNA ratios to detect transcriptional perturbations  
- **Real-Time Network Dynamics**: Track how regulatory networks respond to perturbations in real-time
- **Perturbation Sensitivity**: Detect subtle network changes invisible to traditional expression analysis

## Overview

This framework combines three powerful approaches for comprehensive perturbation analysis:

- **SCENIC+**: Reconstructs enhancer-driven gene regulatory networks from single-cell multi-omics data
- **PINNACLE**: Generates context-specific protein embeddings integrating multiple data modalities
- **RNA Velocity**: Analyzes splicing dynamics to detect transcriptional perturbations and network changes

**Unique Capability**: By integrating RNA velocity with network analysis, BRIDGE can pinpoint which regulatory networks are perturbed by drugs or experimental conditions through analysis of splicing short events, providing early detection of perturbations before they manifest in steady-state expression changes.

## **Perturbation Detection Capabilities**

### **RNA Velocity-Driven Perturbation Analysis**
- **Splicing Dynamics Monitoring**: Track unspliced/spliced RNA ratios to detect transcriptional perturbations
- **Early Perturbation Detection**: Identify network changes hours before steady-state expression changes
- **Drug Response Mapping**: Pinpoint which regulatory networks are affected by drug treatments
- **Condition Perturbation Profiling**: Detect network alterations across experimental conditions
- **Temporal Perturbation Tracking**: Monitor how perturbations propagate through networks over time
- **Dose-Response Perturbation Analysis**: Quantify network sensitivity to perturbation strength

### **Network Perturbation Quantification**
- **Velocity-Network Correlations**: Link transcriptional dynamics to regulatory network structure
- **Perturbed Module Identification**: Isolate specific regulatory modules affected by perturbations
- **Perturbation Propagation Analysis**: Track how perturbations spread through network hierarchies
- **Cross-Modal Perturbation Effects**: Analyze perturbation effects on both regulatory and protein networks
- **Perturbation Signature Discovery**: Identify characteristic patterns of network perturbation

### **Supporting Technologies**

#### **scVI Integration**
- **Denoised Expression**: Improved input quality for perturbation-sensitive network inference
- **Batch Correction**: Remove technical variation to enhance perturbation signal detection
- **Multi-modal Analysis**: Joint RNA+ATAC modeling for comprehensive perturbation analysis

#### **Optimal Transport (OT)**
- **Perturbation Trajectory Analysis**: Map perturbation paths through network space using Wasserstein distances
- **Cross-Condition Alignment**: Align networks across perturbation conditions for comparison
- **Perturbation Quantification**: Measure perturbation strength using optimal transport metrics

## **Applications: Perturbation-Driven Discovery**

### **Drug Discovery & Development**
- **Early Drug Response Detection**: Identify drug effects through splicing dynamics before expression changes
- **Drug Target Validation**: Pinpoint which networks are perturbed by drug candidates
- **Off-Target Effect Detection**: Discover unintended network perturbations causing side effects
- **Dose-Response Optimization**: Determine optimal drug concentrations for network perturbation
- **Drug Resistance Mechanisms**: Identify network adaptations that lead to treatment resistance

### **Experimental Perturbation Studies**
- **Genetic Perturbation Analysis**: Detect network changes from CRISPR/RNAi experiments
- **Environmental Stress Response**: Identify perturbed networks in response to stress conditions
- **Developmental Perturbations**: Track network changes during perturbed development
- **Disease Progression Monitoring**: Detect network perturbations in disease models
- **Treatment Response Prediction**: Predict therapeutic outcomes based on network perturbation patterns

### **Precision Medicine Applications**
- **Patient Stratification**: Group patients based on network perturbation signatures
- **Biomarker Discovery**: Identify splicing-based biomarkers for drug response
- **Personalized Treatment**: Tailor therapies based on individual network perturbation profiles
- **Resistance Prediction**: Forecast treatment resistance through network perturbation analysis

### **Technical Features**
- Support for multiple data formats (pickle, CSV, HDF5, NPZ)
- Automated identifier mapping between genes and proteins
- Quality control and validation at each integration step

### **Perturbation Analysis & Quantification**
- **Velocity-Based Perturbation Scoring**: Quantify perturbation strength using RNA velocity metrics
- **Splicing Event Statistical Testing**: Statistical analysis of unspliced/spliced ratio changes
- **Network Perturbation Significance**: Test for significant changes in regulatory network structure
- **Temporal Perturbation Profiling**: Track perturbation dynamics over time courses
- **Multi-Modal Perturbation Integration**: Combine regulatory and protein network perturbation signals

### *Perturbation Visualization**
- **Velocity Vector Plots**: Visualize RNA velocity vectors overlaid on network structures
- **Perturbation Heatmaps**: Show perturbation strength across genes and conditions
- **Network Perturbation Graphs**: Highlight perturbed edges and nodes in network visualizations
- **Temporal Perturbation Trajectories**: Track perturbation propagation through time
- **Splicing Dynamics Plots**: Visualize unspliced/spliced ratios and velocity magnitudes

### **Scalable Architecture**
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

## Quick Start: Perturbation Detection

### Velocity-Enhanced Perturbation Analysis

```python
from bridge import VelocityEnhancedBridge
import scanpy as sc

# Load your single-cell data with spliced/unspliced counts
adata = sc.read_h5ad('drug_treatment_data.h5ad')  # Must contain 'spliced' and 'unspliced' layers

# Initialize BRIDGE with velocity-based perturbation detection
velocity_bridge = VelocityEnhancedBridge(config={
    'velocity': {
        'mode': 'dynamical',  # Use dynamical model for perturbation sensitivity
        'n_top_genes': 2000,
        'min_shared_counts': 20
    },
    'bridge': {
        'scenic': {'min_regulon_size': 10},
        'pinnacle': {'normalize_embeddings': True}
    }
})

# Detect network perturbations from drug treatment
results = velocity_bridge.analyze_drug_response_networks(
    rna_adata=adata,
    scenic_networks=scenic_networks,  # Your SCENIC+ networks
    pinnacle_embeddings=pinnacle_embeddings,  # Your PINNACLE embeddings
    drug_conditions=['drug_treatment'],
    control_condition='control',
    condition_key='condition'
)

# Extract perturbed networks and targets
perturbed_networks = results['drug_response_results']['drug_treatment']['perturbed_modules']
drug_targets = results['pharmacological_targets']['drug_treatment']['top_targets']

print(f"Detected {len(perturbed_networks['nodes_with_changes'])} perturbed genes")
print(f"Top drug targets: {[target[0] for target in drug_targets[:5]]}")
```

### Splicing-Based Perturbation Detection

```python
from bridge.velocity_integration import VelocityNetworkAnalyzer

# Initialize velocity analyzer for perturbation detection
analyzer = VelocityNetworkAnalyzer(config={
    'velocity': {'mode': 'dynamical'},
    'network_dynamics': {'velocity_threshold': 0.1}
})

# Compute RNA velocity to detect transcriptional perturbations
velocity_adata = analyzer.compute_rna_velocity(
    adata, condition_key='treatment'
)

# Analyze which networks are perturbed
network_dynamics = analyzer.analyze_network_dynamics(
    scenic_networks, velocity_adata, condition_key='treatment'
)

# Identify velocity-responsive modules (perturbed by treatment)
for condition, dynamics in network_dynamics.items():
    responsive_modules = dynamics['responsive_modules']
    print(f"{condition}: {len(responsive_modules)} responsive modules detected")
    
    # Show velocity-network correlations
    correlations = dynamics['velocity_network_correlations']
    print(f"  Velocity-network correlation: {correlations.get('degree_velocity_corr', 0):.3f}")
```

### Installation with Velocity Support

```bash
# Install with RNA velocity capabilities
pip install bridge-omics[velocity]

# Or install all features including velocity
pip install bridge-omics[all]

# Direct velocity dependencies
pip install scvelo cellrank velocyto
```

## Key Features for Perturbation Detection

### **Splicing Event Analysis**
- **Unspliced/Spliced Ratio Monitoring**: Track changes in transcriptional dynamics
- **Velocity Magnitude Analysis**: Quantify transcriptional activity changes
- **Splicing Kinetics**: Analyze transcriptional and splicing rate changes
- **Early Response Detection**: Identify perturbations before steady-state changes

### **Network Perturbation Mapping**
- **Velocity-Network Correlations**: Link transcriptional dynamics to network topology
- **Perturbed Module Discovery**: Identify regulatory modules affected by perturbations
- **Perturbation Propagation**: Track how perturbations spread through regulatory hierarchies
- **Cross-Modal Effects**: Analyze perturbation effects on both regulatory and protein networks

### **Drug Response Applications**
- **Drug Target Identification**: Find regulatory nodes most affected by drug treatment
- **Mechanism of Action**: Understand how drugs perturb regulatory networks
- **Dose-Response Analysis**: Quantify network sensitivity to drug concentration
- **Resistance Mechanisms**: Identify network adaptations leading to drug resistance

## Data Requirements

### **For Perturbation Detection**
- **RNA Velocity Data**: Single-cell RNA-seq with spliced/unspliced counts
- **SCENIC+ Networks**: Regulatory networks from multi-omic single-cell data
- **PINNACLE Embeddings**: Context-specific protein embeddings
- **Condition Labels**: Clear annotation of control vs. perturbed conditions

### **Supported Formats**
- **AnnData**: `.h5ad` files with velocity layers
- **SCENIC+ Output**: Pickle files or CSV format
- **PINNACLE Output**: Pickle files, CSV, or NPZ format
- **Metadata**: Condition, time point, dose information

## Examples

See the `examples/` directory for comprehensive tutorials:

- `velocity_drug_response_example.py`: Complete drug response analysis workflow
- `basic_integration_example.py`: Standard BRIDGE integration
- `enhanced_bridge_example.py`: Advanced features with scVI and optimal transport

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use BRIDGE in your research, please cite:

```bibtex
@software{bridge2025,
  title={BRIDGE: Biological Regulatory Integration for Dynamic Gene Expression},
  author={Manus AI},
  year={2025},
  url={https://github.com/manus-ai/bridge}
}
```

## Support

- **Documentation**: [https://bridge-omics.readthedocs.io/](https://bridge-omics.readthedocs.io/)
- **Issues**: [GitHub Issues](https://github.com/manus-ai/bridge/issues)
- **Discussions**: [GitHub Discussions](https://github.com/manus-ai/bridge/discussions)


<img width="462" height="642" alt="image" src="https://github.com/user-attachments/assets/8017d5b8-193d-48dd-b839-73aa3240e3f6" />

