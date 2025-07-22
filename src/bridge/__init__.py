"""
BRIDGE: Biological Regulatory Integration for Dynamic Gene Expression

A comprehensive framework for integrating SCENIC+ enhancer-driven gene regulatory 
networks with PINNACLE context-specific protein embeddings to enable multi-omic 
network analysis quantifiable across conditions.

This package provides tools for:
- Loading and processing SCENIC+ regulatory networks
- Loading and processing PINNACLE protein embeddings  
- Integrating regulatory and protein networks
- Performing differential analysis across conditions
- Generating comprehensive visualizations
- Quality control and validation

Main Classes:
    BridgeIntegrator: Primary interface for the integration framework
    ScenicProcessor: Handles SCENIC+ regulatory network data
    PinnacleProcessor: Manages PINNACLE protein embeddings
    NetworkIntegrator: Combines regulatory and protein networks
    DifferentialAnalyzer: Performs cross-condition comparisons
    NetworkVisualizer: Creates publication-ready visualizations

Example:
    >>> from bridge import BridgeIntegrator
    >>> integrator = BridgeIntegrator()
    >>> integrator.load_scenic_data('scenic_networks.pkl')
    >>> integrator.load_pinnacle_data('pinnacle_embeddings.pkl')
    >>> integrated_networks = integrator.integrate_networks()
    >>> diff_results = integrator.differential_analysis('cond1', 'cond2')

Author: Manus AI
Version: 1.0.0
License: MIT
"""

from .core import BridgeIntegrator
from .data_processing import ScenicProcessor, PinnacleProcessor
from .network_analysis import NetworkIntegrator, DifferentialAnalyzer
from .visualization import NetworkVisualizer
from .utils import IdentifierMapper, QualityController
from .scvi_integration import ScviPreprocessor, ScviEnhancedBridge
from .optimal_transport import OptimalTransportIntegrator, OTEnhancedBridge
from .velocity_integration import VelocityNetworkAnalyzer, VelocityEnhancedBridge

__version__ = "1.0.0"
__author__ = "Manus AI"
__email__ = "contact@manus.ai"
__license__ = "MIT"

# Main exports
__all__ = [
    "BridgeIntegrator",
    "ScenicProcessor", 
    "PinnacleProcessor",
    "NetworkIntegrator",
    "DifferentialAnalyzer", 
    "NetworkVisualizer",
    "IdentifierMapper",
    "QualityController",
    "ScviPreprocessor",
    "ScviEnhancedBridge",
    "OptimalTransportIntegrator",
    "OTEnhancedBridge",
    "VelocityNetworkAnalyzer",
    "VelocityEnhancedBridge"
]

