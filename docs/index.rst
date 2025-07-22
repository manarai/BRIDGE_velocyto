BRIDGE: Biological Regulatory Integration for Dynamic Gene Expression
====================================================================

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.8+

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

BRIDGE is a comprehensive Python framework for integrating SCENIC+ enhancer-driven gene regulatory networks with PINNACLE context-specific protein embeddings to enable multi-omic network analysis quantifiable across conditions.

Key Features
------------

* **Multi-omic Integration**: Seamlessly combines regulatory networks and protein embeddings
* **scVI Enhancement**: Denoised expression and robust latent representations
* **Optimal Transport**: Principled cross-modal alignment and condition comparison
* **Quantitative Analysis**: Statistical comparison of networks across conditions
* **Rich Visualizations**: Publication-ready plots and interactive visualizations
* **Scalable Processing**: Memory-efficient for large datasets

Quick Start
-----------

Installation::

    pip install bridge-omics

Basic Usage::

    from bridge import BridgeIntegrator

    # Initialize integrator
    integrator = BridgeIntegrator()

    # Load data
    integrator.load_scenic_data('scenic_networks.pkl')
    integrator.load_pinnacle_data('pinnacle_embeddings.pkl')

    # Integrate networks
    integrated_networks = integrator.integrate_networks()

    # Perform differential analysis
    diff_results = integrator.differential_analysis('condition1', 'condition2')

Enhanced Usage with scVI and Optimal Transport::

    from bridge import ScviEnhancedBridge, OTEnhancedBridge

    # scVI-enhanced preprocessing
    scvi_bridge = ScviEnhancedBridge()
    results = scvi_bridge.preprocess_and_integrate(
        rna_adata=rna_data,
        atac_adata=atac_data,
        condition_key='condition'
    )

    # Optimal transport alignment
    ot_bridge = OTEnhancedBridge()
    ot_results = ot_bridge.integrate_with_ot_alignment(
        scenic_networks=scenic_networks,
        pinnacle_embeddings=pinnacle_embeddings
    )

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   tutorials
   examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/core
   api/data_processing
   api/network_analysis
   api/scvi_integration
   api/optimal_transport
   api/visualization
   api/utils

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

