"""
Core integration module for SCENIC+ and PINNACLE integration.

This module provides the main ScenicPinnacleIntegrator class that orchestrates
the complete integration workflow from data processing through analysis.
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import pickle
import json

from .data_processing import ScenicProcessor, PinnacleProcessor
from .network_analysis import NetworkIntegrator, DifferentialAnalyzer
from .utils import IdentifierMapper, QualityController
from .visualization import NetworkVisualizer

class BridgeIntegrator:
    """
    BRIDGE: Biological Regulatory Integration for Dynamic Gene Expression
    
    Main integration class for combining SCENIC+ enhancer-driven gene regulatory
    networks with PINNACLE context-specific protein embeddings.
    
    This class orchestrates the complete workflow from data input through
    integrated network analysis and visualization.
    
    Attributes:
        scenic_processor: Handler for SCENIC+ data processing
        pinnacle_processor: Handler for PINNACLE data processing
        network_integrator: Handler for network integration
        differential_analyzer: Handler for differential analysis
        identifier_mapper: Handler for gene/protein ID mapping
        quality_controller: Handler for quality control
        visualizer: Handler for visualization
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the integrator with optional configuration.
        
        Args:
            config: Configuration dictionary with analysis parameters
        """
        self.config = config or self._default_config()
        self.logger = self._setup_logging()
        
        # Initialize component modules
        self.scenic_processor = ScenicProcessor(config=self.config.get('scenic', {}))
        self.pinnacle_processor = PinnacleProcessor(config=self.config.get('pinnacle', {}))
        self.network_integrator = NetworkIntegrator(config=self.config.get('integration', {}))
        self.differential_analyzer = DifferentialAnalyzer(config=self.config.get('differential', {}))
        self.identifier_mapper = IdentifierMapper(config=self.config.get('mapping', {}))
        self.quality_controller = QualityController(config=self.config.get('quality', {}))
        self.visualizer = NetworkVisualizer(config=self.config.get('visualization', {}))
        
        # Storage for analysis results
        self.scenic_networks = {}
        self.pinnacle_embeddings = {}
        self.integrated_networks = {}
        self.differential_results = {}
        
    def _default_config(self) -> Dict:
        """Return default configuration parameters."""
        return {
            'scenic': {
                'min_regulon_size': 10,
                'min_target_genes': 5,
                'auc_threshold': 0.05
            },
            'pinnacle': {
                'embedding_dim': 256,
                'context_threshold': 0.1
            },
            'integration': {
                'min_overlap': 3,
                'similarity_threshold': 0.5
            },
            'differential': {
                'pvalue_threshold': 0.05,
                'fold_change_threshold': 1.5
            },
            'mapping': {
                'mapping_databases': ['hgnc', 'uniprot', 'ensembl'],
                'validation_strict': True
            },
            'quality': {
                'min_network_size': 10,
                'max_network_size': 10000,
                'connectivity_threshold': 0.1
            },
            'visualization': {
                'layout_algorithm': 'force_atlas2',
                'node_size_range': (10, 100),
                'edge_width_range': (0.5, 5.0)
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the integration workflow."""
        logger = logging.getLogger('scenic_pinnacle_integration')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_scenic_data(self, 
                        scenic_path: Union[str, Path],
                        data_format: str = 'pickle') -> None:
        """
        Load SCENIC+ regulatory network data.
        
        Args:
            scenic_path: Path to SCENIC+ output files
            data_format: Format of SCENIC+ data ('pickle', 'csv', 'h5ad')
        """
        self.logger.info(f"Loading SCENIC+ data from {scenic_path}")
        
        try:
            self.scenic_networks = self.scenic_processor.load_data(
                scenic_path, data_format
            )
            self.logger.info(f"Loaded {len(self.scenic_networks)} SCENIC+ networks")
            
            # Quality control
            self.scenic_networks = self.quality_controller.validate_scenic_networks(
                self.scenic_networks
            )
            
        except Exception as e:
            self.logger.error(f"Error loading SCENIC+ data: {e}")
            raise
    
    def load_pinnacle_data(self,
                          pinnacle_path: Union[str, Path],
                          data_format: str = 'pickle') -> None:
        """
        Load PINNACLE protein embedding data.
        
        Args:
            pinnacle_path: Path to PINNACLE output files
            data_format: Format of PINNACLE data ('pickle', 'csv', 'npz')
        """
        self.logger.info(f"Loading PINNACLE data from {pinnacle_path}")
        
        try:
            self.pinnacle_embeddings = self.pinnacle_processor.load_data(
                pinnacle_path, data_format
            )
            self.logger.info(f"Loaded embeddings for {len(self.pinnacle_embeddings)} contexts")
            
            # Quality control
            self.pinnacle_embeddings = self.quality_controller.validate_pinnacle_embeddings(
                self.pinnacle_embeddings
            )
            
        except Exception as e:
            self.logger.error(f"Error loading PINNACLE data: {e}")
            raise
    
    def integrate_networks(self, 
                          conditions: Optional[List[str]] = None) -> Dict:
        """
        Integrate SCENIC+ networks with PINNACLE embeddings.
        
        Args:
            conditions: List of conditions to integrate (default: all)
            
        Returns:
            Dictionary of integrated networks by condition
        """
        self.logger.info("Starting network integration")
        
        if not self.scenic_networks or not self.pinnacle_embeddings:
            raise ValueError("Must load both SCENIC+ and PINNACLE data before integration")
        
        conditions = conditions or list(self.scenic_networks.keys())
        
        for condition in conditions:
            self.logger.info(f"Integrating networks for condition: {condition}")
            
            # Extract target genes from SCENIC+ networks
            target_genes = self.scenic_processor.extract_target_genes(
                self.scenic_networks[condition]
            )
            
            # Map gene IDs to protein IDs
            gene_protein_mapping = self.identifier_mapper.map_genes_to_proteins(
                target_genes
            )
            
            # Get relevant protein embeddings
            protein_embeddings = self.pinnacle_processor.get_context_embeddings(
                self.pinnacle_embeddings,
                condition,
                list(gene_protein_mapping.values())
            )
            
            # Integrate networks
            integrated_network = self.network_integrator.integrate(
                scenic_network=self.scenic_networks[condition],
                protein_embeddings=protein_embeddings,
                gene_protein_mapping=gene_protein_mapping
            )
            
            self.integrated_networks[condition] = integrated_network
            
        self.logger.info(f"Integration complete for {len(conditions)} conditions")
        return self.integrated_networks
    
    def differential_analysis(self,
                            condition1: str,
                            condition2: str,
                            analysis_type: str = 'both') -> Dict:
        """
        Perform differential analysis between conditions.
        
        Args:
            condition1: First condition for comparison
            condition2: Second condition for comparison
            analysis_type: Type of analysis ('regulatory', 'protein', 'both')
            
        Returns:
            Dictionary containing differential analysis results
        """
        self.logger.info(f"Performing differential analysis: {condition1} vs {condition2}")
        
        if condition1 not in self.integrated_networks or condition2 not in self.integrated_networks:
            raise ValueError("Both conditions must be integrated before differential analysis")
        
        comparison_key = f"{condition1}_vs_{condition2}"
        
        results = self.differential_analyzer.analyze(
            network1=self.integrated_networks[condition1],
            network2=self.integrated_networks[condition2],
            analysis_type=analysis_type
        )
        
        self.differential_results[comparison_key] = results
        
        self.logger.info(f"Differential analysis complete for {comparison_key}")
        return results
    
    def visualize_networks(self,
                          condition: str,
                          output_dir: Union[str, Path],
                          plot_types: List[str] = ['network', 'heatmap', 'embedding']) -> None:
        """
        Generate visualizations for integrated networks.
        
        Args:
            condition: Condition to visualize
            output_dir: Directory for output files
            plot_types: Types of plots to generate
        """
        self.logger.info(f"Generating visualizations for {condition}")
        
        if condition not in self.integrated_networks:
            raise ValueError(f"Condition {condition} not found in integrated networks")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        network = self.integrated_networks[condition]
        
        for plot_type in plot_types:
            self.logger.info(f"Generating {plot_type} plot")
            
            if plot_type == 'network':
                self.visualizer.plot_network(
                    network, 
                    output_path=output_dir / f"{condition}_network.png"
                )
            elif plot_type == 'heatmap':
                self.visualizer.plot_heatmap(
                    network,
                    output_path=output_dir / f"{condition}_heatmap.png"
                )
            elif plot_type == 'embedding':
                self.visualizer.plot_embeddings(
                    network,
                    output_path=output_dir / f"{condition}_embeddings.png"
                )
    
    def export_results(self,
                      output_dir: Union[str, Path],
                      formats: List[str] = ['pickle', 'csv', 'json']) -> None:
        """
        Export integration results in multiple formats.
        
        Args:
            output_dir: Directory for output files
            formats: List of output formats
        """
        self.logger.info("Exporting integration results")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export integrated networks
        for condition, network in self.integrated_networks.items():
            for fmt in formats:
                if fmt == 'pickle':
                    with open(output_dir / f"{condition}_integrated_network.pkl", 'wb') as f:
                        pickle.dump(network, f)
                elif fmt == 'csv':
                    # Export as edge list
                    edges_df = nx.to_pandas_edgelist(network)
                    edges_df.to_csv(output_dir / f"{condition}_edges.csv", index=False)
                    
                    # Export node attributes
                    nodes_df = pd.DataFrame.from_dict(
                        dict(network.nodes(data=True)), 
                        orient='index'
                    )
                    nodes_df.to_csv(output_dir / f"{condition}_nodes.csv")
                elif fmt == 'json':
                    network_data = nx.node_link_data(network)
                    with open(output_dir / f"{condition}_network.json", 'w') as f:
                        json.dump(network_data, f, indent=2)
        
        # Export differential results
        if self.differential_results:
            for comparison, results in self.differential_results.items():
                for fmt in formats:
                    if fmt == 'pickle':
                        with open(output_dir / f"{comparison}_differential.pkl", 'wb') as f:
                            pickle.dump(results, f)
                    elif fmt == 'csv':
                        for result_type, data in results.items():
                            if isinstance(data, pd.DataFrame):
                                data.to_csv(
                                    output_dir / f"{comparison}_{result_type}.csv",
                                    index=False
                                )
                    elif fmt == 'json':
                        # Convert numpy arrays to lists for JSON serialization
                        json_results = {}
                        for key, value in results.items():
                            if isinstance(value, np.ndarray):
                                json_results[key] = value.tolist()
                            elif isinstance(value, pd.DataFrame):
                                json_results[key] = value.to_dict('records')
                            else:
                                json_results[key] = value
                        
                        with open(output_dir / f"{comparison}_differential.json", 'w') as f:
                            json.dump(json_results, f, indent=2)
        
        self.logger.info(f"Results exported to {output_dir}")
    
    def run_complete_workflow(self,
                            scenic_path: Union[str, Path],
                            pinnacle_path: Union[str, Path],
                            output_dir: Union[str, Path],
                            conditions: Optional[List[str]] = None,
                            comparisons: Optional[List[Tuple[str, str]]] = None) -> Dict:
        """
        Run the complete integration workflow.
        
        Args:
            scenic_path: Path to SCENIC+ data
            pinnacle_path: Path to PINNACLE data
            output_dir: Directory for outputs
            conditions: Conditions to analyze
            comparisons: Condition pairs for differential analysis
            
        Returns:
            Dictionary containing all analysis results
        """
        self.logger.info("Starting complete integration workflow")
        
        # Load data
        self.load_scenic_data(scenic_path)
        self.load_pinnacle_data(pinnacle_path)
        
        # Integrate networks
        integrated_networks = self.integrate_networks(conditions)
        
        # Differential analysis
        if comparisons:
            for cond1, cond2 in comparisons:
                self.differential_analysis(cond1, cond2)
        
        # Generate visualizations
        output_dir = Path(output_dir)
        for condition in integrated_networks.keys():
            viz_dir = output_dir / 'visualizations' / condition
            self.visualize_networks(condition, viz_dir)
        
        # Export results
        self.export_results(output_dir / 'results')
        
        # Compile summary results
        summary = {
            'integrated_networks': list(integrated_networks.keys()),
            'differential_comparisons': list(self.differential_results.keys()),
            'network_statistics': self._compute_network_statistics(),
            'quality_metrics': self._compute_quality_metrics()
        }
        
        # Save summary
        with open(output_dir / 'analysis_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info("Complete workflow finished successfully")
        return summary
    
    def _compute_network_statistics(self) -> Dict:
        """Compute basic statistics for integrated networks."""
        stats = {}
        for condition, network in self.integrated_networks.items():
            stats[condition] = {
                'num_nodes': network.number_of_nodes(),
                'num_edges': network.number_of_edges(),
                'density': nx.density(network),
                'avg_clustering': nx.average_clustering(network) if network.number_of_nodes() > 0 else 0,
                'num_components': nx.number_connected_components(network.to_undirected())
            }
        return stats
    
    def _compute_quality_metrics(self) -> Dict:
        """Compute quality metrics for the integration."""
        metrics = {}
        
        # SCENIC+ network quality
        if self.scenic_networks:
            metrics['scenic_quality'] = self.quality_controller.assess_scenic_quality(
                self.scenic_networks
            )
        
        # PINNACLE embedding quality
        if self.pinnacle_embeddings:
            metrics['pinnacle_quality'] = self.quality_controller.assess_pinnacle_quality(
                self.pinnacle_embeddings
            )
        
        # Integration quality
        if self.integrated_networks:
            metrics['integration_quality'] = self.quality_controller.assess_integration_quality(
                self.integrated_networks
            )
        
        return metrics

