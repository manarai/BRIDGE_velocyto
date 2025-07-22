"""
Network analysis module for SCENIC+ and PINNACLE integration.

This module provides classes for integrating regulatory networks with protein
embeddings and performing differential analysis across conditions.
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple, Union
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import community as community_louvain

class NetworkIntegrator:
    """
    Integrates SCENIC+ regulatory networks with PINNACLE protein embeddings.
    
    Creates unified networks that combine regulatory relationships with
    protein functional contexts.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize network integrator.
        
        Args:
            config: Configuration parameters for integration
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__ + '.NetworkIntegrator')
    
    def integrate(self, 
                 scenic_network: nx.DiGraph,
                 protein_embeddings: Dict,
                 gene_protein_mapping: Dict) -> nx.Graph:
        """
        Integrate SCENIC+ network with PINNACLE embeddings.
        
        Args:
            scenic_network: SCENIC+ regulatory network
            protein_embeddings: PINNACLE protein embeddings
            gene_protein_mapping: Mapping from genes to proteins
            
        Returns:
            Integrated network combining regulatory and protein information
        """
        self.logger.info("Starting network integration")
        
        # Create integrated network
        integrated_network = nx.Graph()
        
        # Add regulatory edges from SCENIC+
        self._add_regulatory_edges(integrated_network, scenic_network, gene_protein_mapping)
        
        # Add protein similarity edges from PINNACLE
        self._add_protein_edges(integrated_network, protein_embeddings)
        
        # Add cross-layer connections
        self._add_cross_layer_edges(integrated_network, scenic_network, 
                                  protein_embeddings, gene_protein_mapping)
        
        # Compute network properties
        self._compute_network_properties(integrated_network)
        
        self.logger.info(f"Integration complete: {integrated_network.number_of_nodes()} nodes, "
                        f"{integrated_network.number_of_edges()} edges")
        
        return integrated_network
    
    def _add_regulatory_edges(self, 
                            integrated_network: nx.Graph,
                            scenic_network: nx.DiGraph,
                            gene_protein_mapping: Dict) -> None:
        """Add regulatory edges from SCENIC+ network."""
        for u, v, data in scenic_network.edges(data=True):
            # Map genes to proteins if possible
            u_protein = gene_protein_mapping.get(u, u)
            v_protein = gene_protein_mapping.get(v, v)
            
            # Add nodes with regulatory information
            integrated_network.add_node(u_protein, 
                                      node_type='regulatory',
                                      original_id=u,
                                      scenic_type=scenic_network.nodes[u].get('node_type', 'unknown'))
            
            integrated_network.add_node(v_protein,
                                      node_type='regulatory', 
                                      original_id=v,
                                      scenic_type=scenic_network.nodes[v].get('node_type', 'unknown'))
            
            # Add regulatory edge
            edge_attrs = {
                'edge_type': 'regulatory',
                'importance': data.get('importance', 0),
                'enhancer': data.get('enhancer', None),
                'source': 'SCENIC+'
            }
            
            integrated_network.add_edge(u_protein, v_protein, **edge_attrs)
    
    def _add_protein_edges(self, 
                         integrated_network: nx.Graph,
                         protein_embeddings: Dict) -> None:
        """Add protein similarity edges from PINNACLE embeddings."""
        if not protein_embeddings['protein_ids']:
            return
        
        # Compute protein similarities
        similarity_matrix = cosine_similarity(protein_embeddings['embeddings'])
        protein_ids = protein_embeddings['protein_ids']
        
        # Add protein nodes and edges
        similarity_threshold = self.config.get('similarity_threshold', 0.5)
        
        for i, protein1 in enumerate(protein_ids):
            # Add protein node
            if not integrated_network.has_node(protein1):
                integrated_network.add_node(protein1, 
                                          node_type='protein',
                                          embedding_available=True)
            else:
                # Update existing node
                integrated_network.nodes[protein1]['embedding_available'] = True
            
            for j, protein2 in enumerate(protein_ids):
                if i < j:  # Avoid duplicate edges
                    similarity = similarity_matrix[i, j]
                    
                    if similarity > similarity_threshold:
                        edge_attrs = {
                            'edge_type': 'protein_similarity',
                            'similarity': similarity,
                            'source': 'PINNACLE'
                        }
                        
                        integrated_network.add_edge(protein1, protein2, **edge_attrs)
    
    def _add_cross_layer_edges(self,
                             integrated_network: nx.Graph,
                             scenic_network: nx.DiGraph,
                             protein_embeddings: Dict,
                             gene_protein_mapping: Dict) -> None:
        """Add edges connecting regulatory and protein layers."""
        protein_ids = set(protein_embeddings['protein_ids'])
        
        # Connect TFs to their protein representations
        for node in scenic_network.nodes():
            if scenic_network.nodes[node].get('node_type') == 'TF':
                protein_id = gene_protein_mapping.get(node, node)
                
                if protein_id in protein_ids:
                    # Add cross-layer edge
                    if integrated_network.has_node(node) and integrated_network.has_node(protein_id):
                        edge_attrs = {
                            'edge_type': 'cross_layer',
                            'connection_type': 'TF_to_protein',
                            'source': 'integration'
                        }
                        integrated_network.add_edge(node, protein_id, **edge_attrs)
    
    def _compute_network_properties(self, network: nx.Graph) -> None:
        """Compute and store network-level properties."""
        # Basic properties
        network.graph['num_nodes'] = network.number_of_nodes()
        network.graph['num_edges'] = network.number_of_edges()
        network.graph['density'] = nx.density(network)
        
        # Connectivity
        if network.number_of_nodes() > 0:
            network.graph['avg_clustering'] = nx.average_clustering(network)
            network.graph['num_components'] = nx.number_connected_components(network)
            
            # Centrality measures for largest component
            largest_cc = max(nx.connected_components(network), key=len)
            subgraph = network.subgraph(largest_cc)
            
            if subgraph.number_of_nodes() > 1:
                centrality = nx.degree_centrality(subgraph)
                network.graph['max_degree_centrality'] = max(centrality.values())
                network.graph['avg_degree_centrality'] = np.mean(list(centrality.values()))
        
        # Edge type distribution
        edge_types = {}
        for u, v, data in network.edges(data=True):
            edge_type = data.get('edge_type', 'unknown')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        
        network.graph['edge_type_distribution'] = edge_types
    
    def identify_network_modules(self, network: nx.Graph,
                               method: str = 'louvain') -> Dict:
        """
        Identify modules/communities in the integrated network.
        
        Args:
            network: Integrated network
            method: Community detection method ('louvain', 'kmeans')
            
        Returns:
            Dictionary mapping nodes to module IDs
        """
        if method == 'louvain':
            # Use Louvain algorithm for community detection
            partition = community_louvain.best_partition(network)
            
        elif method == 'kmeans':
            # Use k-means clustering on node embeddings
            n_clusters = self.config.get('n_clusters', 10)
            
            # Create node feature matrix
            node_features = self._create_node_features(network)
            
            if node_features.shape[0] > n_clusters:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(node_features)
                
                partition = {}
                for i, node in enumerate(network.nodes()):
                    partition[node] = cluster_labels[i]
            else:
                # Too few nodes for clustering
                partition = {node: 0 for node in network.nodes()}
        
        else:
            raise ValueError(f"Unsupported clustering method: {method}")
        
        return partition
    
    def _create_node_features(self, network: nx.Graph) -> np.ndarray:
        """Create feature matrix for network nodes."""
        nodes = list(network.nodes())
        features = []
        
        for node in nodes:
            node_data = network.nodes[node]
            
            # Basic features
            degree = network.degree(node)
            clustering = nx.clustering(network, node)
            
            # Node type features
            is_regulatory = 1 if node_data.get('node_type') == 'regulatory' else 0
            is_protein = 1 if node_data.get('node_type') == 'protein' else 0
            has_embedding = 1 if node_data.get('embedding_available', False) else 0
            
            # Edge type features
            regulatory_edges = sum(1 for _, _, d in network.edges(node, data=True) 
                                 if d.get('edge_type') == 'regulatory')
            protein_edges = sum(1 for _, _, d in network.edges(node, data=True)
                              if d.get('edge_type') == 'protein_similarity')
            
            node_features = [
                degree, clustering, is_regulatory, is_protein, 
                has_embedding, regulatory_edges, protein_edges
            ]
            
            features.append(node_features)
        
        return np.array(features)


class DifferentialAnalyzer:
    """
    Performs differential analysis between integrated networks.
    
    Identifies regulatory and protein changes between conditions.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize differential analyzer.
        
        Args:
            config: Configuration parameters for analysis
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__ + '.DifferentialAnalyzer')
    
    def analyze(self,
               network1: nx.Graph,
               network2: nx.Graph,
               analysis_type: str = 'both') -> Dict:
        """
        Perform differential analysis between two networks.
        
        Args:
            network1: First network for comparison
            network2: Second network for comparison
            analysis_type: Type of analysis ('regulatory', 'protein', 'both')
            
        Returns:
            Dictionary containing differential analysis results
        """
        self.logger.info(f"Starting differential analysis ({analysis_type})")
        
        results = {}
        
        if analysis_type in ['regulatory', 'both']:
            results['regulatory'] = self._analyze_regulatory_differences(network1, network2)
        
        if analysis_type in ['protein', 'both']:
            results['protein'] = self._analyze_protein_differences(network1, network2)
        
        if analysis_type == 'both':
            results['integrated'] = self._analyze_integrated_differences(network1, network2)
        
        # Compute summary statistics
        results['summary'] = self._compute_summary_statistics(results)
        
        self.logger.info("Differential analysis complete")
        return results
    
    def _analyze_regulatory_differences(self, 
                                     network1: nx.Graph,
                                     network2: nx.Graph) -> Dict:
        """Analyze differences in regulatory edges."""
        # Extract regulatory edges
        reg_edges1 = self._extract_regulatory_edges(network1)
        reg_edges2 = self._extract_regulatory_edges(network2)
        
        # Find unique and common edges
        edges1_set = set(reg_edges1.keys())
        edges2_set = set(reg_edges2.keys())
        
        unique_to_1 = edges1_set - edges2_set
        unique_to_2 = edges2_set - edges1_set
        common_edges = edges1_set & edges2_set
        
        # Analyze importance changes for common edges
        importance_changes = []
        for edge in common_edges:
            imp1 = reg_edges1[edge]['importance']
            imp2 = reg_edges2[edge]['importance']
            
            fold_change = imp2 / imp1 if imp1 != 0 else np.inf
            
            importance_changes.append({
                'edge': edge,
                'importance_1': imp1,
                'importance_2': imp2,
                'fold_change': fold_change,
                'log_fold_change': np.log2(fold_change) if fold_change > 0 and fold_change != np.inf else np.nan
            })
        
        importance_df = pd.DataFrame(importance_changes)
        
        # Statistical testing
        if len(importance_changes) > 0:
            # Paired t-test for importance changes
            imp1_values = importance_df['importance_1'].values
            imp2_values = importance_df['importance_2'].values
            
            valid_pairs = ~(np.isnan(imp1_values) | np.isnan(imp2_values))
            if np.sum(valid_pairs) > 1:
                t_stat, p_value = stats.ttest_rel(
                    imp1_values[valid_pairs], 
                    imp2_values[valid_pairs]
                )
            else:
                t_stat, p_value = np.nan, np.nan
        else:
            t_stat, p_value = np.nan, np.nan
        
        return {
            'unique_to_condition1': list(unique_to_1),
            'unique_to_condition2': list(unique_to_2),
            'common_edges': list(common_edges),
            'importance_changes': importance_df,
            'statistical_test': {
                't_statistic': t_stat,
                'p_value': p_value
            }
        }
    
    def _analyze_protein_differences(self,
                                   network1: nx.Graph,
                                   network2: nx.Graph) -> Dict:
        """Analyze differences in protein similarity networks."""
        # Extract protein similarity edges
        prot_edges1 = self._extract_protein_edges(network1)
        prot_edges2 = self._extract_protein_edges(network2)
        
        # Find unique and common edges
        edges1_set = set(prot_edges1.keys())
        edges2_set = set(prot_edges2.keys())
        
        unique_to_1 = edges1_set - edges2_set
        unique_to_2 = edges2_set - edges1_set
        common_edges = edges1_set & edges2_set
        
        # Analyze similarity changes for common edges
        similarity_changes = []
        for edge in common_edges:
            sim1 = prot_edges1[edge]['similarity']
            sim2 = prot_edges2[edge]['similarity']
            
            similarity_changes.append({
                'edge': edge,
                'similarity_1': sim1,
                'similarity_2': sim2,
                'similarity_change': sim2 - sim1
            })
        
        similarity_df = pd.DataFrame(similarity_changes)
        
        # Statistical testing
        if len(similarity_changes) > 0:
            sim_changes = similarity_df['similarity_change'].values
            valid_changes = ~np.isnan(sim_changes)
            
            if np.sum(valid_changes) > 1:
                t_stat, p_value = stats.ttest_1samp(sim_changes[valid_changes], 0)
            else:
                t_stat, p_value = np.nan, np.nan
        else:
            t_stat, p_value = np.nan, np.nan
        
        return {
            'unique_to_condition1': list(unique_to_1),
            'unique_to_condition2': list(unique_to_2),
            'common_edges': list(common_edges),
            'similarity_changes': similarity_df,
            'statistical_test': {
                't_statistic': t_stat,
                'p_value': p_value
            }
        }
    
    def _analyze_integrated_differences(self,
                                      network1: nx.Graph,
                                      network2: nx.Graph) -> Dict:
        """Analyze differences in integrated network properties."""
        # Network-level properties
        props1 = network1.graph
        props2 = network2.graph
        
        property_changes = {}
        for prop in ['density', 'avg_clustering', 'num_components']:
            if prop in props1 and prop in props2:
                property_changes[prop] = {
                    'condition1': props1[prop],
                    'condition2': props2[prop],
                    'change': props2[prop] - props1[prop]
                }
        
        # Node-level analysis
        nodes1 = set(network1.nodes())
        nodes2 = set(network2.nodes())
        
        unique_nodes_1 = nodes1 - nodes2
        unique_nodes_2 = nodes2 - nodes1
        common_nodes = nodes1 & nodes2
        
        # Degree changes for common nodes
        degree_changes = []
        for node in common_nodes:
            deg1 = network1.degree(node)
            deg2 = network2.degree(node)
            
            degree_changes.append({
                'node': node,
                'degree_1': deg1,
                'degree_2': deg2,
                'degree_change': deg2 - deg1
            })
        
        degree_df = pd.DataFrame(degree_changes)
        
        return {
            'property_changes': property_changes,
            'unique_nodes_condition1': list(unique_nodes_1),
            'unique_nodes_condition2': list(unique_nodes_2),
            'common_nodes': list(common_nodes),
            'degree_changes': degree_df
        }
    
    def _extract_regulatory_edges(self, network: nx.Graph) -> Dict:
        """Extract regulatory edges from network."""
        reg_edges = {}
        
        for u, v, data in network.edges(data=True):
            if data.get('edge_type') == 'regulatory':
                edge_key = tuple(sorted([u, v]))
                reg_edges[edge_key] = data
        
        return reg_edges
    
    def _extract_protein_edges(self, network: nx.Graph) -> Dict:
        """Extract protein similarity edges from network."""
        prot_edges = {}
        
        for u, v, data in network.edges(data=True):
            if data.get('edge_type') == 'protein_similarity':
                edge_key = tuple(sorted([u, v]))
                prot_edges[edge_key] = data
        
        return prot_edges
    
    def _compute_summary_statistics(self, results: Dict) -> Dict:
        """Compute summary statistics for differential analysis."""
        summary = {}
        
        # Regulatory summary
        if 'regulatory' in results:
            reg_results = results['regulatory']
            summary['regulatory'] = {
                'num_unique_condition1': len(reg_results['unique_to_condition1']),
                'num_unique_condition2': len(reg_results['unique_to_condition2']),
                'num_common': len(reg_results['common_edges']),
                'significant_changes': 0
            }
            
            # Count significant importance changes
            if not reg_results['importance_changes'].empty:
                threshold = self.config.get('fold_change_threshold', 1.5)
                significant = np.abs(reg_results['importance_changes']['log_fold_change']) > np.log2(threshold)
                summary['regulatory']['significant_changes'] = np.sum(significant)
        
        # Protein summary
        if 'protein' in results:
            prot_results = results['protein']
            summary['protein'] = {
                'num_unique_condition1': len(prot_results['unique_to_condition1']),
                'num_unique_condition2': len(prot_results['unique_to_condition2']),
                'num_common': len(prot_results['common_edges']),
                'significant_changes': 0
            }
            
            # Count significant similarity changes
            if not prot_results['similarity_changes'].empty:
                threshold = self.config.get('similarity_change_threshold', 0.1)
                significant = np.abs(prot_results['similarity_changes']['similarity_change']) > threshold
                summary['protein']['significant_changes'] = np.sum(significant)
        
        return summary
    
    def identify_differential_modules(self,
                                    network1: nx.Graph,
                                    network2: nx.Graph) -> Dict:
        """
        Identify modules that differ between networks.
        
        Args:
            network1: First network
            network2: Second network
            
        Returns:
            Dictionary with differential module information
        """
        # Get modules for each network
        integrator = NetworkIntegrator(self.config)
        modules1 = integrator.identify_network_modules(network1)
        modules2 = integrator.identify_network_modules(network2)
        
        # Find nodes that changed modules
        common_nodes = set(modules1.keys()) & set(modules2.keys())
        
        module_changes = []
        for node in common_nodes:
            if modules1[node] != modules2[node]:
                module_changes.append({
                    'node': node,
                    'module_1': modules1[node],
                    'module_2': modules2[node]
                })
        
        # Analyze module stability
        module_stability = self._compute_module_stability(modules1, modules2, common_nodes)
        
        return {
            'module_changes': pd.DataFrame(module_changes),
            'module_stability': module_stability,
            'modules_condition1': modules1,
            'modules_condition2': modules2
        }
    
    def _compute_module_stability(self, 
                                modules1: Dict,
                                modules2: Dict,
                                common_nodes: set) -> float:
        """Compute module stability between conditions."""
        if not common_nodes:
            return 0.0
        
        stable_nodes = sum(1 for node in common_nodes 
                          if modules1[node] == modules2[node])
        
        return stable_nodes / len(common_nodes)

