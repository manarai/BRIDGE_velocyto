"""
Optimal Transport Integration Module for BRIDGE Framework

This module provides optimal transport (OT) methods for multi-omics integration,
enabling principled alignment and comparison of SCENIC+ regulatory networks
with PINNACLE protein embeddings across different biological conditions.

Key Features:
- Cross-modal alignment between regulatory and protein networks
- Condition comparison using Wasserstein distances
- Batch effect correction with OT
- Multi-scale matching of genes to proteins
- Trajectory analysis of network evolution
- Gromov-Wasserstein for structure-aware alignment
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from pathlib import Path
import warnings
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler

# Optimal Transport imports with error handling
try:
    import ot
    from ot.gromov import gromov_wasserstein, fused_gromov_wasserstein
    from ot.unbalanced import sinkhorn_unbalanced
    from ot.bregman import sinkhorn
    OT_AVAILABLE = True
except ImportError:
    OT_AVAILABLE = False
    warnings.warn("POT (Python Optimal Transport) not available. Install with: pip install POT")

try:
    import moscot
    from moscot.problems import TemporalProblem, SpatialProblem
    MOSCOT_AVAILABLE = True
except ImportError:
    MOSCOT_AVAILABLE = False
    warnings.warn("moscot not available. Install with: pip install moscot")


class OptimalTransportIntegrator:
    """
    Optimal Transport methods for multi-omics integration in BRIDGE.
    
    Provides various OT algorithms for aligning regulatory networks
    with protein embeddings and comparing across conditions.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize OT integrator.
        
        Args:
            config: Configuration parameters for OT methods
        """
        if not OT_AVAILABLE:
            raise ImportError("POT is required but not installed. Install with: pip install POT")
        
        self.config = config or {}
        self.logger = logging.getLogger(__name__ + '.OptimalTransportIntegrator')
        
        # Default configuration
        self.default_config = {
            'reg': 0.1,  # Entropic regularization
            'reg_m': 1.0,  # Marginal relaxation
            'max_iter': 1000,
            'tol': 1e-9,
            'method': 'sinkhorn',  # 'sinkhorn', 'emd', 'unbalanced'
            'metric': 'euclidean',
            'normalize': True,
            'alpha': 0.5,  # For fused GW
            'armijo': False,
            'log': True
        }
        
        # Update with user config
        self.config = {**self.default_config, **self.config}
        
        # Storage for computed transport plans
        self.transport_plans = {}
        self.distance_matrices = {}
        self.alignment_results = {}
        
    def align_cross_modal(self,
                         scenic_networks: Dict[str, nx.Graph],
                         pinnacle_embeddings: Dict[str, Dict],
                         gene_protein_mapping: Optional[Dict[str, str]] = None) -> Dict[str, Dict]:
        """
        Align SCENIC+ regulatory networks with PINNACLE protein embeddings using OT.
        
        Args:
            scenic_networks: Dictionary of regulatory networks by condition
            pinnacle_embeddings: Dictionary of protein embeddings by condition
            gene_protein_mapping: Optional mapping from gene IDs to protein IDs
            
        Returns:
            Dictionary of alignment results by condition
        """
        self.logger.info("Performing cross-modal alignment with optimal transport...")
        
        alignment_results = {}
        
        for condition in scenic_networks.keys():
            if condition not in pinnacle_embeddings:
                self.logger.warning(f"No PINNACLE data for condition {condition}, skipping")
                continue
            
            self.logger.info(f"Aligning condition: {condition}")
            
            # Extract network features
            network_features = self._extract_network_features(scenic_networks[condition])
            
            # Extract protein features
            protein_features = pinnacle_embeddings[condition]['embeddings']
            protein_ids = pinnacle_embeddings[condition]['protein_ids']
            
            # Create gene-protein correspondence if mapping provided
            if gene_protein_mapping:
                network_genes = list(network_features.keys())
                aligned_genes, aligned_proteins, gene_features, protein_features_aligned = \
                    self._align_gene_protein_features(
                        network_genes, network_features,
                        protein_ids, protein_features,
                        gene_protein_mapping
                    )
            else:
                # Use all available features
                aligned_genes = list(network_features.keys())
                gene_features = np.array([network_features[gene] for gene in aligned_genes])
                aligned_proteins = protein_ids
                protein_features_aligned = protein_features
            
            # Compute transport plan
            transport_plan = self._compute_transport_plan(
                gene_features, protein_features_aligned,
                method='fused_gromov_wasserstein'
            )
            
            # Compute alignment quality metrics
            alignment_quality = self._compute_alignment_quality(
                gene_features, protein_features_aligned, transport_plan
            )
            
            alignment_results[condition] = {
                'transport_plan': transport_plan,
                'aligned_genes': aligned_genes,
                'aligned_proteins': aligned_proteins,
                'gene_features': gene_features,
                'protein_features': protein_features_aligned,
                'alignment_quality': alignment_quality,
                'wasserstein_distance': self._compute_wasserstein_distance(
                    gene_features, protein_features_aligned
                )
            }
            
            self.logger.info(f"Alignment completed for {condition}: "
                           f"W-distance = {alignment_results[condition]['wasserstein_distance']:.4f}")
        
        self.alignment_results = alignment_results
        return alignment_results
    
    def compare_conditions(self,
                          scenic_networks: Dict[str, nx.Graph],
                          pinnacle_embeddings: Dict[str, Dict],
                          condition_pairs: Optional[List[Tuple[str, str]]] = None) -> Dict[str, Dict]:
        """
        Compare network states across conditions using optimal transport.
        
        Args:
            scenic_networks: Dictionary of regulatory networks by condition
            pinnacle_embeddings: Dictionary of protein embeddings by condition
            condition_pairs: List of condition pairs to compare
            
        Returns:
            Dictionary of comparison results
        """
        self.logger.info("Comparing conditions with optimal transport...")
        
        if condition_pairs is None:
            conditions = list(scenic_networks.keys())
            condition_pairs = [(conditions[i], conditions[j]) 
                             for i in range(len(conditions)) 
                             for j in range(i+1, len(conditions))]
        
        comparison_results = {}
        
        for cond1, cond2 in condition_pairs:
            self.logger.info(f"Comparing {cond1} vs {cond2}")
            
            # Compare regulatory networks
            network_comparison = self._compare_networks_ot(
                scenic_networks[cond1], scenic_networks[cond2]
            )
            
            # Compare protein embeddings
            protein_comparison = self._compare_embeddings_ot(
                pinnacle_embeddings[cond1], pinnacle_embeddings[cond2]
            )
            
            # Compute joint comparison
            joint_comparison = self._compute_joint_comparison(
                network_comparison, protein_comparison
            )
            
            comparison_key = f"{cond1}_vs_{cond2}"
            comparison_results[comparison_key] = {
                'network_comparison': network_comparison,
                'protein_comparison': protein_comparison,
                'joint_comparison': joint_comparison,
                'conditions': (cond1, cond2)
            }
        
        return comparison_results
    
    def trajectory_analysis(self,
                          scenic_networks: Dict[str, nx.Graph],
                          pinnacle_embeddings: Dict[str, Dict],
                          condition_order: List[str],
                          time_points: Optional[List[float]] = None) -> Dict:
        """
        Analyze network evolution trajectory using optimal transport.
        
        Args:
            scenic_networks: Dictionary of regulatory networks by condition
            pinnacle_embeddings: Dictionary of protein embeddings by condition
            condition_order: Ordered list of conditions (e.g., time points)
            time_points: Optional actual time values
            
        Returns:
            Trajectory analysis results
        """
        self.logger.info("Performing trajectory analysis with optimal transport...")
        
        if time_points is None:
            time_points = list(range(len(condition_order)))
        
        # Extract features for all conditions
        network_features_by_condition = {}
        protein_features_by_condition = {}
        
        for condition in condition_order:
            network_features_by_condition[condition] = self._extract_network_features(
                scenic_networks[condition]
            )
            protein_features_by_condition[condition] = pinnacle_embeddings[condition]
        
        # Compute pairwise transport costs along trajectory
        trajectory_costs = []
        transport_plans_trajectory = []
        
        for i in range(len(condition_order) - 1):
            cond1, cond2 = condition_order[i], condition_order[i + 1]
            
            # Network trajectory
            net_features1 = np.array(list(network_features_by_condition[cond1].values()))
            net_features2 = np.array(list(network_features_by_condition[cond2].values()))
            
            # Protein trajectory
            prot_features1 = protein_features_by_condition[cond1]['embeddings']
            prot_features2 = protein_features_by_condition[cond2]['embeddings']
            
            # Compute joint transport
            joint_cost, joint_plan = self._compute_joint_trajectory_step(
                net_features1, net_features2,
                prot_features1, prot_features2
            )
            
            trajectory_costs.append(joint_cost)
            transport_plans_trajectory.append(joint_plan)
        
        # Analyze trajectory properties
        trajectory_analysis = self._analyze_trajectory_properties(
            trajectory_costs, time_points[:-1], condition_order
        )
        
        return {
            'trajectory_costs': trajectory_costs,
            'transport_plans': transport_plans_trajectory,
            'trajectory_analysis': trajectory_analysis,
            'condition_order': condition_order,
            'time_points': time_points
        }
    
    def batch_correction(self,
                        data_by_batch: Dict[str, np.ndarray],
                        reference_batch: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Perform batch correction using optimal transport.
        
        Args:
            data_by_batch: Dictionary of data matrices by batch
            reference_batch: Reference batch for alignment (if None, use first batch)
            
        Returns:
            Dictionary of batch-corrected data
        """
        self.logger.info("Performing batch correction with optimal transport...")
        
        if reference_batch is None:
            reference_batch = list(data_by_batch.keys())[0]
        
        reference_data = data_by_batch[reference_batch]
        corrected_data = {reference_batch: reference_data.copy()}
        
        for batch_name, batch_data in data_by_batch.items():
            if batch_name == reference_batch:
                continue
            
            self.logger.info(f"Correcting batch: {batch_name}")
            
            # Compute transport plan
            transport_plan = self._compute_transport_plan(
                batch_data, reference_data,
                method='unbalanced_sinkhorn'
            )
            
            # Apply transport to correct batch effects
            corrected_batch = self._apply_transport_correction(
                batch_data, reference_data, transport_plan
            )
            
            corrected_data[batch_name] = corrected_batch
        
        return corrected_data
    
    def _extract_network_features(self, network: nx.Graph) -> Dict[str, np.ndarray]:
        """Extract feature vectors from network nodes."""
        features = {}
        
        for node in network.nodes():
            # Compute node features
            node_features = []
            
            # Degree centrality
            node_features.append(network.degree(node))
            
            # Betweenness centrality
            try:
                betweenness = nx.betweenness_centrality(network)
                node_features.append(betweenness.get(node, 0))
            except:
                node_features.append(0)
            
            # Closeness centrality
            try:
                closeness = nx.closeness_centrality(network)
                node_features.append(closeness.get(node, 0))
            except:
                node_features.append(0)
            
            # Clustering coefficient
            node_features.append(nx.clustering(network, node))
            
            # PageRank
            try:
                pagerank = nx.pagerank(network)
                node_features.append(pagerank.get(node, 0))
            except:
                node_features.append(0)
            
            # Edge weights (if available)
            edge_weights = []
            for neighbor in network.neighbors(node):
                edge_data = network.get_edge_data(node, neighbor)
                if edge_data and 'importance' in edge_data:
                    edge_weights.append(edge_data['importance'])
            
            if edge_weights:
                node_features.extend([
                    np.mean(edge_weights),
                    np.std(edge_weights),
                    np.max(edge_weights)
                ])
            else:
                node_features.extend([0, 0, 0])
            
            features[node] = np.array(node_features)
        
        return features
    
    def _align_gene_protein_features(self,
                                   network_genes: List[str],
                                   network_features: Dict[str, np.ndarray],
                                   protein_ids: List[str],
                                   protein_features: np.ndarray,
                                   gene_protein_mapping: Dict[str, str]) -> Tuple:
        """Align gene and protein features using provided mapping."""
        aligned_genes = []
        aligned_proteins = []
        gene_features_aligned = []
        protein_features_aligned = []
        
        for gene in network_genes:
            if gene in gene_protein_mapping:
                protein = gene_protein_mapping[gene]
                if protein in protein_ids:
                    protein_idx = protein_ids.index(protein)
                    
                    aligned_genes.append(gene)
                    aligned_proteins.append(protein)
                    gene_features_aligned.append(network_features[gene])
                    protein_features_aligned.append(protein_features[protein_idx])
        
        return (aligned_genes, aligned_proteins,
                np.array(gene_features_aligned), np.array(protein_features_aligned))
    
    def _compute_transport_plan(self,
                              source_features: np.ndarray,
                              target_features: np.ndarray,
                              method: str = 'sinkhorn') -> np.ndarray:
        """Compute optimal transport plan between feature sets."""
        
        # Normalize features if requested
        if self.config['normalize']:
            scaler_source = StandardScaler()
            scaler_target = StandardScaler()
            source_features = scaler_source.fit_transform(source_features)
            target_features = scaler_target.fit_transform(target_features)
        
        # Compute cost matrix
        cost_matrix = pairwise_distances(
            source_features, target_features,
            metric=self.config['metric']
        )
        
        # Create uniform distributions
        a = np.ones(source_features.shape[0]) / source_features.shape[0]
        b = np.ones(target_features.shape[0]) / target_features.shape[0]
        
        # Compute transport plan based on method
        if method == 'sinkhorn':
            transport_plan = ot.sinkhorn(
                a, b, cost_matrix,
                reg=self.config['reg'],
                numItermax=self.config['max_iter'],
                stopThr=self.config['tol']
            )
        
        elif method == 'unbalanced_sinkhorn':
            transport_plan = sinkhorn_unbalanced(
                a, b, cost_matrix,
                reg=self.config['reg'],
                reg_m=self.config['reg_m'],
                numItermax=self.config['max_iter'],
                stopThr=self.config['tol']
            )
        
        elif method == 'emd':
            transport_plan = ot.emd(a, b, cost_matrix)
        
        elif method == 'gromov_wasserstein':
            # Compute structure matrices
            C1 = pairwise_distances(source_features, metric=self.config['metric'])
            C2 = pairwise_distances(target_features, metric=self.config['metric'])
            
            transport_plan = gromov_wasserstein(
                C1, C2, a, b,
                loss_fun='square_loss',
                epsilon=self.config['reg'],
                max_iter=self.config['max_iter']
            )
        
        elif method == 'fused_gromov_wasserstein':
            # Compute structure matrices
            C1 = pairwise_distances(source_features, metric=self.config['metric'])
            C2 = pairwise_distances(target_features, metric=self.config['metric'])
            
            transport_plan = fused_gromov_wasserstein(
                cost_matrix, C1, C2, a, b,
                loss_fun='square_loss',
                alpha=self.config['alpha'],
                epsilon=self.config['reg'],
                max_iter=self.config['max_iter']
            )
        
        else:
            raise ValueError(f"Unknown transport method: {method}")
        
        return transport_plan
    
    def _compute_alignment_quality(self,
                                 source_features: np.ndarray,
                                 target_features: np.ndarray,
                                 transport_plan: np.ndarray) -> Dict:
        """Compute quality metrics for the alignment."""
        
        # Transport cost
        cost_matrix = pairwise_distances(source_features, target_features)
        transport_cost = np.sum(transport_plan * cost_matrix)
        
        # Entropy of transport plan
        transport_entropy = entropy(transport_plan.flatten() + 1e-10)
        
        # Sparsity of transport plan
        sparsity = np.sum(transport_plan > 1e-6) / transport_plan.size
        
        # Marginal preservation
        source_marginal_error = np.linalg.norm(
            np.sum(transport_plan, axis=1) - np.ones(source_features.shape[0]) / source_features.shape[0]
        )
        target_marginal_error = np.linalg.norm(
            np.sum(transport_plan, axis=0) - np.ones(target_features.shape[0]) / target_features.shape[0]
        )
        
        return {
            'transport_cost': transport_cost,
            'transport_entropy': transport_entropy,
            'sparsity': sparsity,
            'source_marginal_error': source_marginal_error,
            'target_marginal_error': target_marginal_error
        }
    
    def _compute_wasserstein_distance(self,
                                    source_features: np.ndarray,
                                    target_features: np.ndarray) -> float:
        """Compute Wasserstein distance between feature distributions."""
        cost_matrix = pairwise_distances(source_features, target_features)
        
        a = np.ones(source_features.shape[0]) / source_features.shape[0]
        b = np.ones(target_features.shape[0]) / target_features.shape[0]
        
        return ot.emd2(a, b, cost_matrix)
    
    def _compare_networks_ot(self, network1: nx.Graph, network2: nx.Graph) -> Dict:
        """Compare two networks using optimal transport."""
        
        # Extract features
        features1 = self._extract_network_features(network1)
        features2 = self._extract_network_features(network2)
        
        # Find common nodes
        common_nodes = set(features1.keys()) & set(features2.keys())
        
        if len(common_nodes) == 0:
            return {'wasserstein_distance': np.inf, 'common_nodes': 0}
        
        # Align features
        aligned_features1 = np.array([features1[node] for node in common_nodes])
        aligned_features2 = np.array([features2[node] for node in common_nodes])
        
        # Compute Wasserstein distance
        w_distance = self._compute_wasserstein_distance(aligned_features1, aligned_features2)
        
        # Compute transport plan
        transport_plan = self._compute_transport_plan(aligned_features1, aligned_features2)
        
        return {
            'wasserstein_distance': w_distance,
            'transport_plan': transport_plan,
            'common_nodes': len(common_nodes),
            'aligned_features1': aligned_features1,
            'aligned_features2': aligned_features2
        }
    
    def _compare_embeddings_ot(self, embeddings1: Dict, embeddings2: Dict) -> Dict:
        """Compare two protein embedding sets using optimal transport."""
        
        # Find common proteins
        proteins1 = set(embeddings1['protein_ids'])
        proteins2 = set(embeddings2['protein_ids'])
        common_proteins = proteins1 & proteins2
        
        if len(common_proteins) == 0:
            return {'wasserstein_distance': np.inf, 'common_proteins': 0}
        
        # Align embeddings
        indices1 = [embeddings1['protein_ids'].index(p) for p in common_proteins]
        indices2 = [embeddings2['protein_ids'].index(p) for p in common_proteins]
        
        aligned_embeddings1 = embeddings1['embeddings'][indices1]
        aligned_embeddings2 = embeddings2['embeddings'][indices2]
        
        # Compute Wasserstein distance
        w_distance = self._compute_wasserstein_distance(aligned_embeddings1, aligned_embeddings2)
        
        # Compute transport plan
        transport_plan = self._compute_transport_plan(aligned_embeddings1, aligned_embeddings2)
        
        return {
            'wasserstein_distance': w_distance,
            'transport_plan': transport_plan,
            'common_proteins': len(common_proteins),
            'aligned_embeddings1': aligned_embeddings1,
            'aligned_embeddings2': aligned_embeddings2
        }
    
    def _compute_joint_comparison(self, network_comp: Dict, protein_comp: Dict) -> Dict:
        """Compute joint comparison metrics."""
        
        # Weighted combination of distances
        network_weight = 0.5
        protein_weight = 0.5
        
        joint_distance = (network_weight * network_comp['wasserstein_distance'] +
                         protein_weight * protein_comp['wasserstein_distance'])
        
        return {
            'joint_wasserstein_distance': joint_distance,
            'network_weight': network_weight,
            'protein_weight': protein_weight,
            'network_contribution': network_comp['wasserstein_distance'],
            'protein_contribution': protein_comp['wasserstein_distance']
        }
    
    def _compute_joint_trajectory_step(self,
                                     net_features1: np.ndarray,
                                     net_features2: np.ndarray,
                                     prot_features1: np.ndarray,
                                     prot_features2: np.ndarray) -> Tuple[float, Dict]:
        """Compute joint transport for one trajectory step."""
        
        # Compute individual transport costs
        net_cost = self._compute_wasserstein_distance(net_features1, net_features2)
        prot_cost = self._compute_wasserstein_distance(prot_features1, prot_features2)
        
        # Compute joint cost (weighted combination)
        joint_cost = 0.5 * net_cost + 0.5 * prot_cost
        
        # Compute joint transport plan (simplified)
        net_plan = self._compute_transport_plan(net_features1, net_features2)
        prot_plan = self._compute_transport_plan(prot_features1, prot_features2)
        
        joint_plan = {
            'network_plan': net_plan,
            'protein_plan': prot_plan,
            'network_cost': net_cost,
            'protein_cost': prot_cost
        }
        
        return joint_cost, joint_plan
    
    def _analyze_trajectory_properties(self,
                                     trajectory_costs: List[float],
                                     time_points: List[float],
                                     condition_order: List[str]) -> Dict:
        """Analyze properties of the computed trajectory."""
        
        # Compute trajectory velocity (rate of change)
        velocities = []
        for i in range(len(trajectory_costs) - 1):
            dt = time_points[i + 1] - time_points[i] if len(time_points) > i + 1 else 1
            velocity = (trajectory_costs[i + 1] - trajectory_costs[i]) / dt
            velocities.append(velocity)
        
        # Find critical points (local minima/maxima)
        critical_points = []
        for i in range(1, len(trajectory_costs) - 1):
            if ((trajectory_costs[i] > trajectory_costs[i-1] and 
                 trajectory_costs[i] > trajectory_costs[i+1]) or
                (trajectory_costs[i] < trajectory_costs[i-1] and 
                 trajectory_costs[i] < trajectory_costs[i+1])):
                critical_points.append(i)
        
        return {
            'total_cost': sum(trajectory_costs),
            'mean_cost': np.mean(trajectory_costs),
            'cost_variance': np.var(trajectory_costs),
            'velocities': velocities,
            'mean_velocity': np.mean(velocities) if velocities else 0,
            'critical_points': critical_points,
            'critical_conditions': [condition_order[i] for i in critical_points],
            'monotonic': all(v >= 0 for v in velocities) or all(v <= 0 for v in velocities)
        }
    
    def _apply_transport_correction(self,
                                  batch_data: np.ndarray,
                                  reference_data: np.ndarray,
                                  transport_plan: np.ndarray) -> np.ndarray:
        """Apply transport plan to correct batch effects."""
        
        # Transport batch data to reference distribution
        corrected_data = np.zeros_like(batch_data)
        
        for i in range(batch_data.shape[0]):
            # Find the most likely correspondences in reference
            transport_weights = transport_plan[i, :]
            
            # Weighted combination of reference points
            corrected_data[i] = np.average(reference_data, axis=0, weights=transport_weights)
        
        return corrected_data
    
    def get_transport_summary(self) -> Dict:
        """Get summary of all computed transport plans."""
        summary = {
            'n_alignments': len(self.alignment_results),
            'conditions_aligned': list(self.alignment_results.keys()),
            'transport_plans_computed': len(self.transport_plans),
            'distance_matrices_computed': len(self.distance_matrices)
        }
        
        if self.alignment_results:
            # Compute summary statistics
            w_distances = [result['wasserstein_distance'] 
                          for result in self.alignment_results.values()]
            summary.update({
                'mean_wasserstein_distance': np.mean(w_distances),
                'std_wasserstein_distance': np.std(w_distances),
                'min_wasserstein_distance': np.min(w_distances),
                'max_wasserstein_distance': np.max(w_distances)
            })
        
        return summary
    
    def visualize_transport_plan(self,
                               transport_plan: np.ndarray,
                               source_labels: Optional[List[str]] = None,
                               target_labels: Optional[List[str]] = None) -> Any:
        """Visualize transport plan as heatmap."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(transport_plan, 
                       xticklabels=target_labels[:20] if target_labels else False,
                       yticklabels=source_labels[:20] if source_labels else False,
                       cmap='viridis')
            plt.title('Optimal Transport Plan')
            plt.xlabel('Target Features')
            plt.ylabel('Source Features')
            
            return plt.gcf()
            
        except ImportError:
            self.logger.warning("Matplotlib/seaborn not available for visualization")
            return None


class OTEnhancedBridge:
    """
    BRIDGE integrator enhanced with Optimal Transport capabilities.
    
    Combines OT-based alignment and comparison with SCENIC+ and PINNACLE
    integration for improved multi-omics network analysis.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize OT-enhanced BRIDGE."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__ + '.OTEnhancedBridge')
        
        # Initialize OT integrator
        ot_config = self.config.get('optimal_transport', {})
        self.ot_integrator = OptimalTransportIntegrator(config=ot_config)
        
        # Will be initialized when needed
        self.bridge_integrator = None
        self.ot_results = {}
    
    def integrate_with_ot_alignment(self,
                                  scenic_networks: Dict[str, nx.Graph],
                                  pinnacle_embeddings: Dict[str, Dict],
                                  gene_protein_mapping: Optional[Dict[str, str]] = None) -> Dict:
        """
        Perform BRIDGE integration with OT-based cross-modal alignment.
        
        Args:
            scenic_networks: Dictionary of regulatory networks by condition
            pinnacle_embeddings: Dictionary of protein embeddings by condition
            gene_protein_mapping: Optional gene-to-protein mapping
            
        Returns:
            Complete integration results with OT alignment
        """
        self.logger.info("Starting OT-enhanced BRIDGE integration...")
        
        # Step 1: Cross-modal alignment with OT
        alignment_results = self.ot_integrator.align_cross_modal(
            scenic_networks, pinnacle_embeddings, gene_protein_mapping
        )
        
        # Step 2: Initialize BRIDGE integrator
        from .core import BridgeIntegrator
        
        bridge_config = self.config.get('bridge', {})
        self.bridge_integrator = BridgeIntegrator(config=bridge_config)
        
        # Step 3: Load data with OT alignment information
        self.bridge_integrator.scenic_networks = scenic_networks
        self.bridge_integrator.pinnacle_embeddings = pinnacle_embeddings
        
        # Add OT alignment information
        for condition in alignment_results:
            if hasattr(self.bridge_integrator, 'alignment_info'):
                self.bridge_integrator.alignment_info[condition] = alignment_results[condition]
            else:
                self.bridge_integrator.alignment_info = {condition: alignment_results[condition]}
        
        # Step 4: Perform integration with OT guidance
        integrated_networks = self.bridge_integrator.integrate_networks()
        
        # Step 5: OT-based condition comparison
        condition_comparisons = self.ot_integrator.compare_conditions(
            scenic_networks, pinnacle_embeddings
        )
        
        # Store results
        self.ot_results = {
            'alignment_results': alignment_results,
            'condition_comparisons': condition_comparisons,
            'integrated_networks': integrated_networks
        }
        
        return {
            'integrated_networks': integrated_networks,
            'ot_alignment_results': alignment_results,
            'ot_condition_comparisons': condition_comparisons,
            'ot_summary': self.ot_integrator.get_transport_summary(),
            'bridge_integrator': self.bridge_integrator
        }
    
    def trajectory_analysis_with_ot(self,
                                  scenic_networks: Dict[str, nx.Graph],
                                  pinnacle_embeddings: Dict[str, Dict],
                                  condition_order: List[str],
                                  time_points: Optional[List[float]] = None) -> Dict:
        """
        Perform trajectory analysis using optimal transport.
        
        Args:
            scenic_networks: Dictionary of regulatory networks by condition
            pinnacle_embeddings: Dictionary of protein embeddings by condition
            condition_order: Ordered list of conditions
            time_points: Optional time values
            
        Returns:
            Trajectory analysis results
        """
        self.logger.info("Performing OT-based trajectory analysis...")
        
        trajectory_results = self.ot_integrator.trajectory_analysis(
            scenic_networks, pinnacle_embeddings, condition_order, time_points
        )
        
        return trajectory_results

