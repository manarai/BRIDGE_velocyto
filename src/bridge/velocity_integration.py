"""
Velocyto Integration Module for BRIDGE Framework

This module integrates RNA velocity analysis with BRIDGE to enable dynamic
network perturbation analysis for drug response and pharmacology studies.

Key Features:
- RNA velocity-guided network dynamics analysis
- Drug response trajectory mapping
- Perturbed pathway identification
- Pharmacological network profiling
- Temporal regulatory network evolution
- Drug target discovery through velocity-network integration
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import networkx as nx
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from pathlib import Path
import warnings
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Velocyto imports with error handling
try:
    import scvelo as scv
    import velocyto as vcy
    VELOCYTO_AVAILABLE = True
except ImportError:
    VELOCYTO_AVAILABLE = False
    warnings.warn("scVelo/velocyto not available. Install with: pip install scvelo velocyto")

# CellRank for trajectory analysis
try:
    import cellrank as cr
    from cellrank.kernels import VelocityKernel, ConnectivityKernel
    from cellrank.estimators import GPCCA, CFLARE
    CELLRANK_AVAILABLE = True
except ImportError:
    CELLRANK_AVAILABLE = False
    warnings.warn("CellRank not available. Install with: pip install cellrank")


class VelocityNetworkAnalyzer:
    """
    RNA velocity-based network dynamics analyzer for drug response studies.
    
    Integrates RNA velocity with BRIDGE network analysis to track how
    regulatory networks evolve in response to drug perturbations.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize velocity network analyzer.
        
        Args:
            config: Configuration parameters for velocity analysis
        """
        if not VELOCYTO_AVAILABLE:
            raise ImportError("scVelo is required but not installed. Install with: pip install scvelo")
        
        self.config = config or {}
        self.logger = logging.getLogger(__name__ + '.VelocityNetworkAnalyzer')
        
        # Default configuration
        self.default_config = {
            'velocity': {
                'mode': 'dynamical',  # 'stochastic', 'dynamical', 'deterministic'
                'n_top_genes': 2000,
                'min_shared_counts': 20,
                'n_pcs': 30,
                'n_neighbors': 30,
                'min_likelihood': None,
                'copy': False
            },
            'network_dynamics': {
                'time_resolution': 10,  # Number of time points for trajectory
                'velocity_threshold': 0.1,
                'correlation_method': 'pearson',
                'min_correlation': 0.3,
                'network_stability_window': 5
            },
            'drug_response': {
                'response_genes': None,  # List of drug response genes
                'pathway_databases': ['kegg', 'reactome', 'go'],
                'enrichment_method': 'hypergeometric',
                'fdr_threshold': 0.05,
                'effect_size_threshold': 0.5
            },
            'perturbation': {
                'control_condition': 'control',
                'treatment_conditions': None,
                'time_points': None,
                'dose_response': False
            }
        }
        
        # Update with user config
        self.config = {**self.default_config, **self.config}
        
        # Storage for analysis results
        self.velocity_data = {}
        self.dynamic_networks = {}
        self.perturbation_results = {}
        self.drug_response_profiles = {}
        
    def compute_rna_velocity(self,
                           adata: anndata.AnnData,
                           condition_key: str = 'condition',
                           time_key: Optional[str] = None) -> anndata.AnnData:
        """
        Compute RNA velocity for drug response analysis.
        
        Args:
            adata: AnnData object with spliced/unspliced counts
            condition_key: Column indicating experimental conditions
            time_key: Optional column indicating time points
            
        Returns:
            AnnData object with computed velocities
        """
        self.logger.info("Computing RNA velocity for drug response analysis...")
        
        # Ensure required layers exist
        if 'spliced' not in adata.layers or 'unspliced' not in adata.layers:
            raise ValueError("AnnData must contain 'spliced' and 'unspliced' layers")
        
        # Basic preprocessing
        scv.pp.filter_and_normalize(adata, 
                                   min_shared_counts=self.config['velocity']['min_shared_counts'],
                                   n_top_genes=self.config['velocity']['n_top_genes'])
        
        # Compute moments
        scv.pp.moments(adata, 
                      n_pcs=self.config['velocity']['n_pcs'],
                      n_neighbors=self.config['velocity']['n_neighbors'])
        
        # Compute velocity based on mode
        velocity_mode = self.config['velocity']['mode']
        
        if velocity_mode == 'stochastic':
            scv.tl.velocity(adata, mode='stochastic')
        elif velocity_mode == 'dynamical':
            scv.tl.recover_dynamics(adata)
            scv.tl.velocity(adata, mode='dynamical')
        elif velocity_mode == 'deterministic':
            scv.tl.velocity(adata, mode='deterministic')
        else:
            raise ValueError(f"Unknown velocity mode: {velocity_mode}")
        
        # Compute velocity graph
        scv.tl.velocity_graph(adata)
        
        # Compute velocity embedding
        scv.tl.velocity_embedding(adata, basis='umap')
        
        # Store velocity data
        self.velocity_data[condition_key] = adata
        
        self.logger.info(f"RNA velocity computed for {adata.n_obs} cells, {adata.n_vars} genes")
        return adata
    
    def analyze_network_dynamics(self,
                                scenic_networks: Dict[str, nx.Graph],
                                velocity_adata: anndata.AnnData,
                                condition_key: str = 'condition',
                                time_key: Optional[str] = None) -> Dict[str, Dict]:
        """
        Analyze how regulatory networks change along velocity trajectories.
        
        Args:
            scenic_networks: Dictionary of SCENIC+ networks by condition
            velocity_adata: AnnData with computed velocities
            condition_key: Column indicating conditions
            time_key: Optional time point column
            
        Returns:
            Dictionary of network dynamics results
        """
        self.logger.info("Analyzing network dynamics with RNA velocity...")
        
        dynamics_results = {}
        
        # Get velocity vectors
        velocity_vectors = velocity_adata.layers['velocity']
        
        for condition in scenic_networks.keys():
            if condition not in velocity_adata.obs[condition_key].values:
                self.logger.warning(f"Condition {condition} not found in velocity data")
                continue
            
            self.logger.info(f"Analyzing dynamics for condition: {condition}")
            
            # Get cells for this condition
            condition_mask = velocity_adata.obs[condition_key] == condition
            condition_cells = velocity_adata.obs_names[condition_mask]
            condition_velocities = velocity_vectors[condition_mask]
            
            # Get network for this condition
            network = scenic_networks[condition]
            
            # Compute velocity-network correlations
            velocity_network_corr = self._compute_velocity_network_correlations(
                network, velocity_adata[condition_mask], condition_velocities
            )
            
            # Identify velocity-responsive regulatory modules
            responsive_modules = self._identify_velocity_responsive_modules(
                network, velocity_network_corr
            )
            
            # Compute network stability along trajectory
            network_stability = self._compute_network_stability_trajectory(
                network, velocity_adata[condition_mask], condition_velocities
            )
            
            # Predict future network states
            future_networks = self._predict_future_network_states(
                network, velocity_adata[condition_mask], condition_velocities
            )
            
            dynamics_results[condition] = {
                'velocity_network_correlations': velocity_network_corr,
                'responsive_modules': responsive_modules,
                'network_stability': network_stability,
                'future_networks': future_networks,
                'n_cells': len(condition_cells),
                'n_genes_with_velocity': np.sum(np.any(condition_velocities != 0, axis=0))
            }
        
        self.dynamic_networks = dynamics_results
        return dynamics_results
    
    def identify_drug_response_networks(self,
                                      scenic_networks: Dict[str, nx.Graph],
                                      pinnacle_embeddings: Dict[str, Dict],
                                      velocity_adata: anndata.AnnData,
                                      drug_conditions: List[str],
                                      control_condition: str = 'control') -> Dict[str, Dict]:
        """
        Identify regulatory networks perturbed by drug treatment.
        
        Args:
            scenic_networks: Dictionary of SCENIC+ networks
            pinnacle_embeddings: Dictionary of protein embeddings
            velocity_adata: AnnData with velocity information
            drug_conditions: List of drug treatment conditions
            control_condition: Control condition name
            
        Returns:
            Dictionary of drug response network analysis results
        """
        self.logger.info("Identifying drug response networks...")
        
        drug_response_results = {}
        
        for drug_condition in drug_conditions:
            self.logger.info(f"Analyzing drug response for: {drug_condition}")
            
            # Compare networks: drug vs control
            if control_condition not in scenic_networks or drug_condition not in scenic_networks:
                self.logger.warning(f"Missing networks for {drug_condition} vs {control_condition}")
                continue
            
            control_network = scenic_networks[control_condition]
            drug_network = scenic_networks[drug_condition]
            
            # Identify perturbed regulatory modules
            perturbed_modules = self._identify_perturbed_modules(
                control_network, drug_network, velocity_adata, drug_condition
            )
            
            # Analyze protein network changes
            protein_changes = self._analyze_protein_network_changes(
                pinnacle_embeddings.get(control_condition),
                pinnacle_embeddings.get(drug_condition),
                velocity_adata, drug_condition
            )
            
            # Compute drug response trajectories
            response_trajectories = self._compute_drug_response_trajectories(
                velocity_adata, control_condition, drug_condition
            )
            
            # Identify drug targets
            drug_targets = self._identify_drug_targets(
                perturbed_modules, protein_changes, response_trajectories
            )
            
            # Compute pharmacological profile
            pharma_profile = self._compute_pharmacological_profile(
                control_network, drug_network, velocity_adata, drug_condition
            )
            
            drug_response_results[drug_condition] = {
                'perturbed_modules': perturbed_modules,
                'protein_changes': protein_changes,
                'response_trajectories': response_trajectories,
                'drug_targets': drug_targets,
                'pharmacological_profile': pharma_profile,
                'network_perturbation_score': self._compute_network_perturbation_score(
                    control_network, drug_network
                )
            }
        
        self.drug_response_profiles = drug_response_results
        return drug_response_results
    
    def analyze_dose_response_dynamics(self,
                                     scenic_networks: Dict[str, nx.Graph],
                                     velocity_adata: anndata.AnnData,
                                     dose_conditions: List[str],
                                     doses: List[float],
                                     control_condition: str = 'control') -> Dict[str, Any]:
        """
        Analyze dose-response relationships in network dynamics.
        
        Args:
            scenic_networks: Dictionary of networks by condition
            velocity_adata: AnnData with velocity data
            dose_conditions: List of dose condition names
            doses: List of corresponding dose values
            control_condition: Control condition name
            
        Returns:
            Dose-response analysis results
        """
        self.logger.info("Analyzing dose-response network dynamics...")
        
        if len(dose_conditions) != len(doses):
            raise ValueError("Number of dose conditions must match number of doses")
        
        # Sort by dose
        dose_data = sorted(zip(doses, dose_conditions))
        sorted_doses, sorted_conditions = zip(*dose_data)
        
        dose_response_results = {
            'doses': list(sorted_doses),
            'conditions': list(sorted_conditions),
            'network_metrics': {},
            'velocity_metrics': {},
            'dose_response_curves': {},
            'ec50_estimates': {},
            'hill_coefficients': {}
        }
        
        # Compute network metrics for each dose
        for dose, condition in dose_data:
            if condition not in scenic_networks:
                continue
            
            network = scenic_networks[condition]
            
            # Basic network metrics
            network_metrics = {
                'n_nodes': network.number_of_nodes(),
                'n_edges': network.number_of_edges(),
                'density': nx.density(network),
                'avg_clustering': nx.average_clustering(network),
                'avg_path_length': self._safe_average_shortest_path_length(network)
            }
            
            # Velocity-based metrics
            condition_mask = velocity_adata.obs['condition'] == condition
            if np.any(condition_mask):
                velocity_metrics = self._compute_velocity_metrics(
                    velocity_adata[condition_mask]
                )
            else:
                velocity_metrics = {}
            
            dose_response_results['network_metrics'][dose] = network_metrics
            dose_response_results['velocity_metrics'][dose] = velocity_metrics
        
        # Fit dose-response curves
        dose_response_results['dose_response_curves'] = self._fit_dose_response_curves(
            sorted_doses, dose_response_results['network_metrics']
        )
        
        return dose_response_results
    
    def identify_pharmacological_targets(self,
                                       drug_response_results: Dict[str, Dict],
                                       target_databases: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Identify potential pharmacological targets from network perturbation analysis.
        
        Args:
            drug_response_results: Results from drug response analysis
            target_databases: List of target databases to query
            
        Returns:
            Dictionary of identified pharmacological targets
        """
        self.logger.info("Identifying pharmacological targets...")
        
        target_databases = target_databases or ['drugbank', 'chembl', 'stitch']
        
        pharmacological_targets = {}
        
        for drug_condition, results in drug_response_results.items():
            self.logger.info(f"Analyzing targets for {drug_condition}")
            
            # Extract key perturbed genes/proteins
            perturbed_modules = results['perturbed_modules']
            drug_targets = results['drug_targets']
            
            # Prioritize targets based on multiple criteria
            target_scores = self._compute_target_scores(
                perturbed_modules, drug_targets, results
            )
            
            # Annotate with known drug-target interactions
            target_annotations = self._annotate_drug_targets(
                target_scores, target_databases
            )
            
            # Predict novel targets
            novel_targets = self._predict_novel_targets(
                target_scores, target_annotations
            )
            
            pharmacological_targets[drug_condition] = {
                'target_scores': target_scores,
                'known_targets': target_annotations,
                'novel_targets': novel_targets,
                'top_targets': self._rank_top_targets(target_scores, n_top=20)
            }
        
        return pharmacological_targets
    
    def _compute_velocity_network_correlations(self,
                                             network: nx.Graph,
                                             velocity_adata: anndata.AnnData,
                                             velocity_vectors: np.ndarray) -> Dict[str, float]:
        """Compute correlations between velocity and network structure."""
        correlations = {}
        
        # Get network genes that are also in velocity data
        network_genes = set(network.nodes())
        velocity_genes = set(velocity_adata.var_names)
        common_genes = network_genes & velocity_genes
        
        if len(common_genes) == 0:
            return correlations
        
        # Compute network centrality measures
        centrality_measures = {
            'degree': nx.degree_centrality(network),
            'betweenness': nx.betweenness_centrality(network),
            'closeness': nx.closeness_centrality(network),
            'pagerank': nx.pagerank(network)
        }
        
        # Compute velocity magnitudes for common genes
        gene_indices = [velocity_adata.var_names.get_loc(gene) for gene in common_genes]
        velocity_magnitudes = np.linalg.norm(velocity_vectors[:, gene_indices], axis=0)
        
        # Correlate centrality with velocity
        for measure_name, centrality_dict in centrality_measures.items():
            centrality_values = [centrality_dict.get(gene, 0) for gene in common_genes]
            
            if len(centrality_values) > 1 and np.var(centrality_values) > 0:
                corr, pval = pearsonr(centrality_values, velocity_magnitudes)
                correlations[f'{measure_name}_velocity_corr'] = corr
                correlations[f'{measure_name}_velocity_pval'] = pval
        
        return correlations
    
    def _identify_velocity_responsive_modules(self,
                                            network: nx.Graph,
                                            velocity_correlations: Dict[str, float]) -> Dict[str, Any]:
        """Identify network modules that respond to velocity changes."""
        
        # Use community detection to find modules
        try:
            communities = nx.community.greedy_modularity_communities(network)
        except:
            # Fallback to simple connected components
            communities = list(nx.connected_components(network.to_undirected()))
        
        responsive_modules = {}
        
        for i, community in enumerate(communities):
            module_id = f"module_{i}"
            
            # Compute module properties
            subgraph = network.subgraph(community)
            
            module_info = {
                'genes': list(community),
                'size': len(community),
                'density': nx.density(subgraph),
                'avg_clustering': nx.average_clustering(subgraph),
                'velocity_responsiveness': self._compute_module_velocity_responsiveness(
                    community, velocity_correlations
                )
            }
            
            responsive_modules[module_id] = module_info
        
        return responsive_modules
    
    def _compute_network_stability_trajectory(self,
                                            network: nx.Graph,
                                            velocity_adata: anndata.AnnData,
                                            velocity_vectors: np.ndarray) -> Dict[str, Any]:
        """Compute network stability along velocity trajectory."""
        
        # Project cells onto velocity trajectory
        if CELLRANK_AVAILABLE:
            try:
                # Use CellRank for trajectory analysis
                vk = VelocityKernel(velocity_adata)
                vk.compute_transition_matrix()
                
                ck = ConnectivityKernel(velocity_adata)
                ck.compute_transition_matrix()
                
                # Combine kernels
                combined_kernel = 0.8 * vk + 0.2 * ck
                
                # Estimate terminal states
                g = GPCCA(combined_kernel)
                g.compute_schur(n_components=10)
                g.predict_terminal_states()
                
                # Compute trajectory stability
                stability_scores = self._compute_trajectory_stability(
                    network, velocity_adata, g
                )
                
                return {
                    'trajectory_stability': stability_scores,
                    'terminal_states': g.terminal_states,
                    'method': 'cellrank'
                }
                
            except Exception as e:
                self.logger.warning(f"CellRank analysis failed: {e}")
        
        # Fallback to simple velocity-based analysis
        velocity_magnitudes = np.linalg.norm(velocity_vectors, axis=1)
        
        return {
            'velocity_magnitudes': velocity_magnitudes,
            'mean_velocity': np.mean(velocity_magnitudes),
            'velocity_variance': np.var(velocity_magnitudes),
            'method': 'simple'
        }
    
    def _predict_future_network_states(self,
                                     network: nx.Graph,
                                     velocity_adata: anndata.AnnData,
                                     velocity_vectors: np.ndarray) -> Dict[str, Any]:
        """Predict future network states based on velocity."""
        
        # Simple prediction based on velocity direction
        time_steps = self.config['network_dynamics']['time_resolution']
        
        future_states = {}
        
        for t in range(1, time_steps + 1):
            # Predict gene expression at future time point
            current_expr = velocity_adata.X
            predicted_expr = current_expr + t * velocity_vectors
            
            # Compute predicted network properties
            predicted_properties = self._compute_network_properties_from_expression(
                network, predicted_expr, velocity_adata.var_names
            )
            
            future_states[f'time_{t}'] = predicted_properties
        
        return future_states
    
    def _identify_perturbed_modules(self,
                                  control_network: nx.Graph,
                                  drug_network: nx.Graph,
                                  velocity_adata: anndata.AnnData,
                                  drug_condition: str) -> Dict[str, Any]:
        """Identify regulatory modules perturbed by drug treatment."""
        
        # Find common nodes
        common_nodes = set(control_network.nodes()) & set(drug_network.nodes())
        
        if len(common_nodes) == 0:
            return {}
        
        # Compute network differences
        control_subgraph = control_network.subgraph(common_nodes)
        drug_subgraph = drug_network.subgraph(common_nodes)
        
        # Identify changed edges
        control_edges = set(control_subgraph.edges())
        drug_edges = set(drug_subgraph.edges())
        
        lost_edges = control_edges - drug_edges
        gained_edges = drug_edges - control_edges
        
        # Identify modules with significant changes
        perturbed_modules = {
            'lost_edges': list(lost_edges),
            'gained_edges': list(gained_edges),
            'edge_change_ratio': len(gained_edges | lost_edges) / max(len(control_edges), 1),
            'nodes_with_changes': list(set([node for edge in (lost_edges | gained_edges) 
                                          for node in edge]))
        }
        
        # Add velocity information for perturbed nodes
        drug_mask = velocity_adata.obs['condition'] == drug_condition
        if np.any(drug_mask):
            perturbed_nodes = perturbed_modules['nodes_with_changes']
            velocity_info = self._get_velocity_info_for_genes(
                velocity_adata[drug_mask], perturbed_nodes
            )
            perturbed_modules['velocity_info'] = velocity_info
        
        return perturbed_modules
    
    def _analyze_protein_network_changes(self,
                                       control_embeddings: Optional[Dict],
                                       drug_embeddings: Optional[Dict],
                                       velocity_adata: anndata.AnnData,
                                       drug_condition: str) -> Dict[str, Any]:
        """Analyze changes in protein networks due to drug treatment."""
        
        if not control_embeddings or not drug_embeddings:
            return {}
        
        # Find common proteins
        control_proteins = set(control_embeddings['protein_ids'])
        drug_proteins = set(drug_embeddings['protein_ids'])
        common_proteins = control_proteins & drug_proteins
        
        if len(common_proteins) == 0:
            return {}
        
        # Get embeddings for common proteins
        control_indices = [control_embeddings['protein_ids'].index(p) for p in common_proteins]
        drug_indices = [drug_embeddings['protein_ids'].index(p) for p in common_proteins]
        
        control_emb = control_embeddings['embeddings'][control_indices]
        drug_emb = drug_embeddings['embeddings'][drug_indices]
        
        # Compute embedding changes
        embedding_changes = drug_emb - control_emb
        change_magnitudes = np.linalg.norm(embedding_changes, axis=1)
        
        # Identify most changed proteins
        most_changed_indices = np.argsort(change_magnitudes)[-20:]  # Top 20
        most_changed_proteins = [list(common_proteins)[i] for i in most_changed_indices]
        
        return {
            'common_proteins': list(common_proteins),
            'embedding_changes': embedding_changes,
            'change_magnitudes': change_magnitudes,
            'most_changed_proteins': most_changed_proteins,
            'mean_change_magnitude': np.mean(change_magnitudes),
            'max_change_magnitude': np.max(change_magnitudes)
        }
    
    def _compute_drug_response_trajectories(self,
                                          velocity_adata: anndata.AnnData,
                                          control_condition: str,
                                          drug_condition: str) -> Dict[str, Any]:
        """Compute drug response trajectories using velocity information."""
        
        # Get cells for each condition
        control_mask = velocity_adata.obs['condition'] == control_condition
        drug_mask = velocity_adata.obs['condition'] == drug_condition
        
        if not np.any(control_mask) or not np.any(drug_mask):
            return {}
        
        # Compute trajectory from control to drug state
        control_cells = velocity_adata[control_mask]
        drug_cells = velocity_adata[drug_mask]
        
        # Use velocity to predict trajectory
        if 'velocity_umap' in velocity_adata.obsm.keys():
            control_umap = control_cells.obsm['X_umap']
            drug_umap = drug_cells.obsm['X_umap']
            control_velocity_umap = control_cells.obsm['velocity_umap']
            
            # Compute trajectory metrics
            trajectory_metrics = {
                'control_centroid': np.mean(control_umap, axis=0),
                'drug_centroid': np.mean(drug_umap, axis=0),
                'trajectory_distance': np.linalg.norm(
                    np.mean(drug_umap, axis=0) - np.mean(control_umap, axis=0)
                ),
                'velocity_alignment': self._compute_velocity_alignment(
                    control_umap, drug_umap, control_velocity_umap
                )
            }
        else:
            trajectory_metrics = {}
        
        return trajectory_metrics
    
    def _identify_drug_targets(self,
                             perturbed_modules: Dict[str, Any],
                             protein_changes: Dict[str, Any],
                             response_trajectories: Dict[str, Any]) -> Dict[str, Any]:
        """Identify potential drug targets from perturbation analysis."""
        
        drug_targets = {
            'regulatory_targets': [],
            'protein_targets': [],
            'trajectory_targets': [],
            'combined_targets': []
        }
        
        # Regulatory targets from perturbed modules
        if 'nodes_with_changes' in perturbed_modules:
            drug_targets['regulatory_targets'] = perturbed_modules['nodes_with_changes']
        
        # Protein targets from embedding changes
        if 'most_changed_proteins' in protein_changes:
            drug_targets['protein_targets'] = protein_changes['most_changed_proteins']
        
        # Combine and rank targets
        all_targets = set(drug_targets['regulatory_targets'] + drug_targets['protein_targets'])
        
        # Score targets based on multiple criteria
        target_scores = {}
        for target in all_targets:
            score = 0
            
            # Score based on regulatory importance
            if target in drug_targets['regulatory_targets']:
                score += 1
            
            # Score based on protein change
            if target in drug_targets['protein_targets']:
                score += 1
            
            target_scores[target] = score
        
        # Rank targets
        ranked_targets = sorted(target_scores.items(), key=lambda x: x[1], reverse=True)
        drug_targets['combined_targets'] = [target for target, score in ranked_targets]
        
        return drug_targets
    
    def _compute_pharmacological_profile(self,
                                       control_network: nx.Graph,
                                       drug_network: nx.Graph,
                                       velocity_adata: anndata.AnnData,
                                       drug_condition: str) -> Dict[str, Any]:
        """Compute comprehensive pharmacological profile."""
        
        profile = {
            'network_metrics': {},
            'velocity_metrics': {},
            'perturbation_metrics': {},
            'pathway_enrichment': {}
        }
        
        # Network metrics comparison
        control_metrics = self._compute_network_metrics(control_network)
        drug_metrics = self._compute_network_metrics(drug_network)
        
        profile['network_metrics'] = {
            'control': control_metrics,
            'drug': drug_metrics,
            'changes': {k: drug_metrics[k] - control_metrics.get(k, 0) 
                       for k in drug_metrics.keys()}
        }
        
        # Velocity metrics for drug condition
        drug_mask = velocity_adata.obs['condition'] == drug_condition
        if np.any(drug_mask):
            profile['velocity_metrics'] = self._compute_velocity_metrics(
                velocity_adata[drug_mask]
            )
        
        # Perturbation strength
        profile['perturbation_metrics'] = {
            'network_perturbation_score': self._compute_network_perturbation_score(
                control_network, drug_network
            )
        }
        
        return profile
    
    def _compute_target_scores(self,
                             perturbed_modules: Dict[str, Any],
                             drug_targets: Dict[str, Any],
                             results: Dict[str, Any]) -> Dict[str, float]:
        """Compute target scores for drug target prioritization."""
        
        target_scores = {}
        
        # Score regulatory targets
        if 'nodes_with_changes' in perturbed_modules:
            for target in perturbed_modules['nodes_with_changes']:
                target_scores[target] = target_scores.get(target, 0) + 1.0
        
        # Score protein targets
        if 'most_changed_proteins' in results.get('protein_changes', {}):
            for target in results['protein_changes']['most_changed_proteins']:
                target_scores[target] = target_scores.get(target, 0) + 0.8
        
        # Add velocity-based scoring
        if 'velocity_info' in perturbed_modules:
            velocity_info = perturbed_modules['velocity_info']
            for gene, info in velocity_info.items():
                if 'velocity_magnitude' in info:
                    velocity_score = min(info['velocity_magnitude'] / 10.0, 1.0)
                    target_scores[gene] = target_scores.get(gene, 0) + velocity_score
        
        return target_scores
    
    def _annotate_drug_targets(self,
                             target_scores: Dict[str, float],
                             target_databases: List[str]) -> Dict[str, Any]:
        """Annotate targets with known drug-target interactions."""
        
        # This would interface with actual drug databases
        # For now, return placeholder annotations
        
        annotations = {}
        
        for target in target_scores.keys():
            annotations[target] = {
                'known_drugs': [],  # Would be populated from databases
                'target_class': 'unknown',
                'druggability_score': np.random.uniform(0, 1),  # Placeholder
                'clinical_stage': 'preclinical'
            }
        
        return annotations
    
    def _predict_novel_targets(self,
                             target_scores: Dict[str, float],
                             target_annotations: Dict[str, Any]) -> List[str]:
        """Predict novel drug targets."""
        
        # Simple heuristic: targets with high scores but no known drugs
        novel_targets = []
        
        for target, score in target_scores.items():
            if score > 1.0 and target in target_annotations:
                if len(target_annotations[target]['known_drugs']) == 0:
                    novel_targets.append(target)
        
        return novel_targets
    
    def _rank_top_targets(self,
                        target_scores: Dict[str, float],
                        n_top: int = 20) -> List[Tuple[str, float]]:
        """Rank top drug targets by score."""
        
        return sorted(target_scores.items(), key=lambda x: x[1], reverse=True)[:n_top]
    
    # Helper methods
    
    def _safe_average_shortest_path_length(self, network: nx.Graph) -> float:
        """Safely compute average shortest path length."""
        try:
            if nx.is_connected(network):
                return nx.average_shortest_path_length(network)
            else:
                # For disconnected graphs, compute for largest component
                largest_cc = max(nx.connected_components(network), key=len)
                subgraph = network.subgraph(largest_cc)
                return nx.average_shortest_path_length(subgraph)
        except:
            return float('inf')
    
    def _compute_velocity_metrics(self, velocity_adata: anndata.AnnData) -> Dict[str, float]:
        """Compute velocity-based metrics."""
        
        if 'velocity' not in velocity_adata.layers:
            return {}
        
        velocity_vectors = velocity_adata.layers['velocity']
        velocity_magnitudes = np.linalg.norm(velocity_vectors, axis=1)
        
        return {
            'mean_velocity_magnitude': np.mean(velocity_magnitudes),
            'std_velocity_magnitude': np.std(velocity_magnitudes),
            'max_velocity_magnitude': np.max(velocity_magnitudes),
            'velocity_coherence': self._compute_velocity_coherence(velocity_vectors)
        }
    
    def _compute_velocity_coherence(self, velocity_vectors: np.ndarray) -> float:
        """Compute velocity coherence (how aligned velocities are)."""
        
        # Normalize velocity vectors
        norms = np.linalg.norm(velocity_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized_velocities = velocity_vectors / norms
        
        # Compute pairwise cosine similarities
        similarities = np.dot(normalized_velocities, normalized_velocities.T)
        
        # Return mean similarity (excluding diagonal)
        mask = ~np.eye(similarities.shape[0], dtype=bool)
        return np.mean(similarities[mask])
    
    def _compute_network_metrics(self, network: nx.Graph) -> Dict[str, float]:
        """Compute basic network metrics."""
        
        return {
            'n_nodes': network.number_of_nodes(),
            'n_edges': network.number_of_edges(),
            'density': nx.density(network),
            'avg_clustering': nx.average_clustering(network),
            'avg_path_length': self._safe_average_shortest_path_length(network)
        }
    
    def _compute_network_perturbation_score(self,
                                          control_network: nx.Graph,
                                          drug_network: nx.Graph) -> float:
        """Compute overall network perturbation score."""
        
        # Find common nodes
        common_nodes = set(control_network.nodes()) & set(drug_network.nodes())
        
        if len(common_nodes) == 0:
            return 1.0  # Maximum perturbation
        
        # Compute edge overlap
        control_edges = set(control_network.subgraph(common_nodes).edges())
        drug_edges = set(drug_network.subgraph(common_nodes).edges())
        
        if len(control_edges) == 0 and len(drug_edges) == 0:
            return 0.0
        
        # Jaccard distance as perturbation score
        intersection = len(control_edges & drug_edges)
        union = len(control_edges | drug_edges)
        
        jaccard_similarity = intersection / union if union > 0 else 0
        perturbation_score = 1 - jaccard_similarity
        
        return perturbation_score
    
    def _fit_dose_response_curves(self,
                                doses: List[float],
                                network_metrics: Dict[float, Dict]) -> Dict[str, Any]:
        """Fit dose-response curves to network metrics."""
        
        dose_response_curves = {}
        
        # Extract metrics for curve fitting
        metric_names = list(next(iter(network_metrics.values())).keys())
        
        for metric_name in metric_names:
            metric_values = [network_metrics[dose][metric_name] for dose in doses]
            
            # Simple linear fit (could be extended to Hill equation)
            try:
                from scipy.optimize import curve_fit
                
                def linear_model(x, a, b):
                    return a * x + b
                
                popt, pcov = curve_fit(linear_model, doses, metric_values)
                
                dose_response_curves[metric_name] = {
                    'model': 'linear',
                    'parameters': popt,
                    'covariance': pcov,
                    'r_squared': self._compute_r_squared(metric_values, 
                                                       [linear_model(d, *popt) for d in doses])
                }
                
            except:
                dose_response_curves[metric_name] = {
                    'model': 'failed',
                    'parameters': None
                }
        
        return dose_response_curves
    
    def _compute_r_squared(self, y_true: List[float], y_pred: List[float]) -> float:
        """Compute R-squared value."""
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of all velocity-network analyses."""
        
        summary = {
            'velocity_data_processed': len(self.velocity_data),
            'dynamic_networks_analyzed': len(self.dynamic_networks),
            'drug_responses_analyzed': len(self.drug_response_profiles),
            'conditions': list(self.velocity_data.keys()) if self.velocity_data else []
        }
        
        if self.drug_response_profiles:
            # Summarize drug response results
            all_targets = set()
            for drug_result in self.drug_response_profiles.values():
                if 'drug_targets' in drug_result:
                    all_targets.update(drug_result['drug_targets'].get('combined_targets', []))
            
            summary['total_drug_targets_identified'] = len(all_targets)
            summary['top_drug_targets'] = list(all_targets)[:10]
        
        return summary


class VelocityEnhancedBridge:
    """
    BRIDGE integrator enhanced with RNA velocity for drug response analysis.
    
    Combines velocity-guided network dynamics with SCENIC+ and PINNACLE
    integration for comprehensive drug response and pharmacology studies.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize velocity-enhanced BRIDGE."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__ + '.VelocityEnhancedBridge')
        
        # Initialize velocity analyzer
        velocity_config = self.config.get('velocity', {})
        self.velocity_analyzer = VelocityNetworkAnalyzer(config=velocity_config)
        
        # Will be initialized when needed
        self.bridge_integrator = None
        self.velocity_results = {}
    
    def analyze_drug_response_networks(self,
                                     rna_adata: anndata.AnnData,
                                     scenic_networks: Dict[str, nx.Graph],
                                     pinnacle_embeddings: Dict[str, Dict],
                                     drug_conditions: List[str],
                                     control_condition: str = 'control',
                                     condition_key: str = 'condition') -> Dict[str, Any]:
        """
        Complete drug response network analysis with velocity integration.
        
        Args:
            rna_adata: AnnData with spliced/unspliced counts
            scenic_networks: Dictionary of regulatory networks
            pinnacle_embeddings: Dictionary of protein embeddings
            drug_conditions: List of drug treatment conditions
            control_condition: Control condition name
            condition_key: Column indicating conditions
            
        Returns:
            Complete drug response analysis results
        """
        self.logger.info("Starting velocity-enhanced drug response analysis...")
        
        # Step 1: Compute RNA velocity
        velocity_adata = self.velocity_analyzer.compute_rna_velocity(
            rna_adata, condition_key=condition_key
        )
        
        # Step 2: Analyze network dynamics
        network_dynamics = self.velocity_analyzer.analyze_network_dynamics(
            scenic_networks, velocity_adata, condition_key=condition_key
        )
        
        # Step 3: Identify drug response networks
        drug_response_results = self.velocity_analyzer.identify_drug_response_networks(
            scenic_networks, pinnacle_embeddings, velocity_adata,
            drug_conditions, control_condition
        )
        
        # Step 4: Initialize standard BRIDGE for comparison
        from .core import BridgeIntegrator
        
        bridge_config = self.config.get('bridge', {})
        self.bridge_integrator = BridgeIntegrator(config=bridge_config)
        self.bridge_integrator.scenic_networks = scenic_networks
        self.bridge_integrator.pinnacle_embeddings = pinnacle_embeddings
        
        # Step 5: Integrate with velocity information
        integrated_networks = self.bridge_integrator.integrate_networks()
        
        # Step 6: Identify pharmacological targets
        pharmacological_targets = self.velocity_analyzer.identify_pharmacological_targets(
            drug_response_results
        )
        
        # Store results
        self.velocity_results = {
            'velocity_adata': velocity_adata,
            'network_dynamics': network_dynamics,
            'drug_response_results': drug_response_results,
            'integrated_networks': integrated_networks,
            'pharmacological_targets': pharmacological_targets
        }
        
        return {
            'velocity_adata': velocity_adata,
            'network_dynamics': network_dynamics,
            'drug_response_results': drug_response_results,
            'integrated_networks': integrated_networks,
            'pharmacological_targets': pharmacological_targets,
            'analysis_summary': self.velocity_analyzer.get_analysis_summary()
        }
    
    def analyze_dose_response(self,
                            rna_adata: anndata.AnnData,
                            scenic_networks: Dict[str, nx.Graph],
                            dose_conditions: List[str],
                            doses: List[float],
                            control_condition: str = 'control',
                            condition_key: str = 'condition') -> Dict[str, Any]:
        """
        Analyze dose-response relationships in network dynamics.
        
        Args:
            rna_adata: AnnData with velocity information
            scenic_networks: Dictionary of networks by dose condition
            dose_conditions: List of dose condition names
            doses: List of corresponding dose values
            control_condition: Control condition name
            condition_key: Column indicating conditions
            
        Returns:
            Dose-response analysis results
        """
        self.logger.info("Analyzing dose-response network dynamics...")
        
        # Compute velocity
        velocity_adata = self.velocity_analyzer.compute_rna_velocity(
            rna_adata, condition_key=condition_key
        )
        
        # Analyze dose-response dynamics
        dose_response_results = self.velocity_analyzer.analyze_dose_response_dynamics(
            scenic_networks, velocity_adata, dose_conditions, doses, control_condition
        )
        
        return dose_response_results

