"""
Data processing module for SCENIC+ and PINNACLE integration.

This module provides classes for loading, preprocessing, and formatting
data from SCENIC+ and PINNACLE analyses.
"""

import pandas as pd
import numpy as np
import networkx as nx
import scanpy as sc
import anndata
import pickle
import h5py
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import logging

class ScenicProcessor:
    """
    Processor for SCENIC+ regulatory network data.
    
    Handles loading, preprocessing, and extraction of regulatory networks
    from SCENIC+ outputs in various formats.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize SCENIC+ processor.
        
        Args:
            config: Configuration parameters for processing
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__ + '.ScenicProcessor')
        
    def load_data(self, data_path: Union[str, Path], 
                  data_format: str = 'pickle') -> Dict:
        """
        Load SCENIC+ regulatory network data.
        
        Args:
            data_path: Path to SCENIC+ output files
            data_format: Format of data ('pickle', 'csv', 'h5ad')
            
        Returns:
            Dictionary of regulatory networks by condition/cell type
        """
        data_path = Path(data_path)
        
        if data_format == 'pickle':
            return self._load_pickle_data(data_path)
        elif data_format == 'csv':
            return self._load_csv_data(data_path)
        elif data_format == 'h5ad':
            return self._load_h5ad_data(data_path)
        else:
            raise ValueError(f"Unsupported data format: {data_format}")
    
    def _load_pickle_data(self, data_path: Path) -> Dict:
        """Load SCENIC+ data from pickle files."""
        networks = {}
        
        if data_path.is_file():
            # Single pickle file
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, dict):
                    networks = data
                else:
                    networks['default'] = data
        else:
            # Directory with multiple pickle files
            for pickle_file in data_path.glob('*.pkl'):
                condition = pickle_file.stem
                with open(pickle_file, 'rb') as f:
                    networks[condition] = pickle.load(f)
        
        return self._process_scenic_networks(networks)
    
    def _load_csv_data(self, data_path: Path) -> Dict:
        """Load SCENIC+ data from CSV files."""
        networks = {}
        
        if data_path.is_file():
            # Single CSV file with regulons
            df = pd.read_csv(data_path)
            networks['default'] = self._csv_to_network(df)
        else:
            # Directory with multiple CSV files
            for csv_file in data_path.glob('*.csv'):
                condition = csv_file.stem
                df = pd.read_csv(csv_file)
                networks[condition] = self._csv_to_network(df)
        
        return self._process_scenic_networks(networks)
    
    def _load_h5ad_data(self, data_path: Path) -> Dict:
        """Load SCENIC+ data from AnnData h5ad files."""
        networks = {}
        
        if data_path.is_file():
            # Single h5ad file
            adata = sc.read_h5ad(data_path)
            networks = self._extract_networks_from_adata(adata)
        else:
            # Directory with multiple h5ad files
            for h5ad_file in data_path.glob('*.h5ad'):
                condition = h5ad_file.stem
                adata = sc.read_h5ad(h5ad_file)
                networks[condition] = self._extract_networks_from_adata(adata)
        
        return self._process_scenic_networks(networks)
    
    def _csv_to_network(self, df: pd.DataFrame) -> nx.DiGraph:
        """Convert CSV regulon data to NetworkX graph."""
        G = nx.DiGraph()
        
        # Expected columns: TF, target, importance, enhancer (optional)
        for _, row in df.iterrows():
            tf = row['TF']
            target = row['target']
            importance = row.get('importance', 1.0)
            enhancer = row.get('enhancer', None)
            
            # Add nodes
            G.add_node(tf, node_type='TF')
            G.add_node(target, node_type='target')
            
            # Add edge
            edge_attrs = {'importance': importance, 'edge_type': 'regulation'}
            if enhancer:
                edge_attrs['enhancer'] = enhancer
            
            G.add_edge(tf, target, **edge_attrs)
        
        return G
    
    def _extract_networks_from_adata(self, adata: anndata.AnnData) -> Dict:
        """Extract regulatory networks from AnnData object."""
        networks = {}
        
        # Check for regulons in uns
        if 'regulons' in adata.uns:
            regulons = adata.uns['regulons']
            
            # Group by cell type if available
            if 'cell_type' in adata.obs:
                cell_types = adata.obs['cell_type'].unique()
                for cell_type in cell_types:
                    cell_mask = adata.obs['cell_type'] == cell_type
                    networks[cell_type] = self._build_network_from_regulons(
                        regulons, cell_mask, adata
                    )
            else:
                networks['default'] = self._build_network_from_regulons(
                    regulons, None, adata
                )
        
        return networks
    
    def _build_network_from_regulons(self, regulons: Dict, 
                                   cell_mask: Optional[np.ndarray],
                                   adata: anndata.AnnData) -> nx.DiGraph:
        """Build NetworkX graph from regulon dictionary."""
        G = nx.DiGraph()
        
        for tf, targets in regulons.items():
            G.add_node(tf, node_type='TF')
            
            for target in targets:
                G.add_node(target, node_type='target')
                
                # Calculate importance based on expression correlation
                if cell_mask is not None:
                    tf_expr = adata[cell_mask, tf].X
                    target_expr = adata[cell_mask, target].X
                else:
                    tf_expr = adata[:, tf].X
                    target_expr = adata[:, target].X
                
                # Handle sparse matrices
                if hasattr(tf_expr, 'toarray'):
                    tf_expr = tf_expr.toarray().flatten()
                    target_expr = target_expr.toarray().flatten()
                
                importance = np.corrcoef(tf_expr, target_expr)[0, 1]
                importance = np.nan_to_num(importance, nan=0.0)
                
                G.add_edge(tf, target, 
                          importance=importance, 
                          edge_type='regulation')
        
        return G
    
    def _process_scenic_networks(self, networks: Dict) -> Dict:
        """Apply processing and filtering to SCENIC+ networks."""
        processed_networks = {}
        
        for condition, network in networks.items():
            if not isinstance(network, nx.Graph):
                self.logger.warning(f"Converting {condition} to NetworkX graph")
                network = self._convert_to_networkx(network)
            
            # Apply filtering based on configuration
            network = self._filter_network(network)
            
            # Add network-level attributes
            network.graph['condition'] = condition
            network.graph['source'] = 'SCENIC+'
            
            processed_networks[condition] = network
        
        return processed_networks
    
    def _convert_to_networkx(self, data: Any) -> nx.DiGraph:
        """Convert various data formats to NetworkX graph."""
        if isinstance(data, pd.DataFrame):
            return self._csv_to_network(data)
        elif isinstance(data, dict):
            # Assume regulon format
            G = nx.DiGraph()
            for tf, targets in data.items():
                G.add_node(tf, node_type='TF')
                for target in targets:
                    G.add_node(target, node_type='target')
                    G.add_edge(tf, target, importance=1.0, edge_type='regulation')
            return G
        else:
            raise ValueError(f"Cannot convert data type {type(data)} to NetworkX graph")
    
    def _filter_network(self, network: nx.DiGraph) -> nx.DiGraph:
        """Apply filtering criteria to network."""
        filtered_network = network.copy()
        
        # Filter by minimum regulon size
        min_regulon_size = self.config.get('min_regulon_size', 5)
        tfs_to_remove = []
        
        for node in filtered_network.nodes():
            if filtered_network.nodes[node].get('node_type') == 'TF':
                out_degree = filtered_network.out_degree(node)
                if out_degree < min_regulon_size:
                    tfs_to_remove.append(node)
        
        for tf in tfs_to_remove:
            filtered_network.remove_node(tf)
        
        # Filter by importance threshold
        importance_threshold = self.config.get('importance_threshold', 0.1)
        edges_to_remove = []
        
        for u, v, data in filtered_network.edges(data=True):
            importance = data.get('importance', 0)
            if abs(importance) < importance_threshold:
                edges_to_remove.append((u, v))
        
        for edge in edges_to_remove:
            filtered_network.remove_edge(*edge)
        
        # Remove isolated nodes
        isolated_nodes = list(nx.isolates(filtered_network))
        filtered_network.remove_nodes_from(isolated_nodes)
        
        return filtered_network
    
    def extract_target_genes(self, network: nx.DiGraph) -> List[str]:
        """
        Extract target genes from SCENIC+ network.
        
        Args:
            network: SCENIC+ regulatory network
            
        Returns:
            List of target gene identifiers
        """
        target_genes = []
        
        for node in network.nodes():
            if network.nodes[node].get('node_type') == 'target':
                target_genes.append(node)
        
        return target_genes
    
    def get_regulon_info(self, network: nx.DiGraph) -> pd.DataFrame:
        """
        Extract regulon information from network.
        
        Args:
            network: SCENIC+ regulatory network
            
        Returns:
            DataFrame with regulon information
        """
        regulon_data = []
        
        for node in network.nodes():
            if network.nodes[node].get('node_type') == 'TF':
                targets = list(network.successors(node))
                for target in targets:
                    edge_data = network.edges[node, target]
                    regulon_data.append({
                        'TF': node,
                        'target': target,
                        'importance': edge_data.get('importance', 0),
                        'enhancer': edge_data.get('enhancer', None)
                    })
        
        return pd.DataFrame(regulon_data)


class PinnacleProcessor:
    """
    Processor for PINNACLE protein embedding data.
    
    Handles loading, preprocessing, and extraction of protein embeddings
    from PINNACLE outputs in various formats.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize PINNACLE processor.
        
        Args:
            config: Configuration parameters for processing
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__ + '.PinnacleProcessor')
    
    def load_data(self, data_path: Union[str, Path],
                  data_format: str = 'pickle') -> Dict:
        """
        Load PINNACLE protein embedding data.
        
        Args:
            data_path: Path to PINNACLE output files
            data_format: Format of data ('pickle', 'csv', 'npz')
            
        Returns:
            Dictionary of protein embeddings by context
        """
        data_path = Path(data_path)
        
        if data_format == 'pickle':
            return self._load_pickle_data(data_path)
        elif data_format == 'csv':
            return self._load_csv_data(data_path)
        elif data_format == 'npz':
            return self._load_npz_data(data_path)
        else:
            raise ValueError(f"Unsupported data format: {data_format}")
    
    def _load_pickle_data(self, data_path: Path) -> Dict:
        """Load PINNACLE data from pickle files."""
        embeddings = {}
        
        if data_path.is_file():
            # Single pickle file
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, dict):
                    embeddings = data
                else:
                    embeddings['default'] = data
        else:
            # Directory with multiple pickle files
            for pickle_file in data_path.glob('*.pkl'):
                context = pickle_file.stem
                with open(pickle_file, 'rb') as f:
                    embeddings[context] = pickle.load(f)
        
        return self._process_pinnacle_embeddings(embeddings)
    
    def _load_csv_data(self, data_path: Path) -> Dict:
        """Load PINNACLE data from CSV files."""
        embeddings = {}
        
        if data_path.is_file():
            # Single CSV file
            df = pd.read_csv(data_path, index_col=0)
            embeddings['default'] = df
        else:
            # Directory with multiple CSV files
            for csv_file in data_path.glob('*.csv'):
                context = csv_file.stem
                df = pd.read_csv(csv_file, index_col=0)
                embeddings[context] = df
        
        return self._process_pinnacle_embeddings(embeddings)
    
    def _load_npz_data(self, data_path: Path) -> Dict:
        """Load PINNACLE data from NPZ files."""
        embeddings = {}
        
        if data_path.is_file():
            # Single NPZ file
            data = np.load(data_path, allow_pickle=True)
            embeddings['default'] = {
                'embeddings': data['embeddings'],
                'protein_ids': data['protein_ids'],
                'contexts': data.get('contexts', ['default'])
            }
        else:
            # Directory with multiple NPZ files
            for npz_file in data_path.glob('*.npz'):
                context = npz_file.stem
                data = np.load(npz_file, allow_pickle=True)
                embeddings[context] = {
                    'embeddings': data['embeddings'],
                    'protein_ids': data['protein_ids']
                }
        
        return self._process_pinnacle_embeddings(embeddings)
    
    def _process_pinnacle_embeddings(self, embeddings: Dict) -> Dict:
        """Apply processing to PINNACLE embeddings."""
        processed_embeddings = {}
        
        for context, data in embeddings.items():
            if isinstance(data, pd.DataFrame):
                # DataFrame format
                processed_embeddings[context] = {
                    'embeddings': data.values,
                    'protein_ids': data.index.tolist(),
                    'embedding_dim': data.shape[1]
                }
            elif isinstance(data, dict):
                # Dictionary format
                processed_embeddings[context] = data
                if 'embedding_dim' not in data:
                    data['embedding_dim'] = data['embeddings'].shape[1]
            elif isinstance(data, np.ndarray):
                # Array format (assume protein IDs are indices)
                processed_embeddings[context] = {
                    'embeddings': data,
                    'protein_ids': [f'protein_{i}' for i in range(data.shape[0])],
                    'embedding_dim': data.shape[1]
                }
            
            # Apply normalization if specified
            if self.config.get('normalize_embeddings', True):
                embeddings_array = processed_embeddings[context]['embeddings']
                # L2 normalization
                norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
                norms[norms == 0] = 1  # Avoid division by zero
                processed_embeddings[context]['embeddings'] = embeddings_array / norms
        
        return processed_embeddings
    
    def get_context_embeddings(self, embeddings: Dict, 
                             context: str,
                             protein_ids: List[str]) -> Dict:
        """
        Get embeddings for specific proteins in a given context.
        
        Args:
            embeddings: Full embedding dictionary
            context: Context identifier
            protein_ids: List of protein IDs to extract
            
        Returns:
            Dictionary with embeddings for specified proteins
        """
        if context not in embeddings:
            self.logger.warning(f"Context {context} not found, using default")
            context = list(embeddings.keys())[0]
        
        context_data = embeddings[context]
        available_proteins = context_data['protein_ids']
        embedding_matrix = context_data['embeddings']
        
        # Find indices of requested proteins
        protein_indices = []
        found_proteins = []
        
        for protein_id in protein_ids:
            if protein_id in available_proteins:
                idx = available_proteins.index(protein_id)
                protein_indices.append(idx)
                found_proteins.append(protein_id)
            else:
                self.logger.warning(f"Protein {protein_id} not found in {context}")
        
        if not protein_indices:
            self.logger.error(f"No proteins found in context {context}")
            return {
                'embeddings': np.array([]),
                'protein_ids': [],
                'embedding_dim': context_data['embedding_dim']
            }
        
        # Extract embeddings
        selected_embeddings = embedding_matrix[protein_indices]
        
        return {
            'embeddings': selected_embeddings,
            'protein_ids': found_proteins,
            'embedding_dim': context_data['embedding_dim'],
            'context': context
        }
    
    def compute_protein_similarities(self, embeddings: Dict,
                                   metric: str = 'cosine') -> pd.DataFrame:
        """
        Compute pairwise protein similarities from embeddings.
        
        Args:
            embeddings: Protein embeddings dictionary
            metric: Similarity metric ('cosine', 'euclidean', 'correlation')
            
        Returns:
            DataFrame with pairwise similarities
        """
        from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
        from scipy.stats import pearsonr
        
        embedding_matrix = embeddings['embeddings']
        protein_ids = embeddings['protein_ids']
        
        if metric == 'cosine':
            similarity_matrix = cosine_similarity(embedding_matrix)
        elif metric == 'euclidean':
            # Convert distances to similarities
            distance_matrix = euclidean_distances(embedding_matrix)
            similarity_matrix = 1 / (1 + distance_matrix)
        elif metric == 'correlation':
            n_proteins = len(protein_ids)
            similarity_matrix = np.zeros((n_proteins, n_proteins))
            for i in range(n_proteins):
                for j in range(n_proteins):
                    if i == j:
                        similarity_matrix[i, j] = 1.0
                    else:
                        corr, _ = pearsonr(embedding_matrix[i], embedding_matrix[j])
                        similarity_matrix[i, j] = corr if not np.isnan(corr) else 0.0
        else:
            raise ValueError(f"Unsupported similarity metric: {metric}")
        
        return pd.DataFrame(similarity_matrix, 
                          index=protein_ids, 
                          columns=protein_ids)
    
    def get_embedding_statistics(self, embeddings: Dict) -> Dict:
        """
        Compute statistics for protein embeddings.
        
        Args:
            embeddings: Protein embeddings dictionary
            
        Returns:
            Dictionary with embedding statistics
        """
        embedding_matrix = embeddings['embeddings']
        
        stats = {
            'num_proteins': len(embeddings['protein_ids']),
            'embedding_dim': embeddings['embedding_dim'],
            'mean_norm': np.mean(np.linalg.norm(embedding_matrix, axis=1)),
            'std_norm': np.std(np.linalg.norm(embedding_matrix, axis=1)),
            'mean_embedding': np.mean(embedding_matrix, axis=0),
            'std_embedding': np.std(embedding_matrix, axis=0),
            'embedding_range': {
                'min': np.min(embedding_matrix),
                'max': np.max(embedding_matrix)
            }
        }
        
        return stats

