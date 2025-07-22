"""
Utility functions for SCENIC+ and PINNACLE integration.

This module provides helper classes for identifier mapping, quality control,
and other common operations.
"""

import pandas as pd
import numpy as np
import networkx as nx
import requests
import time
from typing import Dict, List, Optional, Set, Tuple, Union
import logging
import re
from pathlib import Path
import json

class IdentifierMapper:
    """
    Maps between different gene and protein identifier systems.
    
    Handles conversion between gene symbols, Ensembl IDs, UniProt IDs,
    and other common identifier formats.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize identifier mapper.
        
        Args:
            config: Configuration parameters for mapping
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__ + '.IdentifierMapper')
        
        # Initialize mapping databases
        self.gene_mappings = {}
        self.protein_mappings = {}
        self.mapping_cache = {}
        
        # Load pre-built mappings if available
        self._load_mapping_databases()
    
    def _load_mapping_databases(self) -> None:
        """Load pre-built identifier mapping databases."""
        mapping_dir = Path(self.config.get('mapping_dir', 'data/mappings'))
        
        if mapping_dir.exists():
            # Load gene mappings
            gene_mapping_file = mapping_dir / 'gene_mappings.json'
            if gene_mapping_file.exists():
                with open(gene_mapping_file, 'r') as f:
                    self.gene_mappings = json.load(f)
                self.logger.info(f"Loaded {len(self.gene_mappings)} gene mappings")
            
            # Load protein mappings
            protein_mapping_file = mapping_dir / 'protein_mappings.json'
            if protein_mapping_file.exists():
                with open(protein_mapping_file, 'r') as f:
                    self.protein_mappings = json.load(f)
                self.logger.info(f"Loaded {len(self.protein_mappings)} protein mappings")
    
    def map_genes_to_proteins(self, gene_ids: List[str]) -> Dict[str, str]:
        """
        Map gene identifiers to protein identifiers.
        
        Args:
            gene_ids: List of gene identifiers
            
        Returns:
            Dictionary mapping gene IDs to protein IDs
        """
        self.logger.info(f"Mapping {len(gene_ids)} genes to proteins")
        
        gene_protein_mapping = {}
        unmapped_genes = []
        
        for gene_id in gene_ids:
            # Check cache first
            if gene_id in self.mapping_cache:
                protein_id = self.mapping_cache[gene_id]
                if protein_id:
                    gene_protein_mapping[gene_id] = protein_id
                continue
            
            # Try pre-loaded mappings
            protein_id = self._lookup_gene_to_protein(gene_id)
            
            if protein_id:
                gene_protein_mapping[gene_id] = protein_id
                self.mapping_cache[gene_id] = protein_id
            else:
                unmapped_genes.append(gene_id)
        
        # Query online databases for unmapped genes
        if unmapped_genes and self.config.get('use_online_mapping', True):
            online_mappings = self._query_online_mappings(unmapped_genes)
            gene_protein_mapping.update(online_mappings)
        
        self.logger.info(f"Mapped {len(gene_protein_mapping)} genes to proteins")
        if unmapped_genes:
            self.logger.warning(f"{len(unmapped_genes)} genes could not be mapped")
        
        return gene_protein_mapping
    
    def _lookup_gene_to_protein(self, gene_id: str) -> Optional[str]:
        """Look up gene to protein mapping in pre-loaded databases."""
        # Direct lookup
        if gene_id in self.gene_mappings:
            return self.gene_mappings[gene_id].get('uniprot_id')
        
        # Try different identifier formats
        gene_id_clean = self._clean_identifier(gene_id)
        
        for stored_gene, mapping_info in self.gene_mappings.items():
            stored_gene_clean = self._clean_identifier(stored_gene)
            
            if gene_id_clean == stored_gene_clean:
                return mapping_info.get('uniprot_id')
            
            # Check alternative symbols
            alt_symbols = mapping_info.get('alt_symbols', [])
            if gene_id_clean in [self._clean_identifier(alt) for alt in alt_symbols]:
                return mapping_info.get('uniprot_id')
        
        return None
    
    def _clean_identifier(self, identifier: str) -> str:
        """Clean and standardize identifier format."""
        # Remove version numbers (e.g., ENSG00000000003.14 -> ENSG00000000003)
        identifier = re.sub(r'\.\d+$', '', identifier)
        
        # Convert to uppercase for consistency
        identifier = identifier.upper()
        
        return identifier
    
    def _query_online_mappings(self, gene_ids: List[str]) -> Dict[str, str]:
        """Query online databases for gene-protein mappings."""
        mappings = {}
        
        # Use MyGene.info API
        if self.config.get('use_mygene', True):
            mygene_mappings = self._query_mygene(gene_ids)
            mappings.update(mygene_mappings)
        
        # Use Ensembl REST API
        if self.config.get('use_ensembl', True):
            remaining_genes = [g for g in gene_ids if g not in mappings]
            if remaining_genes:
                ensembl_mappings = self._query_ensembl(remaining_genes)
                mappings.update(ensembl_mappings)
        
        return mappings
    
    def _query_mygene(self, gene_ids: List[str]) -> Dict[str, str]:
        """Query MyGene.info API for gene information."""
        mappings = {}
        
        try:
            # Batch query to MyGene.info
            url = "http://mygene.info/v3/query"
            
            # Process in batches to avoid API limits
            batch_size = 100
            for i in range(0, len(gene_ids), batch_size):
                batch = gene_ids[i:i + batch_size]
                
                # Create query string
                query = ' OR '.join([f'symbol:{gene}' for gene in batch])
                
                params = {
                    'q': query,
                    'fields': 'uniprot,symbol,ensembl.gene',
                    'species': 'human',
                    'size': batch_size
                }
                
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                
                for hit in data.get('hits', []):
                    gene_symbol = hit.get('symbol')
                    uniprot_info = hit.get('uniprot')
                    
                    if gene_symbol and uniprot_info:
                        # Get primary UniProt ID
                        if isinstance(uniprot_info, dict):
                            uniprot_id = uniprot_info.get('Swiss-Prot')
                            if not uniprot_id:
                                uniprot_id = uniprot_info.get('TrEMBL')
                        elif isinstance(uniprot_info, str):
                            uniprot_id = uniprot_info
                        else:
                            continue
                        
                        if uniprot_id:
                            mappings[gene_symbol] = uniprot_id
                
                # Rate limiting
                time.sleep(0.1)
        
        except Exception as e:
            self.logger.warning(f"MyGene.info query failed: {e}")
        
        return mappings
    
    def _query_ensembl(self, gene_ids: List[str]) -> Dict[str, str]:
        """Query Ensembl REST API for gene information."""
        mappings = {}
        
        try:
            # Use Ensembl REST API
            base_url = "https://rest.ensembl.org"
            
            for gene_id in gene_ids:
                # Try lookup by symbol
                url = f"{base_url}/lookup/symbol/homo_sapiens/{gene_id}"
                headers = {'Content-Type': 'application/json'}
                
                response = requests.get(url, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    ensembl_id = data.get('id')
                    
                    if ensembl_id:
                        # Get UniProt mapping
                        xref_url = f"{base_url}/xrefs/id/{ensembl_id}"
                        xref_response = requests.get(xref_url, headers=headers, timeout=10)
                        
                        if xref_response.status_code == 200:
                            xref_data = xref_response.json()
                            
                            for xref in xref_data:
                                if xref.get('dbname') == 'Uniprot/SWISSPROT':
                                    mappings[gene_id] = xref.get('primary_id')
                                    break
                
                # Rate limiting
                time.sleep(0.1)
        
        except Exception as e:
            self.logger.warning(f"Ensembl query failed: {e}")
        
        return mappings
    
    def validate_mappings(self, mappings: Dict[str, str]) -> Dict[str, str]:
        """
        Validate gene-protein mappings.
        
        Args:
            mappings: Dictionary of gene-protein mappings
            
        Returns:
            Dictionary of validated mappings
        """
        validated_mappings = {}
        
        for gene_id, protein_id in mappings.items():
            if self._validate_gene_id(gene_id) and self._validate_protein_id(protein_id):
                validated_mappings[gene_id] = protein_id
            else:
                self.logger.warning(f"Invalid mapping: {gene_id} -> {protein_id}")
        
        return validated_mappings
    
    def _validate_gene_id(self, gene_id: str) -> bool:
        """Validate gene identifier format."""
        if not gene_id or not isinstance(gene_id, str):
            return False
        
        # Check for common gene ID patterns
        patterns = [
            r'^[A-Z][A-Z0-9-]+$',  # Gene symbols (e.g., TP53, BRCA1)
            r'^ENSG\d{11}$',       # Ensembl gene IDs
            r'^[0-9]+$'            # Entrez gene IDs
        ]
        
        return any(re.match(pattern, gene_id) for pattern in patterns)
    
    def _validate_protein_id(self, protein_id: str) -> bool:
        """Validate protein identifier format."""
        if not protein_id or not isinstance(protein_id, str):
            return False
        
        # Check for UniProt ID patterns
        patterns = [
            r'^[OPQ][0-9][A-Z0-9]{3}[0-9]$',     # UniProt accession
            r'^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$'  # UniProt accession
        ]
        
        return any(re.match(pattern, protein_id) for pattern in patterns)
    
    def save_mappings(self, output_dir: Union[str, Path]) -> None:
        """Save mapping cache to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save mapping cache
        cache_file = output_dir / 'mapping_cache.json'
        with open(cache_file, 'w') as f:
            json.dump(self.mapping_cache, f, indent=2)
        
        self.logger.info(f"Saved {len(self.mapping_cache)} mappings to {cache_file}")


class QualityController:
    """
    Quality control for SCENIC+ and PINNACLE integration.
    
    Validates data quality and provides quality metrics for integration results.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize quality controller.
        
        Args:
            config: Configuration parameters for quality control
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__ + '.QualityController')
    
    def validate_scenic_networks(self, networks: Dict[str, nx.DiGraph]) -> Dict[str, nx.DiGraph]:
        """
        Validate SCENIC+ regulatory networks.
        
        Args:
            networks: Dictionary of SCENIC+ networks
            
        Returns:
            Dictionary of validated networks
        """
        self.logger.info(f"Validating {len(networks)} SCENIC+ networks")
        
        validated_networks = {}
        
        for condition, network in networks.items():
            if self._validate_network_structure(network):
                if self._validate_network_size(network):
                    validated_networks[condition] = network
                else:
                    self.logger.warning(f"Network {condition} failed size validation")
            else:
                self.logger.warning(f"Network {condition} failed structure validation")
        
        self.logger.info(f"Validated {len(validated_networks)} networks")
        return validated_networks
    
    def validate_pinnacle_embeddings(self, embeddings: Dict) -> Dict:
        """
        Validate PINNACLE protein embeddings.
        
        Args:
            embeddings: Dictionary of protein embeddings
            
        Returns:
            Dictionary of validated embeddings
        """
        self.logger.info(f"Validating {len(embeddings)} PINNACLE embedding sets")
        
        validated_embeddings = {}
        
        for context, embedding_data in embeddings.items():
            if self._validate_embedding_structure(embedding_data):
                if self._validate_embedding_quality(embedding_data):
                    validated_embeddings[context] = embedding_data
                else:
                    self.logger.warning(f"Embeddings {context} failed quality validation")
            else:
                self.logger.warning(f"Embeddings {context} failed structure validation")
        
        self.logger.info(f"Validated {len(validated_embeddings)} embedding sets")
        return validated_embeddings
    
    def _validate_network_structure(self, network: nx.DiGraph) -> bool:
        """Validate network structure."""
        if not isinstance(network, nx.DiGraph):
            return False
        
        if network.number_of_nodes() == 0:
            return False
        
        # Check for required node attributes
        for node in network.nodes():
            node_data = network.nodes[node]
            if 'node_type' not in node_data:
                self.logger.warning(f"Node {node} missing node_type attribute")
                return False
        
        # Check for required edge attributes
        for u, v, data in network.edges(data=True):
            if 'importance' not in data:
                self.logger.warning(f"Edge ({u}, {v}) missing importance attribute")
                return False
        
        return True
    
    def _validate_network_size(self, network: nx.DiGraph) -> bool:
        """Validate network size constraints."""
        min_size = self.config.get('min_network_size', 10)
        max_size = self.config.get('max_network_size', 10000)
        
        num_nodes = network.number_of_nodes()
        
        if num_nodes < min_size:
            self.logger.warning(f"Network too small: {num_nodes} < {min_size}")
            return False
        
        if num_nodes > max_size:
            self.logger.warning(f"Network too large: {num_nodes} > {max_size}")
            return False
        
        return True
    
    def _validate_embedding_structure(self, embedding_data: Dict) -> bool:
        """Validate embedding data structure."""
        required_keys = ['embeddings', 'protein_ids', 'embedding_dim']
        
        for key in required_keys:
            if key not in embedding_data:
                self.logger.warning(f"Missing required key: {key}")
                return False
        
        embeddings = embedding_data['embeddings']
        protein_ids = embedding_data['protein_ids']
        embedding_dim = embedding_data['embedding_dim']
        
        # Check array dimensions
        if not isinstance(embeddings, np.ndarray):
            self.logger.warning("Embeddings not a numpy array")
            return False
        
        if embeddings.ndim != 2:
            self.logger.warning(f"Embeddings wrong dimensions: {embeddings.ndim}")
            return False
        
        if embeddings.shape[0] != len(protein_ids):
            self.logger.warning("Mismatch between embeddings and protein IDs")
            return False
        
        if embeddings.shape[1] != embedding_dim:
            self.logger.warning("Mismatch between embeddings and embedding_dim")
            return False
        
        return True
    
    def _validate_embedding_quality(self, embedding_data: Dict) -> bool:
        """Validate embedding quality."""
        embeddings = embedding_data['embeddings']
        
        # Check for NaN or infinite values
        if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
            self.logger.warning("Embeddings contain NaN or infinite values")
            return False
        
        # Check embedding norms
        norms = np.linalg.norm(embeddings, axis=1)
        if np.any(norms == 0):
            self.logger.warning("Some embeddings have zero norm")
            return False
        
        # Check for reasonable value ranges
        if np.max(np.abs(embeddings)) > 100:
            self.logger.warning("Embeddings have very large values")
            return False
        
        return True
    
    def assess_scenic_quality(self, networks: Dict[str, nx.DiGraph]) -> Dict:
        """Assess quality of SCENIC+ networks."""
        quality_metrics = {}
        
        for condition, network in networks.items():
            metrics = {
                'num_nodes': network.number_of_nodes(),
                'num_edges': network.number_of_edges(),
                'density': nx.density(network),
                'avg_clustering': nx.average_clustering(network) if network.number_of_nodes() > 0 else 0,
                'num_components': nx.number_connected_components(network.to_undirected())
            }
            
            # Regulatory-specific metrics
            tf_nodes = [n for n in network.nodes() 
                       if network.nodes[n].get('node_type') == 'TF']
            target_nodes = [n for n in network.nodes()
                           if network.nodes[n].get('node_type') == 'target']
            
            metrics.update({
                'num_tfs': len(tf_nodes),
                'num_targets': len(target_nodes),
                'avg_regulon_size': np.mean([network.out_degree(tf) for tf in tf_nodes]) if tf_nodes else 0,
                'avg_importance': np.mean([data.get('importance', 0) 
                                        for _, _, data in network.edges(data=True)])
            })
            
            quality_metrics[condition] = metrics
        
        return quality_metrics
    
    def assess_pinnacle_quality(self, embeddings: Dict) -> Dict:
        """Assess quality of PINNACLE embeddings."""
        quality_metrics = {}
        
        for context, embedding_data in embeddings.items():
            embedding_matrix = embedding_data['embeddings']
            
            metrics = {
                'num_proteins': len(embedding_data['protein_ids']),
                'embedding_dim': embedding_data['embedding_dim'],
                'mean_norm': np.mean(np.linalg.norm(embedding_matrix, axis=1)),
                'std_norm': np.std(np.linalg.norm(embedding_matrix, axis=1)),
                'mean_pairwise_similarity': np.mean(cosine_similarity(embedding_matrix)),
                'embedding_variance': np.mean(np.var(embedding_matrix, axis=0))
            }
            
            quality_metrics[context] = metrics
        
        return quality_metrics
    
    def assess_integration_quality(self, integrated_networks: Dict[str, nx.Graph]) -> Dict:
        """Assess quality of integrated networks."""
        quality_metrics = {}
        
        for condition, network in integrated_networks.items():
            # Basic network metrics
            metrics = {
                'num_nodes': network.number_of_nodes(),
                'num_edges': network.number_of_edges(),
                'density': nx.density(network),
                'avg_clustering': nx.average_clustering(network) if network.number_of_nodes() > 0 else 0
            }
            
            # Edge type distribution
            edge_types = {}
            for _, _, data in network.edges(data=True):
                edge_type = data.get('edge_type', 'unknown')
                edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
            
            metrics['edge_type_distribution'] = edge_types
            
            # Node type distribution
            node_types = {}
            for node in network.nodes():
                node_type = network.nodes[node].get('node_type', 'unknown')
                node_types[node_type] = node_types.get(node_type, 0) + 1
            
            metrics['node_type_distribution'] = node_types
            
            # Integration-specific metrics
            regulatory_edges = edge_types.get('regulatory', 0)
            protein_edges = edge_types.get('protein_similarity', 0)
            cross_layer_edges = edge_types.get('cross_layer', 0)
            
            total_edges = network.number_of_edges()
            if total_edges > 0:
                metrics.update({
                    'regulatory_edge_fraction': regulatory_edges / total_edges,
                    'protein_edge_fraction': protein_edges / total_edges,
                    'cross_layer_edge_fraction': cross_layer_edges / total_edges
                })
            
            quality_metrics[condition] = metrics
        
        return quality_metrics


def cosine_similarity(X: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix."""
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
    return sklearn_cosine_similarity(X)

