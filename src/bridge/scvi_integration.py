"""
scVI Integration Module for BRIDGE Framework

This module provides integration with scVI (single-cell Variational Inference)
to enhance data preprocessing and representation learning for SCENIC+ and PINNACLE
analyses.

Key Features:
- Denoised expression matrices for better SCENIC+ input
- Robust latent representations for PINNACLE context
- Batch correction across experimental conditions
- Multi-modal integration (RNA + ATAC) with totalVI/MultiVI
- Enhanced peak-gene correlation for regulatory network inference
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from pathlib import Path
import warnings

# scVI imports with error handling
try:
    import scvi
    from scvi.model import SCVI, TOTALVI, MULTIVI
    from scvi.data import setup_anndata
    SCVI_AVAILABLE = True
except ImportError:
    SCVI_AVAILABLE = False
    warnings.warn("scVI not available. Install with: pip install scvi-tools")

class ScviPreprocessor:
    """
    scVI-based preprocessing for single-cell data.
    
    Provides denoising, batch correction, and latent representation
    learning to improve downstream SCENIC+ and PINNACLE analyses.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize scVI preprocessor.
        
        Args:
            config: Configuration parameters for scVI processing
        """
        if not SCVI_AVAILABLE:
            raise ImportError("scVI is required but not installed. Install with: pip install scvi-tools")
        
        self.config = config or {}
        self.logger = logging.getLogger(__name__ + '.ScviPreprocessor')
        
        # Default configuration
        self.default_config = {
            'model_type': 'scvi',  # 'scvi', 'totalvi', 'multivi'
            'n_latent': 30,
            'n_layers': 2,
            'n_hidden': 128,
            'dropout_rate': 0.1,
            'dispersion': 'gene',
            'gene_likelihood': 'nb',
            'max_epochs': 400,
            'early_stopping': True,
            'batch_size': 128,
            'train_size': 0.9,
            'use_gpu': True,
            'seed': 42
        }
        
        # Update with user config
        self.config = {**self.default_config, **self.config}
        
        # Set random seed
        scvi.settings.seed = self.config['seed']
        
        # Models storage
        self.models = {}
        self.processed_data = {}
        
    def preprocess_for_scenic(self, 
                            adata: anndata.AnnData,
                            condition_key: str = 'condition',
                            batch_key: Optional[str] = None) -> Dict[str, anndata.AnnData]:
        """
        Preprocess single-cell data for SCENIC+ analysis using scVI.
        
        Args:
            adata: AnnData object with single-cell expression data
            condition_key: Column in adata.obs indicating experimental conditions
            batch_key: Optional column for batch correction
            
        Returns:
            Dictionary of processed AnnData objects by condition
        """
        self.logger.info("Preprocessing data for SCENIC+ with scVI...")
        
        # Setup data for scVI
        if batch_key:
            setup_anndata(adata, batch_key=batch_key)
        else:
            setup_anndata(adata)
        
        # Train scVI model
        model = self._train_scvi_model(adata, model_type='scvi')
        
        # Get denoised expression
        denoised_expr = model.get_normalized_expression(
            adata, 
            return_mean=True,
            n_samples=25
        )
        
        # Get latent representation
        latent_repr = model.get_latents(adata)
        
        # Create processed AnnData objects by condition
        processed_data = {}
        conditions = adata.obs[condition_key].unique()
        
        for condition in conditions:
            condition_mask = adata.obs[condition_key] == condition
            condition_adata = adata[condition_mask].copy()
            
            # Replace expression with denoised version
            condition_adata.X = denoised_expr[condition_mask]
            
            # Add latent representation
            condition_adata.obsm['X_scvi'] = latent_repr[condition_mask]
            
            # Add scVI-specific metrics
            condition_adata.obs['scvi_batch_corrected'] = True
            condition_adata.uns['scvi_model_params'] = self.config
            
            processed_data[condition] = condition_adata
            
            self.logger.info(f"Processed {condition}: {condition_adata.n_obs} cells, "
                           f"{condition_adata.n_vars} genes")
        
        self.processed_data['scenic'] = processed_data
        return processed_data
    
    def preprocess_for_pinnacle(self,
                              adata: anndata.AnnData,
                              condition_key: str = 'condition',
                              batch_key: Optional[str] = None,
                              protein_genes: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Preprocess data for PINNACLE analysis using scVI latent representations.
        
        Args:
            adata: AnnData object with single-cell expression data
            condition_key: Column indicating experimental conditions
            batch_key: Optional column for batch correction
            protein_genes: List of genes encoding proteins of interest
            
        Returns:
            Dictionary of protein embeddings by condition
        """
        self.logger.info("Preprocessing data for PINNACLE with scVI...")
        
        # Use existing model if available, otherwise train new one
        if 'scenic' in self.processed_data:
            # Use latent representations from SCENIC preprocessing
            processed_scenic = self.processed_data['scenic']
            protein_embeddings = {}
            
            for condition, condition_adata in processed_scenic.items():
                latent_repr = condition_adata.obsm['X_scvi']
                
                # Create protein-level embeddings
                if protein_genes:
                    # Filter for protein-coding genes
                    protein_mask = condition_adata.var_names.isin(protein_genes)
                    protein_expr = condition_adata.X[:, protein_mask]
                    protein_ids = condition_adata.var_names[protein_mask].tolist()
                else:
                    # Use all genes as potential proteins
                    protein_expr = condition_adata.X
                    protein_ids = condition_adata.var_names.tolist()
                
                # Compute protein embeddings using latent space
                protein_embeddings[condition] = self._compute_protein_embeddings(
                    latent_repr, protein_expr, protein_ids
                )
                
                self.logger.info(f"Generated embeddings for {condition}: "
                               f"{len(protein_ids)} proteins, "
                               f"{latent_repr.shape[1]}D latent space")
        
        else:
            # Train new model specifically for PINNACLE
            if batch_key:
                setup_anndata(adata, batch_key=batch_key)
            else:
                setup_anndata(adata)
            
            model = self._train_scvi_model(adata, model_type='scvi')
            latent_repr = model.get_latents(adata)
            
            # Process by condition
            protein_embeddings = {}
            conditions = adata.obs[condition_key].unique()
            
            for condition in conditions:
                condition_mask = adata.obs[condition_key] == condition
                condition_latent = latent_repr[condition_mask]
                condition_adata = adata[condition_mask]
                
                if protein_genes:
                    protein_mask = condition_adata.var_names.isin(protein_genes)
                    protein_expr = condition_adata.X[:, protein_mask]
                    protein_ids = condition_adata.var_names[protein_mask].tolist()
                else:
                    protein_expr = condition_adata.X
                    protein_ids = condition_adata.var_names.tolist()
                
                protein_embeddings[condition] = self._compute_protein_embeddings(
                    condition_latent, protein_expr, protein_ids
                )
        
        self.processed_data['pinnacle'] = protein_embeddings
        return protein_embeddings
    
    def preprocess_multimodal(self,
                            rna_adata: anndata.AnnData,
                            atac_adata: Optional[anndata.AnnData] = None,
                            condition_key: str = 'condition',
                            batch_key: Optional[str] = None,
                            model_type: str = 'totalvi') -> Dict[str, Dict]:
        """
        Preprocess multi-modal data (RNA + ATAC) using totalVI or MultiVI.
        
        Args:
            rna_adata: RNA expression AnnData object
            atac_adata: Optional ATAC accessibility AnnData object
            condition_key: Column indicating experimental conditions
            batch_key: Optional column for batch correction
            model_type: 'totalvi' or 'multivi'
            
        Returns:
            Dictionary with processed data for both SCENIC+ and PINNACLE
        """
        self.logger.info(f"Preprocessing multi-modal data with {model_type}...")
        
        if model_type == 'totalvi' and atac_adata is None:
            raise ValueError("totalVI requires ATAC data")
        
        # Prepare multi-modal data
        if model_type == 'totalvi':
            # totalVI expects protein data in .obsm['protein_expression']
            # For ATAC, we'll use peak accessibility as "protein" data
            combined_adata = rna_adata.copy()
            combined_adata.obsm['protein_expression'] = atac_adata.X.toarray() if hasattr(atac_adata.X, 'toarray') else atac_adata.X
            
            # Setup for totalVI
            setup_anndata(
                combined_adata,
                protein_expression_obsm_key='protein_expression',
                batch_key=batch_key
            )
            
            model = self._train_scvi_model(combined_adata, model_type='totalvi')
            
        elif model_type == 'multivi':
            # MultiVI handles multiple modalities
            if atac_adata is not None:
                # Combine RNA and ATAC data
                combined_adata = self._combine_rna_atac(rna_adata, atac_adata)
            else:
                combined_adata = rna_adata.copy()
            
            setup_anndata(combined_adata, batch_key=batch_key)
            model = self._train_scvi_model(combined_adata, model_type='multivi')
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Extract processed data
        denoised_rna = model.get_normalized_expression(
            combined_adata,
            return_mean=True,
            n_samples=25
        )
        
        latent_repr = model.get_latents(combined_adata)
        
        # Process by condition for SCENIC+
        scenic_data = {}
        pinnacle_data = {}
        conditions = combined_adata.obs[condition_key].unique()
        
        for condition in conditions:
            condition_mask = combined_adata.obs[condition_key] == condition
            condition_adata = combined_adata[condition_mask].copy()
            
            # SCENIC+ data (denoised RNA)
            condition_adata.X = denoised_rna[condition_mask]
            condition_adata.obsm['X_scvi'] = latent_repr[condition_mask]
            
            # Add peak-gene correlations if ATAC available
            if atac_adata is not None:
                peak_gene_corr = self._compute_peak_gene_correlations(
                    condition_adata, atac_adata[condition_mask]
                )
                condition_adata.uns['peak_gene_correlations'] = peak_gene_corr
            
            scenic_data[condition] = condition_adata
            
            # PINNACLE data (protein embeddings from latent space)
            protein_ids = condition_adata.var_names.tolist()
            pinnacle_data[condition] = self._compute_protein_embeddings(
                latent_repr[condition_mask],
                denoised_rna[condition_mask],
                protein_ids
            )
        
        self.processed_data['scenic'] = scenic_data
        self.processed_data['pinnacle'] = pinnacle_data
        
        return {
            'scenic': scenic_data,
            'pinnacle': pinnacle_data,
            'model': model,
            'latent_representation': latent_repr
        }
    
    def _train_scvi_model(self, adata: anndata.AnnData, model_type: str = 'scvi'):
        """Train scVI model on the data."""
        self.logger.info(f"Training {model_type} model...")
        
        # Select model class
        if model_type == 'scvi':
            model_class = SCVI
        elif model_type == 'totalvi':
            model_class = TOTALVI
        elif model_type == 'multivi':
            model_class = MULTIVI
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Initialize model
        model = model_class(
            adata,
            n_latent=self.config['n_latent'],
            n_layers=self.config['n_layers'],
            n_hidden=self.config['n_hidden'],
            dropout_rate=self.config['dropout_rate'],
            dispersion=self.config['dispersion'],
            gene_likelihood=self.config['gene_likelihood']
        )
        
        # Train model
        model.train(
            max_epochs=self.config['max_epochs'],
            batch_size=self.config['batch_size'],
            train_size=self.config['train_size'],
            early_stopping=self.config['early_stopping'],
            use_gpu=self.config['use_gpu']
        )
        
        # Store model
        self.models[model_type] = model
        
        self.logger.info(f"{model_type} training completed")
        return model
    
    def _compute_protein_embeddings(self,
                                  latent_repr: np.ndarray,
                                  expression_data: np.ndarray,
                                  protein_ids: List[str]) -> Dict:
        """
        Compute protein embeddings from latent representations and expression.
        
        Args:
            latent_repr: Latent representation from scVI
            expression_data: Expression matrix
            protein_ids: List of protein identifiers
            
        Returns:
            Dictionary with protein embeddings in PINNACLE format
        """
        # Method 1: Use latent representation directly as protein context
        if self.config.get('use_latent_as_embedding', True):
            # Average latent representation across cells for each protein
            protein_embeddings = np.zeros((len(protein_ids), latent_repr.shape[1]))
            
            for i, protein_id in enumerate(protein_ids):
                # Weight latent representation by protein expression
                if hasattr(expression_data, 'toarray'):
                    expr_weights = expression_data[:, i].toarray().flatten()
                else:
                    expr_weights = expression_data[:, i]
                
                # Weighted average of latent representations
                if np.sum(expr_weights) > 0:
                    weights = expr_weights / np.sum(expr_weights)
                    protein_embeddings[i] = np.average(latent_repr, axis=0, weights=weights)
                else:
                    protein_embeddings[i] = np.mean(latent_repr, axis=0)
        
        # Method 2: Project expression to embedding space
        else:
            # Use expression data projected through latent space
            from sklearn.decomposition import PCA
            
            pca = PCA(n_components=min(256, latent_repr.shape[1]))
            if hasattr(expression_data, 'toarray'):
                expr_matrix = expression_data.toarray().T
            else:
                expr_matrix = expression_data.T
            
            protein_embeddings = pca.fit_transform(expr_matrix)
        
        # Normalize embeddings
        norms = np.linalg.norm(protein_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        protein_embeddings = protein_embeddings / norms
        
        return {
            'embeddings': protein_embeddings,
            'protein_ids': protein_ids,
            'embedding_dim': protein_embeddings.shape[1],
            'method': 'scvi_latent'
        }
    
    def _compute_peak_gene_correlations(self,
                                      rna_adata: anndata.AnnData,
                                      atac_adata: anndata.AnnData) -> pd.DataFrame:
        """Compute peak-gene correlations for enhanced SCENIC+ analysis."""
        from scipy.stats import pearsonr
        
        correlations = []
        
        # Get common cells
        common_cells = rna_adata.obs_names.intersection(atac_adata.obs_names)
        rna_subset = rna_adata[common_cells]
        atac_subset = atac_adata[common_cells]
        
        # Compute correlations between peaks and genes
        for i, gene in enumerate(rna_subset.var_names):
            gene_expr = rna_subset.X[:, i]
            if hasattr(gene_expr, 'toarray'):
                gene_expr = gene_expr.toarray().flatten()
            
            for j, peak in enumerate(atac_subset.var_names):
                peak_acc = atac_subset.X[:, j]
                if hasattr(peak_acc, 'toarray'):
                    peak_acc = peak_acc.toarray().flatten()
                
                # Compute correlation
                if np.var(gene_expr) > 0 and np.var(peak_acc) > 0:
                    corr, pval = pearsonr(gene_expr, peak_acc)
                    
                    correlations.append({
                        'gene': gene,
                        'peak': peak,
                        'correlation': corr,
                        'pvalue': pval
                    })
        
        return pd.DataFrame(correlations)
    
    def _combine_rna_atac(self,
                         rna_adata: anndata.AnnData,
                         atac_adata: anndata.AnnData) -> anndata.AnnData:
        """Combine RNA and ATAC data for MultiVI."""
        # This is a simplified combination - in practice, you might need
        # more sophisticated integration depending on your data structure
        
        # Find common cells
        common_cells = rna_adata.obs_names.intersection(atac_adata.obs_names)
        
        # Subset to common cells
        rna_subset = rna_adata[common_cells].copy()
        atac_subset = atac_adata[common_cells].copy()
        
        # Add modality information
        rna_subset.var['modality'] = 'RNA'
        atac_subset.var['modality'] = 'ATAC'
        
        # Concatenate (this is simplified - real implementation would be more complex)
        combined = anndata.concat([rna_subset, atac_subset], axis=1)
        
        return combined
    
    def get_quality_metrics(self) -> Dict:
        """Get quality metrics for scVI preprocessing."""
        metrics = {}
        
        for model_type, model in self.models.items():
            if hasattr(model, 'history'):
                metrics[model_type] = {
                    'final_train_loss': model.history['train_loss_epoch'][-1],
                    'final_validation_loss': model.history['validation_loss_epoch'][-1],
                    'n_epochs_trained': len(model.history['train_loss_epoch']),
                    'converged': model.history.get('early_stopping_epoch', None) is not None
                }
        
        return metrics
    
    def save_models(self, save_dir: Union[str, Path]) -> None:
        """Save trained scVI models."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for model_type, model in self.models.items():
            model_path = save_dir / f"{model_type}_model"
            model.save(model_path, overwrite=True)
            self.logger.info(f"Saved {model_type} model to {model_path}")
    
    def load_models(self, load_dir: Union[str, Path]) -> None:
        """Load trained scVI models."""
        load_dir = Path(load_dir)
        
        for model_file in load_dir.glob("*_model"):
            model_type = model_file.name.replace("_model", "")
            
            if model_type == 'scvi':
                model = SCVI.load(model_file)
            elif model_type == 'totalvi':
                model = TOTALVI.load(model_file)
            elif model_type == 'multivi':
                model = MULTIVI.load(model_file)
            else:
                continue
            
            self.models[model_type] = model
            self.logger.info(f"Loaded {model_type} model from {model_file}")


class ScviEnhancedBridge:
    """
    Enhanced BRIDGE integrator with scVI preprocessing capabilities.
    
    Combines scVI preprocessing with SCENIC+ and PINNACLE integration
    for improved multi-omic network analysis.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize enhanced BRIDGE with scVI integration."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__ + '.ScviEnhancedBridge')
        
        # Initialize scVI preprocessor
        scvi_config = self.config.get('scvi', {})
        self.scvi_preprocessor = ScviPreprocessor(config=scvi_config)
        
        # Will be initialized when needed
        self.bridge_integrator = None
    
    def preprocess_and_integrate(self,
                               rna_adata: anndata.AnnData,
                               atac_adata: Optional[anndata.AnnData] = None,
                               condition_key: str = 'condition',
                               batch_key: Optional[str] = None,
                               use_multimodal: bool = True) -> Dict:
        """
        Complete workflow: scVI preprocessing + BRIDGE integration.
        
        Args:
            rna_adata: RNA expression data
            atac_adata: Optional ATAC accessibility data
            condition_key: Column indicating conditions
            batch_key: Optional batch correction column
            use_multimodal: Whether to use multi-modal scVI models
            
        Returns:
            Complete analysis results including integrated networks
        """
        self.logger.info("Starting scVI-enhanced BRIDGE analysis...")
        
        # Step 1: scVI preprocessing
        if use_multimodal and atac_adata is not None:
            processed_data = self.scvi_preprocessor.preprocess_multimodal(
                rna_adata, atac_adata, condition_key, batch_key
            )
            scenic_data = processed_data['scenic']
            pinnacle_data = processed_data['pinnacle']
        else:
            # Separate preprocessing for RNA-only data
            scenic_data = self.scvi_preprocessor.preprocess_for_scenic(
                rna_adata, condition_key, batch_key
            )
            pinnacle_data = self.scvi_preprocessor.preprocess_for_pinnacle(
                rna_adata, condition_key, batch_key
            )
        
        # Step 2: Initialize BRIDGE integrator
        from .core import BridgeIntegrator
        
        bridge_config = self.config.get('bridge', {})
        self.bridge_integrator = BridgeIntegrator(config=bridge_config)
        
        # Step 3: Load processed data into BRIDGE
        self.bridge_integrator.scenic_networks = self._convert_scenic_data(scenic_data)
        self.bridge_integrator.pinnacle_embeddings = pinnacle_data
        
        # Step 4: Perform BRIDGE integration
        integrated_networks = self.bridge_integrator.integrate_networks()
        
        # Step 5: Return comprehensive results
        return {
            'integrated_networks': integrated_networks,
            'scvi_processed_scenic': scenic_data,
            'scvi_processed_pinnacle': pinnacle_data,
            'scvi_quality_metrics': self.scvi_preprocessor.get_quality_metrics(),
            'bridge_integrator': self.bridge_integrator
        }
    
    def _convert_scenic_data(self, scenic_data: Dict[str, anndata.AnnData]) -> Dict:
        """Convert scVI-processed AnnData to SCENIC+ network format."""
        # This would interface with actual SCENIC+ to generate networks
        # For now, we'll create placeholder networks
        
        import networkx as nx
        
        scenic_networks = {}
        
        for condition, adata in scenic_data.items():
            # Create regulatory network from processed expression data
            G = nx.DiGraph()
            
            # Add nodes (genes)
            for gene in adata.var_names:
                G.add_node(gene, node_type='gene')
            
            # Add edges based on expression correlations
            # (In practice, this would use SCENIC+ algorithm)
            expr_matrix = adata.X
            if hasattr(expr_matrix, 'toarray'):
                expr_matrix = expr_matrix.toarray()
            
            # Simple correlation-based network (placeholder)
            from scipy.stats import pearsonr
            
            for i, gene1 in enumerate(adata.var_names[:50]):  # Limit for demo
                for j, gene2 in enumerate(adata.var_names[:50]):
                    if i != j:
                        corr, pval = pearsonr(expr_matrix[:, i], expr_matrix[:, j])
                        if abs(corr) > 0.3 and pval < 0.05:
                            G.add_edge(gene1, gene2, 
                                     importance=abs(corr),
                                     edge_type='regulation')
            
            scenic_networks[condition] = G
        
        return scenic_networks

