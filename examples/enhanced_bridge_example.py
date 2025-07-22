"""
Enhanced BRIDGE Example: scVI + Optimal Transport Integration

This example demonstrates the enhanced BRIDGE framework with:
1. scVI preprocessing for denoising and batch correction
2. Optimal transport for cross-modal alignment
3. Complete multi-omic network integration
4. Trajectory analysis across conditions

Requirements:
- pip install scvi-tools POT moscot
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata
from pathlib import Path
import matplotlib.pyplot as plt

# BRIDGE imports
from bridge import (
    BridgeIntegrator, 
    ScviEnhancedBridge, 
    OTEnhancedBridge,
    OptimalTransportIntegrator
)

def create_synthetic_multimodal_data():
    """Create synthetic multi-modal single-cell data for demonstration."""
    print("Creating synthetic multi-modal data...")
    
    # Parameters
    n_cells_per_condition = 500
    n_genes = 2000
    n_peaks = 1500
    conditions = ['healthy', 'disease', 'treatment']
    
    # Create RNA data
    rna_data = {}
    atac_data = {}
    
    for i, condition in enumerate(conditions):
        # Generate RNA expression with condition-specific patterns
        np.random.seed(42 + i)
        
        # Base expression
        base_expr = np.random.negative_binomial(5, 0.3, (n_cells_per_condition, n_genes))
        
        # Add condition-specific effects
        if condition == 'disease':
            # Upregulate some genes
            disease_genes = np.random.choice(n_genes, 200, replace=False)
            base_expr[:, disease_genes] *= np.random.uniform(1.5, 3.0, len(disease_genes))
        elif condition == 'treatment':
            # Different pattern for treatment
            treatment_genes = np.random.choice(n_genes, 150, replace=False)
            base_expr[:, treatment_genes] *= np.random.uniform(0.5, 2.5, len(treatment_genes))
        
        # Create AnnData object
        gene_names = [f'Gene_{j}' for j in range(n_genes)]
        cell_names = [f'{condition}_cell_{j}' for j in range(n_cells_per_condition)]
        
        adata_rna = anndata.AnnData(
            X=base_expr,
            obs=pd.DataFrame({
                'condition': condition,
                'batch': f'batch_{i % 2}',  # Two batches
                'cell_type': np.random.choice(['TypeA', 'TypeB', 'TypeC'], n_cells_per_condition)
            }, index=cell_names),
            var=pd.DataFrame({'gene_name': gene_names}, index=gene_names)
        )
        
        rna_data[condition] = adata_rna
        
        # Generate ATAC data (correlated with RNA)
        atac_expr = np.random.negative_binomial(2, 0.4, (n_cells_per_condition, n_peaks))
        
        # Add some correlation with RNA
        for j in range(min(n_peaks, n_genes)):
            correlation = np.random.uniform(0.3, 0.7)
            noise = np.random.normal(0, 0.1, n_cells_per_condition)
            atac_expr[:, j] = (correlation * base_expr[:, j] + 
                              (1 - correlation) * atac_expr[:, j] + noise).astype(int)
            atac_expr[:, j] = np.maximum(atac_expr[:, j], 0)
        
        peak_names = [f'Peak_{j}' for j in range(n_peaks)]
        
        adata_atac = anndata.AnnData(
            X=atac_expr,
            obs=pd.DataFrame({
                'condition': condition,
                'batch': f'batch_{i % 2}',
                'cell_type': np.random.choice(['TypeA', 'TypeB', 'TypeC'], n_cells_per_condition)
            }, index=cell_names),
            var=pd.DataFrame({'peak_name': peak_names}, index=peak_names)
        )
        
        atac_data[condition] = adata_atac
    
    # Combine all conditions
    combined_rna = anndata.concat(list(rna_data.values()))
    combined_atac = anndata.concat(list(atac_data.values()))
    
    return combined_rna, combined_atac, rna_data, atac_data


def example_scvi_enhanced_integration():
    """Example of BRIDGE integration with scVI preprocessing."""
    print("\n" + "="*60)
    print("EXAMPLE 1: scVI-Enhanced BRIDGE Integration")
    print("="*60)
    
    # Create synthetic data
    combined_rna, combined_atac, rna_data, atac_data = create_synthetic_multimodal_data()
    
    # Initialize scVI-enhanced BRIDGE
    config = {
        'scvi': {
            'n_latent': 20,
            'max_epochs': 100,  # Reduced for demo
            'use_gpu': False,   # Set to True if GPU available
        },
        'bridge': {
            'scenic': {'min_regulon_size': 5},
            'pinnacle': {'normalize_embeddings': True}
        }
    }
    
    scvi_bridge = ScviEnhancedBridge(config=config)
    
    # Perform complete analysis
    print("Running scVI preprocessing and BRIDGE integration...")
    results = scvi_bridge.preprocess_and_integrate(
        rna_adata=combined_rna,
        atac_adata=combined_atac,
        condition_key='condition',
        batch_key='batch',
        use_multimodal=True
    )
    
    # Display results
    print(f"\nResults Summary:")
    print(f"- Integrated networks: {len(results['integrated_networks'])}")
    print(f"- Conditions processed: {list(results['integrated_networks'].keys())}")
    
    # Show scVI quality metrics
    scvi_metrics = results['scvi_quality_metrics']
    print(f"\nscVI Quality Metrics:")
    for model_type, metrics in scvi_metrics.items():
        print(f"  {model_type}:")
        for metric, value in metrics.items():
            print(f"    {metric}: {value}")
    
    return results


def example_optimal_transport_integration():
    """Example of BRIDGE integration with Optimal Transport."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Optimal Transport Enhanced BRIDGE")
    print("="*60)
    
    # Create synthetic data
    combined_rna, combined_atac, rna_data, atac_data = create_synthetic_multimodal_data()
    
    # First, create some mock SCENIC+ networks and PINNACLE embeddings
    print("Creating mock SCENIC+ networks and PINNACLE embeddings...")
    
    # Mock SCENIC+ networks
    import networkx as nx
    scenic_networks = {}
    
    for condition in ['healthy', 'disease', 'treatment']:
        G = nx.DiGraph()
        
        # Add nodes (genes)
        genes = [f'Gene_{i}' for i in range(100)]  # Subset for demo
        for gene in genes:
            G.add_node(gene, node_type='gene')
        
        # Add random regulatory edges
        np.random.seed(42)
        for i in range(200):  # 200 edges
            source = np.random.choice(genes)
            target = np.random.choice(genes)
            if source != target:
                importance = np.random.uniform(0.1, 0.9)
                G.add_edge(source, target, importance=importance, edge_type='regulation')
        
        scenic_networks[condition] = G
    
    # Mock PINNACLE embeddings
    pinnacle_embeddings = {}
    
    for condition in ['healthy', 'disease', 'treatment']:
        n_proteins = 80
        embedding_dim = 128
        
        # Generate random embeddings with condition-specific patterns
        np.random.seed(42)
        embeddings = np.random.randn(n_proteins, embedding_dim)
        
        # Add condition-specific shifts
        if condition == 'disease':
            embeddings += np.random.normal(0.2, 0.1, embeddings.shape)
        elif condition == 'treatment':
            embeddings += np.random.normal(-0.1, 0.15, embeddings.shape)
        
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        
        protein_ids = [f'Protein_{i}' for i in range(n_proteins)]
        
        pinnacle_embeddings[condition] = {
            'embeddings': embeddings,
            'protein_ids': protein_ids,
            'embedding_dim': embedding_dim
        }
    
    # Initialize OT-enhanced BRIDGE
    config = {
        'optimal_transport': {
            'reg': 0.1,
            'method': 'sinkhorn',
            'normalize': True
        },
        'bridge': {
            'scenic': {'min_regulon_size': 5},
            'pinnacle': {'normalize_embeddings': True}
        }
    }
    
    ot_bridge = OTEnhancedBridge(config=config)
    
    # Create simple gene-protein mapping
    gene_protein_mapping = {f'Gene_{i}': f'Protein_{i}' for i in range(80)}
    
    # Perform OT-enhanced integration
    print("Running optimal transport alignment and integration...")
    results = ot_bridge.integrate_with_ot_alignment(
        scenic_networks=scenic_networks,
        pinnacle_embeddings=pinnacle_embeddings,
        gene_protein_mapping=gene_protein_mapping
    )
    
    # Display results
    print(f"\nOT Integration Results:")
    print(f"- Integrated networks: {len(results['integrated_networks'])}")
    print(f"- OT alignments computed: {len(results['ot_alignment_results'])}")
    print(f"- Condition comparisons: {len(results['ot_condition_comparisons'])}")
    
    # Show OT alignment quality
    print(f"\nOT Alignment Quality:")
    for condition, alignment in results['ot_alignment_results'].items():
        quality = alignment['alignment_quality']
        print(f"  {condition}:")
        print(f"    Wasserstein distance: {alignment['wasserstein_distance']:.4f}")
        print(f"    Transport cost: {quality['transport_cost']:.4f}")
        print(f"    Sparsity: {quality['sparsity']:.4f}")
    
    # Show condition comparisons
    print(f"\nCondition Comparisons (Wasserstein distances):")
    for comparison, result in results['ot_condition_comparisons'].items():
        joint_dist = result['joint_comparison']['joint_wasserstein_distance']
        print(f"  {comparison}: {joint_dist:.4f}")
    
    return results


def example_trajectory_analysis():
    """Example of trajectory analysis with optimal transport."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Trajectory Analysis with Optimal Transport")
    print("="*60)
    
    # Create time-series data
    print("Creating time-series multi-modal data...")
    
    time_points = ['T0', 'T1', 'T2', 'T3', 'T4']
    time_values = [0, 6, 12, 24, 48]  # Hours
    
    # Create mock networks and embeddings for trajectory
    scenic_networks = {}
    pinnacle_embeddings = {}
    
    for i, time_point in enumerate(time_points):
        # Create network with evolving structure
        G = nx.DiGraph()
        genes = [f'Gene_{j}' for j in range(50)]
        
        for gene in genes:
            G.add_node(gene, node_type='gene')
        
        # Add edges with time-dependent patterns
        np.random.seed(42 + i)
        n_edges = 80 + i * 20  # Increasing connectivity over time
        
        for _ in range(n_edges):
            source = np.random.choice(genes)
            target = np.random.choice(genes)
            if source != target:
                # Importance changes over time
                base_importance = np.random.uniform(0.1, 0.9)
                time_effect = 0.1 * i * np.random.uniform(-1, 1)
                importance = np.clip(base_importance + time_effect, 0.1, 0.9)
                
                G.add_edge(source, target, importance=importance, edge_type='regulation')
        
        scenic_networks[time_point] = G
        
        # Create embeddings with temporal evolution
        n_proteins = 40
        embedding_dim = 64
        
        np.random.seed(42)
        base_embeddings = np.random.randn(n_proteins, embedding_dim)
        
        # Add temporal drift
        temporal_drift = 0.2 * i * np.random.randn(n_proteins, embedding_dim)
        embeddings = base_embeddings + temporal_drift
        
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        
        protein_ids = [f'Protein_{j}' for j in range(n_proteins)]
        
        pinnacle_embeddings[time_point] = {
            'embeddings': embeddings,
            'protein_ids': protein_ids,
            'embedding_dim': embedding_dim
        }
    
    # Initialize OT integrator
    ot_integrator = OptimalTransportIntegrator(config={'reg': 0.05})
    
    # Perform trajectory analysis
    print("Running trajectory analysis...")
    trajectory_results = ot_integrator.trajectory_analysis(
        scenic_networks=scenic_networks,
        pinnacle_embeddings=pinnacle_embeddings,
        condition_order=time_points,
        time_points=time_values
    )
    
    # Display results
    print(f"\nTrajectory Analysis Results:")
    analysis = trajectory_results['trajectory_analysis']
    print(f"- Total trajectory cost: {analysis['total_cost']:.4f}")
    print(f"- Mean step cost: {analysis['mean_cost']:.4f}")
    print(f"- Mean velocity: {analysis['mean_velocity']:.4f}")
    print(f"- Trajectory is monotonic: {analysis['monotonic']}")
    print(f"- Critical points: {analysis['critical_conditions']}")
    
    # Plot trajectory if matplotlib available
    try:
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(time_values[:-1], trajectory_results['trajectory_costs'], 'o-')
        plt.xlabel('Time (hours)')
        plt.ylabel('Transport Cost')
        plt.title('Network Evolution Trajectory')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        if analysis['velocities']:
            plt.plot(time_values[:-2], analysis['velocities'], 's-', color='red')
            plt.xlabel('Time (hours)')
            plt.ylabel('Velocity (cost/time)')
            plt.title('Trajectory Velocity')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('trajectory_analysis.png', dpi=150, bbox_inches='tight')
        print("Trajectory plot saved as 'trajectory_analysis.png'")
        
    except ImportError:
        print("Matplotlib not available for plotting")
    
    return trajectory_results


def example_complete_workflow():
    """Complete workflow combining all enhancements."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Complete Enhanced BRIDGE Workflow")
    print("="*60)
    
    # Create synthetic data
    combined_rna, combined_atac, rna_data, atac_data = create_synthetic_multimodal_data()
    
    print("Step 1: scVI preprocessing...")
    
    # Configure enhanced BRIDGE
    config = {
        'scvi': {
            'n_latent': 15,
            'max_epochs': 50,  # Reduced for demo
            'use_gpu': False,
        },
        'optimal_transport': {
            'reg': 0.1,
            'method': 'fused_gromov_wasserstein',
            'normalize': True
        },
        'bridge': {
            'scenic': {'min_regulon_size': 3},
            'pinnacle': {'normalize_embeddings': True}
        }
    }
    
    # Initialize enhanced BRIDGE
    from bridge.scvi_integration import ScviPreprocessor
    from bridge.optimal_transport import OptimalTransportIntegrator
    
    # Step 1: scVI preprocessing
    scvi_preprocessor = ScviPreprocessor(config=config['scvi'])
    
    processed_data = scvi_preprocessor.preprocess_multimodal(
        rna_adata=combined_rna,
        atac_adata=combined_atac,
        condition_key='condition',
        batch_key='batch',
        model_type='totalvi'
    )
    
    print("Step 2: Converting to SCENIC+/PINNACLE format...")
    
    # Extract processed data
    scenic_data = processed_data['scenic']
    pinnacle_data = processed_data['pinnacle']
    
    # Convert to network format (simplified)
    scenic_networks = {}
    for condition, adata in scenic_data.items():
        G = nx.DiGraph()
        
        # Add top variable genes as nodes
        top_genes = adata.var_names[:50]  # Top 50 for demo
        for gene in top_genes:
            G.add_node(gene, node_type='gene')
        
        # Add edges based on correlation (simplified SCENIC+ simulation)
        expr_data = adata.X
        if hasattr(expr_data, 'toarray'):
            expr_data = expr_data.toarray()
        
        from scipy.stats import pearsonr
        
        for i, gene1 in enumerate(top_genes[:20]):  # Limit for demo
            for j, gene2 in enumerate(top_genes[:20]):
                if i != j:
                    gene1_idx = list(adata.var_names).index(gene1)
                    gene2_idx = list(adata.var_names).index(gene2)
                    
                    corr, pval = pearsonr(expr_data[:, gene1_idx], expr_data[:, gene2_idx])
                    if abs(corr) > 0.4 and pval < 0.05:
                        G.add_edge(gene1, gene2, importance=abs(corr), edge_type='regulation')
        
        scenic_networks[condition] = G
    
    print("Step 3: Optimal transport integration...")
    
    # Initialize OT integrator
    ot_integrator = OptimalTransportIntegrator(config=config['optimal_transport'])
    
    # Create gene-protein mapping
    gene_protein_mapping = {}
    for condition in pinnacle_data:
        proteins = pinnacle_data[condition]['protein_ids'][:30]  # Subset
        for i, protein in enumerate(proteins):
            gene_name = protein.replace('Protein_', 'Gene_')
            if gene_name in scenic_networks[condition].nodes():
                gene_protein_mapping[gene_name] = protein
    
    # Perform cross-modal alignment
    alignment_results = ot_integrator.align_cross_modal(
        scenic_networks=scenic_networks,
        pinnacle_embeddings=pinnacle_data,
        gene_protein_mapping=gene_protein_mapping
    )
    
    print("Step 4: Final BRIDGE integration...")
    
    # Initialize standard BRIDGE integrator
    bridge_integrator = BridgeIntegrator(config=config['bridge'])
    bridge_integrator.scenic_networks = scenic_networks
    bridge_integrator.pinnacle_embeddings = pinnacle_data
    
    # Add OT alignment information
    bridge_integrator.alignment_info = alignment_results
    
    # Perform final integration
    integrated_networks = bridge_integrator.integrate_networks()
    
    print("Step 5: Results summary...")
    
    # Comprehensive results
    results = {
        'scvi_processed_data': processed_data,
        'scenic_networks': scenic_networks,
        'pinnacle_embeddings': pinnacle_data,
        'ot_alignment_results': alignment_results,
        'integrated_networks': integrated_networks,
        'scvi_quality_metrics': scvi_preprocessor.get_quality_metrics(),
        'ot_summary': ot_integrator.get_transport_summary()
    }
    
    # Display comprehensive summary
    print(f"\nComplete Workflow Results:")
    print(f"- Conditions processed: {list(integrated_networks.keys())}")
    print(f"- Networks integrated: {len(integrated_networks)}")
    
    for condition in integrated_networks:
        network = integrated_networks[condition]
        print(f"  {condition}: {network.number_of_nodes()} nodes, {network.number_of_edges()} edges")
    
    # OT alignment quality
    print(f"\nOT Alignment Quality:")
    for condition, alignment in alignment_results.items():
        w_dist = alignment['wasserstein_distance']
        print(f"  {condition}: Wasserstein distance = {w_dist:.4f}")
    
    # scVI quality
    scvi_metrics = results['scvi_quality_metrics']
    if scvi_metrics:
        print(f"\nscVI Model Quality:")
        for model_type, metrics in scvi_metrics.items():
            if 'final_train_loss' in metrics:
                print(f"  {model_type}: Final loss = {metrics['final_train_loss']:.4f}")
    
    return results


if __name__ == "__main__":
    print("Enhanced BRIDGE Framework Examples")
    print("==================================")
    
    # Run examples
    try:
        # Example 1: scVI-enhanced integration
        scvi_results = example_scvi_enhanced_integration()
        
        # Example 2: Optimal transport integration
        ot_results = example_optimal_transport_integration()
        
        # Example 3: Trajectory analysis
        trajectory_results = example_trajectory_analysis()
        
        # Example 4: Complete workflow
        complete_results = example_complete_workflow()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install scvi-tools POT moscot")

