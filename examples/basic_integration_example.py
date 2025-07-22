"""
Basic Integration Example: SCENIC+ and PINNACLE

This example demonstrates the basic workflow for integrating SCENIC+ 
regulatory networks with PINNACLE protein embeddings.

Requirements:
- SCENIC+ output files (pickle or CSV format)
- PINNACLE embedding files (pickle or NPZ format)
- Python environment with scenic_pinnacle package installed

Author: Manus AI
Date: July 2025
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd

# Add the package to path if running from source
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from scenic_pinnacle import ScenicPinnacleIntegrator

def main():
    """Run basic integration example."""
    
    print("SCENIC+ and PINNACLE Integration Example")
    print("=" * 50)
    
    # Configuration for the integration
    config = {
        'scenic': {
            'min_regulon_size': 10,
            'min_target_genes': 5,
            'importance_threshold': 0.1
        },
        'pinnacle': {
            'embedding_dim': 256,
            'normalize_embeddings': True
        },
        'integration': {
            'similarity_threshold': 0.5,
            'min_overlap': 3
        },
        'differential': {
            'pvalue_threshold': 0.05,
            'fold_change_threshold': 1.5
        }
    }
    
    # Initialize the integrator
    print("\n1. Initializing integrator...")
    integrator = ScenicPinnacleIntegrator(config=config)
    
    # Example data paths (replace with your actual data paths)
    scenic_data_path = "data/scenic_networks.pkl"
    pinnacle_data_path = "data/pinnacle_embeddings.pkl"
    output_dir = "results/basic_integration"
    
    # Check if example data exists
    if not Path(scenic_data_path).exists() or not Path(pinnacle_data_path).exists():
        print("\nGenerating example data...")
        generate_example_data(scenic_data_path, pinnacle_data_path)
    
    try:
        # Load SCENIC+ data
        print("\n2. Loading SCENIC+ data...")
        integrator.load_scenic_data(scenic_data_path, data_format='pickle')
        print(f"   Loaded {len(integrator.scenic_networks)} SCENIC+ networks")
        
        # Load PINNACLE data
        print("\n3. Loading PINNACLE data...")
        integrator.load_pinnacle_data(pinnacle_data_path, data_format='pickle')
        print(f"   Loaded embeddings for {len(integrator.pinnacle_embeddings)} contexts")
        
        # Integrate networks
        print("\n4. Integrating networks...")
        integrated_networks = integrator.integrate_networks()
        print(f"   Created {len(integrated_networks)} integrated networks")
        
        # Print network statistics
        for condition, network in integrated_networks.items():
            print(f"   {condition}: {network.number_of_nodes()} nodes, "
                  f"{network.number_of_edges()} edges")
        
        # Perform differential analysis (if multiple conditions)
        conditions = list(integrated_networks.keys())
        if len(conditions) >= 2:
            print("\n5. Performing differential analysis...")
            diff_results = integrator.differential_analysis(
                conditions[0], conditions[1], analysis_type='both'
            )
            
            # Print summary
            summary = diff_results.get('summary', {})
            for analysis_type, stats in summary.items():
                print(f"   {analysis_type.title()} changes:")
                for stat_name, value in stats.items():
                    print(f"     {stat_name}: {value}")
        
        # Generate visualizations
        print("\n6. Generating visualizations...")
        for condition in conditions[:2]:  # Limit to first 2 conditions
            viz_dir = Path(output_dir) / 'visualizations' / condition
            integrator.visualize_networks(condition, viz_dir)
            print(f"   Visualizations saved for {condition}")
        
        # Export results
        print("\n7. Exporting results...")
        integrator.export_results(Path(output_dir) / 'results')
        print(f"   Results exported to {output_dir}/results")
        
        # Run complete workflow (alternative approach)
        print("\n8. Running complete workflow...")
        summary = integrator.run_complete_workflow(
            scenic_path=scenic_data_path,
            pinnacle_path=pinnacle_data_path,
            output_dir=Path(output_dir) / 'complete_workflow',
            conditions=conditions[:2],
            comparisons=[(conditions[0], conditions[1])] if len(conditions) >= 2 else None
        )
        
        print("\nWorkflow Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        print("\n" + "=" * 50)
        print("Integration completed successfully!")
        print(f"Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"\nError during integration: {e}")
        print("Please check your data files and configuration.")
        return False
    
    return True


def generate_example_data(scenic_path: str, pinnacle_path: str):
    """Generate example data for demonstration."""
    import networkx as nx
    import pickle
    from pathlib import Path
    
    # Create data directory
    Path(scenic_path).parent.mkdir(parents=True, exist_ok=True)
    Path(pinnacle_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Generate example SCENIC+ networks
    scenic_networks = {}
    
    for condition in ['condition1', 'condition2', 'control']:
        G = nx.DiGraph()
        
        # Add TF nodes
        tfs = [f'TF_{i}' for i in range(1, 11)]
        for tf in tfs:
            G.add_node(tf, node_type='TF')
        
        # Add target nodes
        targets = [f'Gene_{i}' for i in range(1, 51)]
        for target in targets:
            G.add_node(target, node_type='target')
        
        # Add regulatory edges
        np.random.seed(42 + hash(condition) % 100)
        for tf in tfs:
            # Each TF regulates 5-15 targets
            n_targets = np.random.randint(5, 16)
            selected_targets = np.random.choice(targets, n_targets, replace=False)
            
            for target in selected_targets:
                importance = np.random.uniform(0.1, 1.0)
                G.add_edge(tf, target, 
                          importance=importance,
                          edge_type='regulation')
        
        scenic_networks[condition] = G
    
    # Save SCENIC+ networks
    with open(scenic_path, 'wb') as f:
        pickle.dump(scenic_networks, f)
    
    # Generate example PINNACLE embeddings
    pinnacle_embeddings = {}
    
    # Create protein list (corresponding to genes)
    all_genes = [f'TF_{i}' for i in range(1, 11)] + [f'Gene_{i}' for i in range(1, 51)]
    protein_ids = [f'Protein_{gene}' for gene in all_genes]
    
    for condition in ['condition1', 'condition2', 'control']:
        np.random.seed(42 + hash(condition) % 100)
        
        # Generate random embeddings
        embeddings = np.random.randn(len(protein_ids), 256)
        
        # Add some structure based on condition
        if condition == 'condition1':
            embeddings[:10] += 0.5  # TF proteins cluster
        elif condition == 'condition2':
            embeddings[10:] += 0.3  # Target proteins cluster
        
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        
        pinnacle_embeddings[condition] = {
            'embeddings': embeddings,
            'protein_ids': protein_ids,
            'embedding_dim': 256
        }
    
    # Save PINNACLE embeddings
    with open(pinnacle_path, 'wb') as f:
        pickle.dump(pinnacle_embeddings, f)
    
    print(f"Generated example SCENIC+ data: {len(scenic_networks)} networks")
    print(f"Generated example PINNACLE data: {len(pinnacle_embeddings)} embedding sets")


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

