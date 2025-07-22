"""
BRIDGE-Velocity: Drug Response and Pharmacology Analysis Example

This example demonstrates the velocity-enhanced BRIDGE framework for:
1. Drug response network analysis with RNA velocity
2. Identification of perturbed regulatory pathways
3. Drug target discovery through velocity-network integration
4. Dose-response relationship analysis
5. Pharmacological profiling and target prioritization

Requirements:
- pip install scvelo cellrank velocyto
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import networkx as nx
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# BRIDGE imports
from bridge import BridgeIntegrator
from bridge.velocity_integration import VelocityEnhancedBridge, VelocityNetworkAnalyzer

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)

def create_drug_response_data():
    """Create synthetic drug response data with velocity information."""
    print("Creating synthetic drug response data with RNA velocity...")
    
    # Parameters
    n_cells_per_condition = 400
    n_genes = 1500
    conditions = ['control', 'drug_low', 'drug_medium', 'drug_high', 'drug_combo']
    doses = [0, 1, 5, 25, 100]  # Drug concentrations
    
    # Create time-course data
    time_points = [0, 2, 6, 12, 24]  # Hours
    
    all_data = {}
    
    for i, (condition, dose) in enumerate(zip(conditions, doses)):
        print(f"Generating data for {condition} (dose: {dose})")
        
        # Generate spliced and unspliced counts
        np.random.seed(42 + i)
        
        # Base expression levels
        base_spliced = np.random.negative_binomial(8, 0.3, (n_cells_per_condition, n_genes))
        base_unspliced = np.random.negative_binomial(5, 0.4, (n_cells_per_condition, n_genes))
        
        # Add dose-dependent effects
        if dose > 0:
            # Drug affects specific gene sets
            drug_responsive_genes = np.random.choice(n_genes, int(0.2 * n_genes), replace=False)
            
            # Dose-response relationship
            dose_effect = np.log1p(dose) / np.log1p(100)  # Normalize to 0-1
            
            for gene_idx in drug_responsive_genes:
                # Some genes upregulated, some downregulated
                if np.random.random() > 0.5:
                    effect_multiplier = 1 + dose_effect * np.random.uniform(0.5, 2.0)
                else:
                    effect_multiplier = 1 - dose_effect * np.random.uniform(0.2, 0.8)
                
                base_spliced[:, gene_idx] = (base_spliced[:, gene_idx] * effect_multiplier).astype(int)
                base_unspliced[:, gene_idx] = (base_unspliced[:, gene_idx] * effect_multiplier).astype(int)
        
        # Add velocity-related patterns
        # Simulate transcriptional dynamics
        for cell_idx in range(n_cells_per_condition):
            # Add cell-specific velocity patterns
            velocity_strength = np.random.uniform(0.1, 0.8)
            
            # Some genes have higher velocity (active transcription)
            high_velocity_genes = np.random.choice(n_genes, int(0.1 * n_genes), replace=False)
            
            for gene_idx in high_velocity_genes:
                # Higher unspliced relative to spliced indicates active transcription
                velocity_factor = 1 + velocity_strength * np.random.uniform(0.5, 1.5)
                base_unspliced[cell_idx, gene_idx] = int(base_unspliced[cell_idx, gene_idx] * velocity_factor)
        
        # Create cell and gene names
        cell_names = [f'{condition}_cell_{j}' for j in range(n_cells_per_condition)]
        gene_names = [f'Gene_{j}' for j in range(n_genes)]
        
        # Create AnnData object
        adata = anndata.AnnData(
            X=base_spliced,
            obs=pd.DataFrame({
                'condition': condition,
                'dose': dose,
                'time_point': np.random.choice(time_points, n_cells_per_condition),
                'cell_type': np.random.choice(['TypeA', 'TypeB', 'TypeC'], n_cells_per_condition),
                'batch': f'batch_{i % 2}'
            }, index=cell_names),
            var=pd.DataFrame({
                'gene_name': gene_names,
                'highly_variable': np.random.choice([True, False], n_genes, p=[0.3, 0.7])
            }, index=gene_names)
        )
        
        # Add spliced and unspliced layers
        adata.layers['spliced'] = base_spliced
        adata.layers['unspliced'] = base_unspliced
        
        # Add some metadata
        adata.uns['condition'] = condition
        adata.uns['dose'] = dose
        
        all_data[condition] = adata
    
    # Combine all conditions
    combined_adata = anndata.concat(list(all_data.values()))
    
    return combined_adata, all_data


def create_drug_response_networks():
    """Create synthetic SCENIC+ networks for drug response conditions."""
    print("Creating drug response regulatory networks...")
    
    conditions = ['control', 'drug_low', 'drug_medium', 'drug_high', 'drug_combo']
    networks = {}
    
    # Base network structure
    n_genes = 100  # Subset for network analysis
    base_genes = [f'Gene_{i}' for i in range(n_genes)]
    
    for i, condition in enumerate(conditions):
        print(f"Creating network for {condition}")
        
        G = nx.DiGraph()
        
        # Add nodes
        for gene in base_genes:
            G.add_node(gene, node_type='gene')
        
        # Create base regulatory structure
        np.random.seed(42 + i)
        
        # Add transcription factor -> target relationships
        n_tfs = 20
        tfs = np.random.choice(base_genes, n_tfs, replace=False)
        
        for tf in tfs:
            # Each TF regulates 5-15 targets
            n_targets = np.random.randint(5, 16)
            targets = np.random.choice([g for g in base_genes if g != tf], n_targets, replace=False)
            
            for target in targets:
                # Base importance
                importance = np.random.uniform(0.3, 0.9)
                
                # Modify importance based on drug condition
                if condition != 'control':
                    # Drug affects some regulatory relationships
                    if np.random.random() < 0.3:  # 30% of edges affected
                        if 'drug' in condition:
                            # Drug can strengthen or weaken regulation
                            dose_effect = ['low', 'medium', 'high', 'combo'].index(condition.split('_')[1]) + 1
                            effect_strength = dose_effect * 0.1
                            
                            if np.random.random() > 0.5:
                                importance = min(importance + effect_strength, 0.95)
                            else:
                                importance = max(importance - effect_strength, 0.1)
                
                G.add_edge(tf, target, importance=importance, edge_type='regulation')
        
        # Add some drug-specific edges for treated conditions
        if condition != 'control':
            # Add new regulatory relationships induced by drug
            n_new_edges = np.random.randint(10, 25)
            
            for _ in range(n_new_edges):
                source = np.random.choice(base_genes)
                target = np.random.choice(base_genes)
                
                if source != target and not G.has_edge(source, target):
                    importance = np.random.uniform(0.2, 0.7)
                    G.add_edge(source, target, importance=importance, edge_type='drug_induced')
        
        networks[condition] = G
        print(f"  {condition}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    return networks


def create_drug_response_protein_embeddings():
    """Create synthetic PINNACLE protein embeddings for drug conditions."""
    print("Creating drug response protein embeddings...")
    
    conditions = ['control', 'drug_low', 'drug_medium', 'drug_high', 'drug_combo']
    embeddings = {}
    
    n_proteins = 80
    embedding_dim = 128
    protein_ids = [f'Protein_{i}' for i in range(n_proteins)]
    
    # Base embeddings
    np.random.seed(42)
    base_embeddings = np.random.randn(n_proteins, embedding_dim)
    base_embeddings = base_embeddings / np.linalg.norm(base_embeddings, axis=1, keepdims=True)
    
    for i, condition in enumerate(conditions):
        print(f"Creating embeddings for {condition}")
        
        # Start with base embeddings
        condition_embeddings = base_embeddings.copy()
        
        if condition != 'control':
            # Add drug-specific perturbations
            dose_level = ['low', 'medium', 'high', 'combo'].index(condition.split('_')[1]) + 1
            perturbation_strength = dose_level * 0.1
            
            # Perturb subset of proteins
            n_perturbed = int(0.4 * n_proteins)  # 40% of proteins affected
            perturbed_indices = np.random.choice(n_proteins, n_perturbed, replace=False)
            
            for idx in perturbed_indices:
                # Add dose-dependent perturbation
                perturbation = np.random.randn(embedding_dim) * perturbation_strength
                condition_embeddings[idx] += perturbation
            
            # Renormalize
            condition_embeddings = condition_embeddings / np.linalg.norm(
                condition_embeddings, axis=1, keepdims=True
            )
        
        embeddings[condition] = {
            'embeddings': condition_embeddings,
            'protein_ids': protein_ids,
            'embedding_dim': embedding_dim
        }
    
    return embeddings


def example_basic_drug_response_analysis():
    """Example of basic drug response analysis with velocity."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Drug Response Analysis with RNA Velocity")
    print("="*70)
    
    # Create synthetic data
    combined_adata, individual_data = create_drug_response_data()
    networks = create_drug_response_networks()
    protein_embeddings = create_drug_response_protein_embeddings()
    
    # Initialize velocity-enhanced BRIDGE
    config = {
        'velocity': {
            'mode': 'dynamical',
            'n_top_genes': 1000,
            'n_neighbors': 20
        },
        'bridge': {
            'scenic': {'min_regulon_size': 3},
            'pinnacle': {'normalize_embeddings': True}
        }
    }
    
    velocity_bridge = VelocityEnhancedBridge(config=config)
    
    # Define drug conditions
    drug_conditions = ['drug_low', 'drug_medium', 'drug_high', 'drug_combo']
    control_condition = 'control'
    
    # Run complete drug response analysis
    print("Running velocity-enhanced drug response analysis...")
    results = velocity_bridge.analyze_drug_response_networks(
        rna_adata=combined_adata,
        scenic_networks=networks,
        pinnacle_embeddings=protein_embeddings,
        drug_conditions=drug_conditions,
        control_condition=control_condition,
        condition_key='condition'
    )
    
    # Display results
    print(f"\nDrug Response Analysis Results:")
    print(f"- Conditions analyzed: {len(results['drug_response_results'])}")
    print(f"- Networks integrated: {len(results['integrated_networks'])}")
    
    # Show drug response summary for each condition
    for drug_condition in drug_conditions:
        if drug_condition in results['drug_response_results']:
            drug_result = results['drug_response_results'][drug_condition]
            
            print(f"\n{drug_condition.upper()}:")
            print(f"  Network perturbation score: {drug_result['network_perturbation_score']:.3f}")
            
            if 'perturbed_modules' in drug_result:
                perturbed = drug_result['perturbed_modules']
                print(f"  Edges lost: {len(perturbed.get('lost_edges', []))}")
                print(f"  Edges gained: {len(perturbed.get('gained_edges', []))}")
                print(f"  Edge change ratio: {perturbed.get('edge_change_ratio', 0):.3f}")
            
            if 'drug_targets' in drug_result:
                targets = drug_result['drug_targets']
                print(f"  Regulatory targets: {len(targets.get('regulatory_targets', []))}")
                print(f"  Protein targets: {len(targets.get('protein_targets', []))}")
                print(f"  Combined targets: {len(targets.get('combined_targets', []))}")
    
    # Show pharmacological targets
    if 'pharmacological_targets' in results:
        print(f"\nPharmacological Target Summary:")
        for drug_condition, target_info in results['pharmacological_targets'].items():
            top_targets = target_info.get('top_targets', [])[:5]  # Top 5
            print(f"  {drug_condition}: {[target[0] for target in top_targets]}")
    
    return results


def example_dose_response_analysis():
    """Example of dose-response relationship analysis."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Dose-Response Network Dynamics Analysis")
    print("="*70)
    
    # Create dose-response data
    combined_adata, individual_data = create_drug_response_data()
    networks = create_drug_response_networks()
    
    # Initialize velocity analyzer
    velocity_analyzer = VelocityNetworkAnalyzer()
    
    # Define dose conditions and values
    dose_conditions = ['drug_low', 'drug_medium', 'drug_high']
    doses = [1, 5, 25]
    control_condition = 'control'
    
    # Compute velocity
    print("Computing RNA velocity for dose-response analysis...")
    velocity_adata = velocity_analyzer.compute_rna_velocity(
        combined_adata, condition_key='condition'
    )
    
    # Analyze dose-response dynamics
    print("Analyzing dose-response network dynamics...")
    dose_response_results = velocity_analyzer.analyze_dose_response_dynamics(
        networks, velocity_adata, dose_conditions, doses, control_condition
    )
    
    # Display results
    print(f"\nDose-Response Analysis Results:")
    print(f"Doses analyzed: {dose_response_results['doses']}")
    print(f"Conditions: {dose_response_results['conditions']}")
    
    # Show network metrics across doses
    print(f"\nNetwork Metrics Across Doses:")
    for dose in dose_response_results['doses']:
        if dose in dose_response_results['network_metrics']:
            metrics = dose_response_results['network_metrics'][dose]
            print(f"  Dose {dose}:")
            print(f"    Nodes: {metrics['n_nodes']}")
            print(f"    Edges: {metrics['n_edges']}")
            print(f"    Density: {metrics['density']:.3f}")
            print(f"    Avg clustering: {metrics['avg_clustering']:.3f}")
    
    # Show dose-response curves
    if 'dose_response_curves' in dose_response_results:
        print(f"\nDose-Response Curve Fitting:")
        for metric, curve_info in dose_response_results['dose_response_curves'].items():
            if curve_info['model'] != 'failed':
                r_squared = curve_info.get('r_squared', 0)
                print(f"  {metric}: R² = {r_squared:.3f}")
    
    # Plot dose-response relationships
    try:
        plot_dose_response_curves(dose_response_results)
    except Exception as e:
        print(f"Plotting failed: {e}")
    
    return dose_response_results


def example_drug_target_discovery():
    """Example of drug target discovery through velocity-network integration."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Drug Target Discovery with Velocity Integration")
    print("="*70)
    
    # Create data
    combined_adata, individual_data = create_drug_response_data()
    networks = create_drug_response_networks()
    protein_embeddings = create_drug_response_protein_embeddings()
    
    # Initialize analyzer
    velocity_analyzer = VelocityNetworkAnalyzer()
    
    # Compute velocity
    velocity_adata = velocity_analyzer.compute_rna_velocity(
        combined_adata, condition_key='condition'
    )
    
    # Focus on high-dose drug condition
    drug_condition = 'drug_high'
    control_condition = 'control'
    
    # Identify drug response networks
    print(f"Identifying drug response networks for {drug_condition}...")
    drug_response_results = velocity_analyzer.identify_drug_response_networks(
        networks, protein_embeddings, velocity_adata,
        [drug_condition], control_condition
    )
    
    # Identify pharmacological targets
    print("Identifying pharmacological targets...")
    pharmacological_targets = velocity_analyzer.identify_pharmacological_targets(
        drug_response_results
    )
    
    # Display target discovery results
    if drug_condition in pharmacological_targets:
        target_info = pharmacological_targets[drug_condition]
        
        print(f"\nDrug Target Discovery Results for {drug_condition}:")
        
        # Top targets
        top_targets = target_info.get('top_targets', [])
        print(f"Top 10 Drug Targets:")
        for i, (target, score) in enumerate(top_targets[:10], 1):
            print(f"  {i:2d}. {target}: {score:.3f}")
        
        # Target categories
        regulatory_targets = target_info.get('known_targets', {})
        novel_targets = target_info.get('novel_targets', [])
        
        print(f"\nTarget Categories:")
        print(f"  Known targets: {len(regulatory_targets)}")
        print(f"  Novel targets: {len(novel_targets)}")
        
        if novel_targets:
            print(f"  Top novel targets: {novel_targets[:5]}")
    
    # Analyze network perturbations
    if drug_condition in drug_response_results:
        drug_result = drug_response_results[drug_condition]
        
        print(f"\nNetwork Perturbation Analysis:")
        print(f"  Overall perturbation score: {drug_result['network_perturbation_score']:.3f}")
        
        if 'perturbed_modules' in drug_result:
            perturbed = drug_result['perturbed_modules']
            print(f"  Nodes with changes: {len(perturbed.get('nodes_with_changes', []))}")
            
            # Show most perturbed nodes
            perturbed_nodes = perturbed.get('nodes_with_changes', [])
            if perturbed_nodes:
                print(f"  Most perturbed nodes: {perturbed_nodes[:10]}")
        
        if 'protein_changes' in drug_result:
            protein_changes = drug_result['protein_changes']
            print(f"  Proteins with significant changes: {len(protein_changes.get('most_changed_proteins', []))}")
            print(f"  Mean protein change magnitude: {protein_changes.get('mean_change_magnitude', 0):.3f}")
    
    return pharmacological_targets


def example_temporal_drug_response():
    """Example of temporal drug response analysis."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Temporal Drug Response Trajectory Analysis")
    print("="*70)
    
    # Create time-course drug response data
    print("Creating temporal drug response data...")
    
    # Simulate drug treatment over time
    time_points = ['T0_control', 'T2_drug', 'T6_drug', 'T12_drug', 'T24_drug']
    time_values = [0, 2, 6, 12, 24]
    
    # Create networks for each time point
    temporal_networks = {}
    temporal_embeddings = {}
    
    n_genes = 80
    n_proteins = 60
    base_genes = [f'Gene_{i}' for i in range(n_genes)]
    protein_ids = [f'Protein_{i}' for i in range(n_proteins)]
    
    for i, (time_point, time_val) in enumerate(zip(time_points, time_values)):
        print(f"Creating network for {time_point}")
        
        # Create network with temporal evolution
        G = nx.DiGraph()
        
        for gene in base_genes:
            G.add_node(gene, node_type='gene')
        
        # Add edges with time-dependent changes
        np.random.seed(42 + i)
        
        # Base regulatory structure
        n_edges = 120 + i * 20  # Increasing complexity over time
        
        for _ in range(n_edges):
            source = np.random.choice(base_genes)
            target = np.random.choice(base_genes)
            
            if source != target:
                # Importance changes over time
                base_importance = np.random.uniform(0.2, 0.8)
                
                # Drug effect increases over time
                if time_val > 0:
                    drug_effect = (time_val / 24) * np.random.uniform(-0.3, 0.5)
                    importance = np.clip(base_importance + drug_effect, 0.1, 0.9)
                else:
                    importance = base_importance
                
                G.add_edge(source, target, importance=importance, edge_type='regulation')
        
        temporal_networks[time_point] = G
        
        # Create protein embeddings with temporal drift
        embedding_dim = 64
        np.random.seed(42)
        base_embeddings = np.random.randn(n_proteins, embedding_dim)
        
        # Add temporal drift
        if time_val > 0:
            temporal_drift = (time_val / 24) * 0.3 * np.random.randn(n_proteins, embedding_dim)
            embeddings = base_embeddings + temporal_drift
        else:
            embeddings = base_embeddings
        
        # Normalize
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        temporal_embeddings[time_point] = {
            'embeddings': embeddings,
            'protein_ids': protein_ids,
            'embedding_dim': embedding_dim
        }
    
    # Create temporal AnnData
    n_cells_per_time = 200
    all_cells_data = []
    
    for time_point, time_val in zip(time_points, time_values):
        # Create expression data
        n_genes_expr = 500
        spliced = np.random.negative_binomial(6, 0.3, (n_cells_per_time, n_genes_expr))
        unspliced = np.random.negative_binomial(4, 0.4, (n_cells_per_time, n_genes_expr))
        
        # Add temporal patterns
        if time_val > 0:
            # Drug response genes
            drug_genes = np.random.choice(n_genes_expr, int(0.3 * n_genes_expr), replace=False)
            time_effect = time_val / 24
            
            for gene_idx in drug_genes:
                effect = 1 + time_effect * np.random.uniform(-0.5, 1.0)
                spliced[:, gene_idx] = (spliced[:, gene_idx] * effect).astype(int)
                unspliced[:, gene_idx] = (unspliced[:, gene_idx] * effect).astype(int)
        
        # Create AnnData for this time point
        cell_names = [f'{time_point}_cell_{j}' for j in range(n_cells_per_time)]
        gene_names = [f'Gene_{j}' for j in range(n_genes_expr)]
        
        adata = anndata.AnnData(
            X=spliced,
            obs=pd.DataFrame({
                'condition': time_point,
                'time_point': time_val,
                'treatment': 'control' if time_val == 0 else 'drug'
            }, index=cell_names),
            var=pd.DataFrame({'gene_name': gene_names}, index=gene_names)
        )
        
        adata.layers['spliced'] = spliced
        adata.layers['unspliced'] = unspliced
        
        all_cells_data.append(adata)
    
    # Combine temporal data
    temporal_adata = anndata.concat(all_cells_data)
    
    # Initialize analyzer
    velocity_analyzer = VelocityNetworkAnalyzer()
    
    # Compute velocity
    print("Computing RNA velocity for temporal analysis...")
    velocity_adata = velocity_analyzer.compute_rna_velocity(
        temporal_adata, condition_key='condition'
    )
    
    # Analyze temporal trajectory
    print("Analyzing temporal drug response trajectory...")
    
    # Use optimal transport for trajectory analysis
    from bridge.optimal_transport import OptimalTransportIntegrator
    
    ot_integrator = OptimalTransportIntegrator(config={'reg': 0.05})
    
    trajectory_results = ot_integrator.trajectory_analysis(
        scenic_networks=temporal_networks,
        pinnacle_embeddings=temporal_embeddings,
        condition_order=time_points,
        time_points=time_values
    )
    
    # Display temporal analysis results
    print(f"\nTemporal Drug Response Analysis Results:")
    
    if 'trajectory_analysis' in trajectory_results:
        analysis = trajectory_results['trajectory_analysis']
        print(f"  Total trajectory cost: {analysis['total_cost']:.3f}")
        print(f"  Mean step cost: {analysis['mean_cost']:.3f}")
        print(f"  Mean velocity: {analysis['mean_velocity']:.3f}")
        print(f"  Trajectory is monotonic: {analysis['monotonic']}")
        
        if analysis['critical_conditions']:
            print(f"  Critical time points: {analysis['critical_conditions']}")
    
    # Show network evolution
    print(f"\nNetwork Evolution Over Time:")
    for time_point in time_points:
        if time_point in temporal_networks:
            network = temporal_networks[time_point]
            print(f"  {time_point}: {network.number_of_nodes()} nodes, {network.number_of_edges()} edges")
    
    # Plot trajectory if possible
    try:
        plot_temporal_trajectory(trajectory_results, time_values)
    except Exception as e:
        print(f"Trajectory plotting failed: {e}")
    
    return trajectory_results


def plot_dose_response_curves(dose_response_results):
    """Plot dose-response curves for network metrics."""
    
    doses = dose_response_results['doses']
    network_metrics = dose_response_results['network_metrics']
    
    # Extract metrics for plotting
    metrics_to_plot = ['n_edges', 'density', 'avg_clustering']
    
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(15, 5))
    
    for i, metric in enumerate(metrics_to_plot):
        values = [network_metrics[dose][metric] for dose in doses]
        
        axes[i].plot(doses, values, 'o-', linewidth=2, markersize=8)
        axes[i].set_xlabel('Dose')
        axes[i].set_ylabel(metric.replace('_', ' ').title())
        axes[i].set_title(f'{metric.replace("_", " ").title()} vs Dose')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dose_response_curves.png', dpi=150, bbox_inches='tight')
    print("Dose-response curves saved as 'dose_response_curves.png'")


def plot_temporal_trajectory(trajectory_results, time_values):
    """Plot temporal drug response trajectory."""
    
    if 'trajectory_costs' not in trajectory_results:
        return
    
    costs = trajectory_results['trajectory_costs']
    
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(time_values[:-1], costs, 'o-', linewidth=2, markersize=8, color='red')
    plt.xlabel('Time (hours)')
    plt.ylabel('Network Change Cost')
    plt.title('Drug Response Trajectory')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    if 'trajectory_analysis' in trajectory_results:
        analysis = trajectory_results['trajectory_analysis']
        if 'velocities' in analysis and analysis['velocities']:
            velocities = analysis['velocities']
            plt.plot(time_values[:-2], velocities, 's-', linewidth=2, markersize=8, color='blue')
            plt.xlabel('Time (hours)')
            plt.ylabel('Trajectory Velocity')
            plt.title('Network Change Velocity')
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('temporal_trajectory.png', dpi=150, bbox_inches='tight')
    print("Temporal trajectory saved as 'temporal_trajectory.png'")


def example_complete_pharmacology_workflow():
    """Complete pharmacology workflow combining all features."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Complete Pharmacology Workflow")
    print("="*70)
    
    # Create comprehensive drug response dataset
    combined_adata, individual_data = create_drug_response_data()
    networks = create_drug_response_networks()
    protein_embeddings = create_drug_response_protein_embeddings()
    
    # Initialize velocity-enhanced BRIDGE
    config = {
        'velocity': {
            'mode': 'dynamical',
            'n_top_genes': 800,
            'n_neighbors': 25
        },
        'bridge': {
            'scenic': {'min_regulon_size': 5},
            'pinnacle': {'normalize_embeddings': True}
        }
    }
    
    velocity_bridge = VelocityEnhancedBridge(config=config)
    
    print("Step 1: Complete drug response analysis...")
    
    # Run complete analysis
    drug_conditions = ['drug_low', 'drug_medium', 'drug_high', 'drug_combo']
    results = velocity_bridge.analyze_drug_response_networks(
        rna_adata=combined_adata,
        scenic_networks=networks,
        pinnacle_embeddings=protein_embeddings,
        drug_conditions=drug_conditions,
        control_condition='control',
        condition_key='condition'
    )
    
    print("Step 2: Dose-response analysis...")
    
    # Dose-response analysis
    dose_conditions = ['drug_low', 'drug_medium', 'drug_high']
    doses = [1, 5, 25]
    
    dose_results = velocity_bridge.analyze_dose_response(
        rna_adata=combined_adata,
        scenic_networks={k: v for k, v in networks.items() if k in dose_conditions + ['control']},
        dose_conditions=dose_conditions,
        doses=doses,
        control_condition='control',
        condition_key='condition'
    )
    
    print("Step 3: Comprehensive results summary...")
    
    # Create comprehensive pharmacology report
    pharmacology_report = {
        'drug_response_analysis': results,
        'dose_response_analysis': dose_results,
        'summary_statistics': {},
        'top_drug_targets': {},
        'network_perturbation_ranking': {},
        'pharmacological_insights': {}
    }
    
    # Extract summary statistics
    pharmacology_report['summary_statistics'] = {
        'total_conditions_analyzed': len(drug_conditions) + 1,  # +1 for control
        'total_networks_integrated': len(results['integrated_networks']),
        'total_drug_targets_identified': 0,
        'most_perturbed_condition': None,
        'strongest_dose_response': None
    }
    
    # Find most perturbed condition
    max_perturbation = 0
    most_perturbed = None
    
    for condition, drug_result in results['drug_response_results'].items():
        perturbation_score = drug_result.get('network_perturbation_score', 0)
        if perturbation_score > max_perturbation:
            max_perturbation = perturbation_score
            most_perturbed = condition
    
    pharmacology_report['summary_statistics']['most_perturbed_condition'] = most_perturbed
    pharmacology_report['summary_statistics']['max_perturbation_score'] = max_perturbation
    
    # Collect all drug targets
    all_targets = set()
    for condition, target_info in results.get('pharmacological_targets', {}).items():
        targets = [target[0] for target in target_info.get('top_targets', [])]
        all_targets.update(targets)
        pharmacology_report['top_drug_targets'][condition] = targets[:10]
    
    pharmacology_report['summary_statistics']['total_drug_targets_identified'] = len(all_targets)
    
    # Display comprehensive results
    print(f"\n" + "="*50)
    print("COMPREHENSIVE PHARMACOLOGY ANALYSIS RESULTS")
    print("="*50)
    
    stats = pharmacology_report['summary_statistics']
    print(f"Conditions analyzed: {stats['total_conditions_analyzed']}")
    print(f"Networks integrated: {stats['total_networks_integrated']}")
    print(f"Drug targets identified: {stats['total_drug_targets_identified']}")
    print(f"Most perturbed condition: {stats['most_perturbed_condition']} (score: {stats['max_perturbation_score']:.3f})")
    
    print(f"\nTop Drug Targets by Condition:")
    for condition, targets in pharmacology_report['top_drug_targets'].items():
        print(f"  {condition}: {targets[:5]}")  # Top 5 per condition
    
    print(f"\nDose-Response Analysis:")
    if 'doses' in dose_results:
        print(f"  Doses tested: {dose_results['doses']}")
        print(f"  Dose-response curves fitted: {len(dose_results.get('dose_response_curves', {}))}")
    
    # Generate pharmacological insights
    insights = []
    
    if most_perturbed:
        insights.append(f"Condition '{most_perturbed}' shows the strongest network perturbation")
    
    if len(all_targets) > 50:
        insights.append(f"Large number of targets ({len(all_targets)}) suggests broad drug effects")
    elif len(all_targets) < 20:
        insights.append(f"Focused target set ({len(all_targets)}) suggests specific drug mechanism")
    
    pharmacology_report['pharmacological_insights'] = insights
    
    print(f"\nPharmacological Insights:")
    for insight in insights:
        print(f"  • {insight}")
    
    print(f"\n" + "="*50)
    print("Analysis complete! Ready for drug development applications.")
    print("="*50)
    
    return pharmacology_report


if __name__ == "__main__":
    print("BRIDGE-Velocity: Drug Response and Pharmacology Analysis")
    print("=" * 60)
    
    # Run examples
    try:
        # Example 1: Basic drug response analysis
        basic_results = example_basic_drug_response_analysis()
        
        # Example 2: Dose-response analysis
        dose_results = example_dose_response_analysis()
        
        # Example 3: Drug target discovery
        target_results = example_drug_target_discovery()
        
        # Example 4: Temporal analysis
        temporal_results = example_temporal_drug_response()
        
        # Example 5: Complete workflow
        complete_results = example_complete_pharmacology_workflow()
        
        print("\n" + "="*60)
        print("All BRIDGE-Velocity examples completed successfully!")
        print("="*60)
        print("\nKey capabilities demonstrated:")
        print("✓ RNA velocity-guided network dynamics")
        print("✓ Drug response pathway identification")
        print("✓ Dose-response relationship analysis")
        print("✓ Drug target discovery and prioritization")
        print("✓ Temporal trajectory analysis")
        print("✓ Comprehensive pharmacological profiling")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install scvelo cellrank velocyto")

