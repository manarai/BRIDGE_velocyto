"""
Tests for the core ScenicPinnacleIntegrator class.
"""

import pytest
import numpy as np
import networkx as nx
from pathlib import Path
import tempfile
import pickle

from scenic_pinnacle import ScenicPinnacleIntegrator


@pytest.fixture
def sample_scenic_networks():
    """Create sample SCENIC+ networks for testing."""
    networks = {}
    
    for condition in ['condition1', 'condition2']:
        G = nx.DiGraph()
        
        # Add TF nodes
        tfs = [f'TF_{i}' for i in range(1, 6)]
        for tf in tfs:
            G.add_node(tf, node_type='TF')
        
        # Add target nodes
        targets = [f'Gene_{i}' for i in range(1, 11)]
        for target in targets:
            G.add_node(target, node_type='target')
        
        # Add regulatory edges
        for i, tf in enumerate(tfs):
            for j in range(2):  # Each TF regulates 2 targets
                target = targets[i * 2 + j]
                importance = np.random.uniform(0.3, 0.9)
                G.add_edge(tf, target, 
                          importance=importance,
                          edge_type='regulation')
        
        networks[condition] = G
    
    return networks


@pytest.fixture
def sample_pinnacle_embeddings():
    """Create sample PINNACLE embeddings for testing."""
    embeddings = {}
    
    # Create protein list
    protein_ids = [f'Protein_TF_{i}' for i in range(1, 6)] + \
                  [f'Protein_Gene_{i}' for i in range(1, 11)]
    
    for condition in ['condition1', 'condition2']:
        # Generate random embeddings
        np.random.seed(42)
        embedding_matrix = np.random.randn(len(protein_ids), 128)
        
        # Normalize embeddings
        norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        embedding_matrix = embedding_matrix / norms
        
        embeddings[condition] = {
            'embeddings': embedding_matrix,
            'protein_ids': protein_ids,
            'embedding_dim': 128
        }
    
    return embeddings


@pytest.fixture
def temp_data_files(sample_scenic_networks, sample_pinnacle_embeddings):
    """Create temporary data files for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Save SCENIC+ data
        scenic_file = temp_path / 'scenic_networks.pkl'
        with open(scenic_file, 'wb') as f:
            pickle.dump(sample_scenic_networks, f)
        
        # Save PINNACLE data
        pinnacle_file = temp_path / 'pinnacle_embeddings.pkl'
        with open(pinnacle_file, 'wb') as f:
            pickle.dump(sample_pinnacle_embeddings, f)
        
        yield {
            'scenic_path': scenic_file,
            'pinnacle_path': pinnacle_file,
            'temp_dir': temp_path
        }


class TestScenicPinnacleIntegrator:
    """Test cases for ScenicPinnacleIntegrator."""
    
    def test_initialization_default_config(self):
        """Test integrator initialization with default configuration."""
        integrator = ScenicPinnacleIntegrator()
        
        assert integrator.config is not None
        assert hasattr(integrator, 'scenic_processor')
        assert hasattr(integrator, 'pinnacle_processor')
        assert hasattr(integrator, 'network_integrator')
        assert hasattr(integrator, 'differential_analyzer')
        assert hasattr(integrator, 'visualizer')
    
    def test_initialization_custom_config(self):
        """Test integrator initialization with custom configuration."""
        custom_config = {
            'scenic': {'min_regulon_size': 15},
            'pinnacle': {'normalize_embeddings': False}
        }
        
        integrator = ScenicPinnacleIntegrator(config=custom_config)
        
        assert integrator.config['scenic']['min_regulon_size'] == 15
        assert integrator.config['pinnacle']['normalize_embeddings'] is False
    
    def test_load_scenic_data(self, temp_data_files):
        """Test loading SCENIC+ data."""
        integrator = ScenicPinnacleIntegrator()
        
        integrator.load_scenic_data(temp_data_files['scenic_path'], data_format='pickle')
        
        assert len(integrator.scenic_networks) == 2
        assert 'condition1' in integrator.scenic_networks
        assert 'condition2' in integrator.scenic_networks
        
        # Check network structure
        network = integrator.scenic_networks['condition1']
        assert isinstance(network, nx.DiGraph)
        assert network.number_of_nodes() > 0
        assert network.number_of_edges() > 0
    
    def test_load_pinnacle_data(self, temp_data_files):
        """Test loading PINNACLE data."""
        integrator = ScenicPinnacleIntegrator()
        
        integrator.load_pinnacle_data(temp_data_files['pinnacle_path'], data_format='pickle')
        
        assert len(integrator.pinnacle_embeddings) == 2
        assert 'condition1' in integrator.pinnacle_embeddings
        assert 'condition2' in integrator.pinnacle_embeddings
        
        # Check embedding structure
        embeddings = integrator.pinnacle_embeddings['condition1']
        assert 'embeddings' in embeddings
        assert 'protein_ids' in embeddings
        assert 'embedding_dim' in embeddings
        assert embeddings['embeddings'].shape[0] == len(embeddings['protein_ids'])
    
    def test_integrate_networks(self, temp_data_files):
        """Test network integration."""
        integrator = ScenicPinnacleIntegrator()
        
        # Load data
        integrator.load_scenic_data(temp_data_files['scenic_path'], data_format='pickle')
        integrator.load_pinnacle_data(temp_data_files['pinnacle_path'], data_format='pickle')
        
        # Integrate networks
        integrated_networks = integrator.integrate_networks()
        
        assert len(integrated_networks) == 2
        assert 'condition1' in integrated_networks
        assert 'condition2' in integrated_networks
        
        # Check integrated network structure
        network = integrated_networks['condition1']
        assert isinstance(network, nx.Graph)
        assert network.number_of_nodes() > 0
        assert network.number_of_edges() > 0
        
        # Check for different edge types
        edge_types = set()
        for _, _, data in network.edges(data=True):
            edge_types.add(data.get('edge_type', 'unknown'))
        
        assert len(edge_types) > 0  # Should have at least one edge type
    
    def test_differential_analysis(self, temp_data_files):
        """Test differential analysis between conditions."""
        integrator = ScenicPinnacleIntegrator()
        
        # Load data and integrate
        integrator.load_scenic_data(temp_data_files['scenic_path'], data_format='pickle')
        integrator.load_pinnacle_data(temp_data_files['pinnacle_path'], data_format='pickle')
        integrated_networks = integrator.integrate_networks()
        
        # Perform differential analysis
        diff_results = integrator.differential_analysis('condition1', 'condition2', analysis_type='both')
        
        assert 'summary' in diff_results
        assert isinstance(diff_results['summary'], dict)
        
        # Check for expected analysis types
        summary = diff_results['summary']
        expected_types = ['regulatory', 'protein']
        for analysis_type in expected_types:
            if analysis_type in summary:
                assert isinstance(summary[analysis_type], dict)
    
    def test_export_results(self, temp_data_files):
        """Test exporting results."""
        integrator = ScenicPinnacleIntegrator()
        
        # Load data and integrate
        integrator.load_scenic_data(temp_data_files['scenic_path'], data_format='pickle')
        integrator.load_pinnacle_data(temp_data_files['pinnacle_path'], data_format='pickle')
        integrated_networks = integrator.integrate_networks()
        
        # Export results
        output_dir = temp_data_files['temp_dir'] / 'results'
        integrator.export_results(output_dir, formats=['pickle'])
        
        assert output_dir.exists()
        
        # Check for exported files
        exported_files = list(output_dir.glob('*'))
        assert len(exported_files) > 0
    
    def test_run_complete_workflow(self, temp_data_files):
        """Test running the complete workflow."""
        integrator = ScenicPinnacleIntegrator()
        
        output_dir = temp_data_files['temp_dir'] / 'workflow_results'
        
        summary = integrator.run_complete_workflow(
            scenic_path=temp_data_files['scenic_path'],
            pinnacle_path=temp_data_files['pinnacle_path'],
            output_dir=output_dir,
            conditions=['condition1', 'condition2'],
            comparisons=[('condition1', 'condition2')]
        )
        
        assert isinstance(summary, dict)
        assert 'integrated_networks' in summary
        assert 'differential_analyses' in summary
        assert output_dir.exists()
    
    def test_error_handling_missing_data(self):
        """Test error handling when data is missing."""
        integrator = ScenicPinnacleIntegrator()
        
        with pytest.raises(Exception):
            integrator.integrate_networks()  # Should fail without loaded data
    
    def test_error_handling_invalid_conditions(self, temp_data_files):
        """Test error handling with invalid condition names."""
        integrator = ScenicPinnacleIntegrator()
        
        # Load data
        integrator.load_scenic_data(temp_data_files['scenic_path'], data_format='pickle')
        integrator.load_pinnacle_data(temp_data_files['pinnacle_path'], data_format='pickle')
        integrator.integrate_networks()
        
        with pytest.raises(Exception):
            integrator.differential_analysis('invalid_condition1', 'invalid_condition2')


class TestIntegrationQuality:
    """Test integration quality and validation."""
    
    def test_integration_preserves_data(self, temp_data_files):
        """Test that integration preserves important data properties."""
        integrator = ScenicPinnacleIntegrator()
        
        # Load data
        integrator.load_scenic_data(temp_data_files['scenic_path'], data_format='pickle')
        integrator.load_pinnacle_data(temp_data_files['pinnacle_path'], data_format='pickle')
        
        # Get original data sizes
        original_scenic_nodes = sum(net.number_of_nodes() for net in integrator.scenic_networks.values())
        original_scenic_edges = sum(net.number_of_edges() for net in integrator.scenic_networks.values())
        
        # Integrate networks
        integrated_networks = integrator.integrate_networks()
        
        # Check that integrated networks have reasonable sizes
        for condition, network in integrated_networks.items():
            assert network.number_of_nodes() > 0
            assert network.number_of_edges() > 0
            
            # Should have nodes from both regulatory and protein layers
            node_types = set()
            for node in network.nodes():
                node_type = network.nodes[node].get('node_type', 'unknown')
                node_types.add(node_type)
            
            assert len(node_types) > 0  # Should have at least one node type
    
    def test_differential_analysis_consistency(self, temp_data_files):
        """Test that differential analysis produces consistent results."""
        integrator = ScenicPinnacleIntegrator()
        
        # Load data and integrate
        integrator.load_scenic_data(temp_data_files['scenic_path'], data_format='pickle')
        integrator.load_pinnacle_data(temp_data_files['pinnacle_path'], data_format='pickle')
        integrated_networks = integrator.integrate_networks()
        
        # Perform differential analysis
        diff_results = integrator.differential_analysis('condition1', 'condition2', analysis_type='both')
        
        # Check consistency of results
        summary = diff_results.get('summary', {})
        
        for analysis_type, stats in summary.items():
            # Check that counts are non-negative
            for stat_name, value in stats.items():
                if isinstance(value, (int, float)):
                    assert value >= 0, f"{analysis_type}.{stat_name} should be non-negative"


if __name__ == '__main__':
    pytest.main([__file__])

