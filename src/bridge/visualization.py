"""
Visualization module for SCENIC+ and PINNACLE integration.

This module provides visualization capabilities for integrated networks,
embeddings, and differential analysis results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

class NetworkVisualizer:
    """
    Visualization tools for integrated networks and analysis results.
    
    Provides methods for creating network plots, heatmaps, embedding
    visualizations, and differential analysis plots.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize network visualizer.
        
        Args:
            config: Configuration parameters for visualization
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__ + '.NetworkVisualizer')
        
        # Set default style parameters
        self.default_style = {
            'figure_size': (12, 8),
            'dpi': 300,
            'font_size': 12,
            'color_palette': 'Set1',
            'node_size_range': (20, 200),
            'edge_width_range': (0.5, 3.0)
        }
        
        # Update with user config
        self.style = {**self.default_style, **self.config.get('style', {})}
    
    def plot_network(self, 
                    network: nx.Graph,
                    output_path: Optional[Union[str, Path]] = None,
                    layout: str = 'spring',
                    color_by: str = 'node_type',
                    size_by: str = 'degree',
                    show_labels: bool = True,
                    interactive: bool = False) -> None:
        """
        Plot integrated network.
        
        Args:
            network: NetworkX graph to plot
            output_path: Path to save plot
            layout: Layout algorithm ('spring', 'circular', 'kamada_kawai')
            color_by: Node attribute for coloring
            size_by: Node attribute for sizing
            show_labels: Whether to show node labels
            interactive: Whether to create interactive plot
        """
        self.logger.info("Creating network plot")
        
        if interactive:
            self._plot_network_interactive(network, output_path, color_by, size_by)
        else:
            self._plot_network_static(network, output_path, layout, color_by, 
                                    size_by, show_labels)
    
    def _plot_network_static(self,
                           network: nx.Graph,
                           output_path: Optional[Union[str, Path]],
                           layout: str,
                           color_by: str,
                           size_by: str,
                           show_labels: bool) -> None:
        """Create static network plot using matplotlib."""
        fig, ax = plt.subplots(figsize=self.style['figure_size'], 
                              dpi=self.style['dpi'])
        
        # Compute layout
        if layout == 'spring':
            pos = nx.spring_layout(network, k=1, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(network)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(network)
        else:
            pos = nx.spring_layout(network)
        
        # Get node colors
        node_colors = self._get_node_colors(network, color_by)
        
        # Get node sizes
        node_sizes = self._get_node_sizes(network, size_by)
        
        # Draw edges by type
        edge_types = set(data.get('edge_type', 'unknown') 
                        for _, _, data in network.edges(data=True))
        
        edge_colors = {
            'regulatory': 'red',
            'protein_similarity': 'blue', 
            'cross_layer': 'green',
            'unknown': 'gray'
        }
        
        for edge_type in edge_types:
            edges = [(u, v) for u, v, data in network.edges(data=True)
                    if data.get('edge_type') == edge_type]
            
            if edges:
                nx.draw_networkx_edges(network, pos, edgelist=edges,
                                     edge_color=edge_colors.get(edge_type, 'gray'),
                                     alpha=0.6, width=1.0, ax=ax)
        
        # Draw nodes
        nx.draw_networkx_nodes(network, pos, node_color=node_colors,
                             node_size=node_sizes, alpha=0.8, ax=ax)
        
        # Draw labels if requested
        if show_labels and network.number_of_nodes() < 100:
            nx.draw_networkx_labels(network, pos, font_size=8, ax=ax)
        
        # Create legend
        self._add_network_legend(ax, edge_types, edge_colors, color_by)
        
        ax.set_title(f"Integrated Network ({network.number_of_nodes()} nodes, "
                    f"{network.number_of_edges()} edges)")
        ax.axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.style['dpi'], bbox_inches='tight')
            self.logger.info(f"Network plot saved to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def _plot_network_interactive(self,
                                network: nx.Graph,
                                output_path: Optional[Union[str, Path]],
                                color_by: str,
                                size_by: str) -> None:
        """Create interactive network plot using plotly."""
        # Compute layout
        pos = nx.spring_layout(network, k=1, iterations=50)
        
        # Prepare node data
        node_x = [pos[node][0] for node in network.nodes()]
        node_y = [pos[node][1] for node in network.nodes()]
        node_text = list(network.nodes())
        
        # Get node colors and sizes
        node_colors = self._get_node_colors(network, color_by, categorical=True)
        node_sizes = self._get_node_sizes(network, size_by)
        
        # Prepare edge data
        edge_x = []
        edge_y = []
        edge_info = []
        
        for u, v, data in network.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_info.append(f"{u} - {v}: {data.get('edge_type', 'unknown')}")
        
        # Create edge trace
        edge_trace = go.Scatter(x=edge_x, y=edge_y,
                               line=dict(width=0.5, color='#888'),
                               hoverinfo='none',
                               mode='lines')
        
        # Create node trace
        node_trace = go.Scatter(x=node_x, y=node_y,
                               mode='markers+text',
                               hoverinfo='text',
                               text=node_text,
                               textposition="middle center",
                               marker=dict(size=node_sizes,
                                         color=node_colors,
                                         colorscale='Viridis',
                                         showscale=True,
                                         line=dict(width=0.5, color='black')))
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Interactive Integrated Network',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Network visualization",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(color='black', size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
        
        if output_path:
            fig.write_html(str(output_path).replace('.png', '.html'))
            self.logger.info(f"Interactive network plot saved to {output_path}")
        else:
            fig.show()
    
    def plot_heatmap(self,
                    network: nx.Graph,
                    output_path: Optional[Union[str, Path]] = None,
                    metric: str = 'adjacency',
                    cluster: bool = True) -> None:
        """
        Plot network as heatmap.
        
        Args:
            network: NetworkX graph
            output_path: Path to save plot
            metric: Type of matrix to plot ('adjacency', 'similarity')
            cluster: Whether to cluster rows/columns
        """
        self.logger.info("Creating network heatmap")
        
        # Create adjacency matrix
        if metric == 'adjacency':
            matrix = nx.adjacency_matrix(network).toarray()
            title = "Network Adjacency Matrix"
        elif metric == 'similarity':
            # Create similarity matrix based on edge weights
            matrix = self._create_similarity_matrix(network)
            title = "Network Similarity Matrix"
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        # Create DataFrame for better labeling
        nodes = list(network.nodes())
        df = pd.DataFrame(matrix, index=nodes, columns=nodes)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=self.style['figure_size'], 
                              dpi=self.style['dpi'])
        
        sns.heatmap(df, cmap='viridis', center=0, square=True,
                   linewidths=0.1, cbar_kws={"shrink": 0.8},
                   xticklabels=False, yticklabels=False,
                   ax=ax)
        
        ax.set_title(title)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.style['dpi'], bbox_inches='tight')
            self.logger.info(f"Heatmap saved to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_embeddings(self,
                       network: nx.Graph,
                       output_path: Optional[Union[str, Path]] = None,
                       method: str = 'umap',
                       color_by: str = 'node_type') -> None:
        """
        Plot protein embeddings in 2D.
        
        Args:
            network: Integrated network with embedding information
            output_path: Path to save plot
            method: Dimensionality reduction method ('pca', 'tsne', 'umap')
            color_by: Node attribute for coloring
        """
        self.logger.info(f"Creating embedding plot using {method}")
        
        # Extract embeddings from network
        embeddings, node_labels, colors = self._extract_embeddings_from_network(
            network, color_by
        )
        
        if embeddings.shape[0] == 0:
            self.logger.warning("No embeddings found in network")
            return
        
        # Perform dimensionality reduction
        if method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
        elif method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, embeddings.shape[0]-1))
        elif method == 'umap':
            reducer = umap.UMAP(n_components=2, random_state=42)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        coords_2d = reducer.fit_transform(embeddings)
        
        # Create plot
        fig, ax = plt.subplots(figsize=self.style['figure_size'], 
                              dpi=self.style['dpi'])
        
        # Create scatter plot
        scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], 
                           c=colors, alpha=0.7, s=50)
        
        # Add labels for a subset of points
        if len(node_labels) < 50:
            for i, label in enumerate(node_labels):
                ax.annotate(label, (coords_2d[i, 0], coords_2d[i, 1]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.8)
        
        ax.set_xlabel(f'{method.upper()} 1')
        ax.set_ylabel(f'{method.upper()} 2')
        ax.set_title(f'Protein Embeddings ({method.upper()})')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.style['dpi'], bbox_inches='tight')
            self.logger.info(f"Embedding plot saved to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_differential_analysis(self,
                                 results: Dict,
                                 output_path: Optional[Union[str, Path]] = None,
                                 plot_type: str = 'volcano') -> None:
        """
        Plot differential analysis results.
        
        Args:
            results: Differential analysis results
            output_path: Path to save plot
            plot_type: Type of plot ('volcano', 'heatmap', 'barplot')
        """
        self.logger.info(f"Creating differential analysis plot ({plot_type})")
        
        if plot_type == 'volcano':
            self._plot_volcano(results, output_path)
        elif plot_type == 'heatmap':
            self._plot_differential_heatmap(results, output_path)
        elif plot_type == 'barplot':
            self._plot_differential_barplot(results, output_path)
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")
    
    def _plot_volcano(self, results: Dict, output_path: Optional[Union[str, Path]]) -> None:
        """Create volcano plot for differential analysis."""
        if 'regulatory' not in results:
            self.logger.warning("No regulatory results for volcano plot")
            return
        
        reg_results = results['regulatory']
        if 'importance_changes' not in reg_results or reg_results['importance_changes'].empty:
            self.logger.warning("No importance changes for volcano plot")
            return
        
        df = reg_results['importance_changes']
        
        # Calculate -log10(p-value) - using fold change as proxy
        df['neg_log_p'] = np.abs(df['log_fold_change'])
        
        fig, ax = plt.subplots(figsize=self.style['figure_size'], 
                              dpi=self.style['dpi'])
        
        # Create scatter plot
        ax.scatter(df['log_fold_change'], df['neg_log_p'], 
                  alpha=0.6, s=30)
        
        # Add threshold lines
        fold_change_threshold = np.log2(1.5)
        ax.axvline(-fold_change_threshold, color='red', linestyle='--', alpha=0.7)
        ax.axvline(fold_change_threshold, color='red', linestyle='--', alpha=0.7)
        
        ax.set_xlabel('Log2 Fold Change')
        ax.set_ylabel('|Log2 Fold Change|')
        ax.set_title('Regulatory Changes (Volcano Plot)')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.style['dpi'], bbox_inches='tight')
            self.logger.info(f"Volcano plot saved to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def _plot_differential_heatmap(self, results: Dict, 
                                 output_path: Optional[Union[str, Path]]) -> None:
        """Create heatmap for differential analysis."""
        # Combine different result types
        data_for_heatmap = []
        
        if 'regulatory' in results:
            reg_summary = results['summary'].get('regulatory', {})
            data_for_heatmap.append({
                'Type': 'Regulatory',
                'Unique_Condition1': reg_summary.get('num_unique_condition1', 0),
                'Unique_Condition2': reg_summary.get('num_unique_condition2', 0),
                'Common': reg_summary.get('num_common', 0),
                'Significant_Changes': reg_summary.get('significant_changes', 0)
            })
        
        if 'protein' in results:
            prot_summary = results['summary'].get('protein', {})
            data_for_heatmap.append({
                'Type': 'Protein',
                'Unique_Condition1': prot_summary.get('num_unique_condition1', 0),
                'Unique_Condition2': prot_summary.get('num_unique_condition2', 0),
                'Common': prot_summary.get('num_common', 0),
                'Significant_Changes': prot_summary.get('significant_changes', 0)
            })
        
        if not data_for_heatmap:
            self.logger.warning("No data for differential heatmap")
            return
        
        df = pd.DataFrame(data_for_heatmap)
        df = df.set_index('Type')
        
        fig, ax = plt.subplots(figsize=self.style['figure_size'], 
                              dpi=self.style['dpi'])
        
        sns.heatmap(df, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Differential Analysis Summary')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.style['dpi'], bbox_inches='tight')
            self.logger.info(f"Differential heatmap saved to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def _plot_differential_barplot(self, results: Dict,
                                 output_path: Optional[Union[str, Path]]) -> None:
        """Create bar plot for differential analysis."""
        summary = results.get('summary', {})
        
        categories = []
        values = []
        
        for analysis_type in ['regulatory', 'protein']:
            if analysis_type in summary:
                type_summary = summary[analysis_type]
                categories.extend([
                    f'{analysis_type.title()}\nUnique 1',
                    f'{analysis_type.title()}\nUnique 2', 
                    f'{analysis_type.title()}\nCommon',
                    f'{analysis_type.title()}\nSignificant'
                ])
                values.extend([
                    type_summary.get('num_unique_condition1', 0),
                    type_summary.get('num_unique_condition2', 0),
                    type_summary.get('num_common', 0),
                    type_summary.get('significant_changes', 0)
                ])
        
        if not categories:
            self.logger.warning("No data for differential bar plot")
            return
        
        fig, ax = plt.subplots(figsize=self.style['figure_size'], 
                              dpi=self.style['dpi'])
        
        bars = ax.bar(categories, values, alpha=0.7)
        
        # Color bars by type
        colors = ['red', 'red', 'blue', 'blue'] * (len(categories) // 4)
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_ylabel('Count')
        ax.set_title('Differential Analysis Summary')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.style['dpi'], bbox_inches='tight')
            self.logger.info(f"Differential bar plot saved to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def _get_node_colors(self, network: nx.Graph, color_by: str, 
                        categorical: bool = False) -> Union[List, np.ndarray]:
        """Get node colors based on attribute."""
        if color_by not in ['node_type', 'degree', 'clustering']:
            color_by = 'node_type'
        
        if color_by == 'node_type':
            node_types = [network.nodes[node].get('node_type', 'unknown') 
                         for node in network.nodes()]
            
            if categorical:
                return node_types
            
            # Map to numeric values
            unique_types = list(set(node_types))
            type_to_num = {t: i for i, t in enumerate(unique_types)}
            return [type_to_num[t] for t in node_types]
        
        elif color_by == 'degree':
            return [network.degree(node) for node in network.nodes()]
        
        elif color_by == 'clustering':
            clustering = nx.clustering(network)
            return [clustering[node] for node in network.nodes()]
    
    def _get_node_sizes(self, network: nx.Graph, size_by: str) -> List[float]:
        """Get node sizes based on attribute."""
        if size_by == 'degree':
            degrees = [network.degree(node) for node in network.nodes()]
            if max(degrees) > 0:
                # Normalize to size range
                min_size, max_size = self.style['node_size_range']
                normalized = np.array(degrees) / max(degrees)
                return (normalized * (max_size - min_size) + min_size).tolist()
            else:
                return [self.style['node_size_range'][0]] * len(degrees)
        else:
            return [50] * network.number_of_nodes()
    
    def _add_network_legend(self, ax, edge_types: set, edge_colors: Dict, 
                          color_by: str) -> None:
        """Add legend to network plot."""
        # Edge type legend
        legend_elements = []
        for edge_type in edge_types:
            color = edge_colors.get(edge_type, 'gray')
            legend_elements.append(plt.Line2D([0], [0], color=color, lw=2, 
                                            label=edge_type.replace('_', ' ').title()))
        
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    def _create_similarity_matrix(self, network: nx.Graph) -> np.ndarray:
        """Create similarity matrix from network."""
        nodes = list(network.nodes())
        n = len(nodes)
        matrix = np.zeros((n, n))
        
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if network.has_edge(node1, node2):
                    edge_data = network.edges[node1, node2]
                    similarity = edge_data.get('similarity', 1.0)
                    matrix[i, j] = similarity
        
        return matrix
    
    def _extract_embeddings_from_network(self, network: nx.Graph, 
                                       color_by: str) -> Tuple[np.ndarray, List[str], List]:
        """Extract embeddings from network nodes."""
        embeddings = []
        node_labels = []
        colors = []
        
        for node in network.nodes():
            node_data = network.nodes[node]
            
            # Check if node has embedding
            if 'embedding' in node_data:
                embeddings.append(node_data['embedding'])
                node_labels.append(str(node))
                
                # Get color value
                if color_by == 'node_type':
                    colors.append(node_data.get('node_type', 'unknown'))
                else:
                    colors.append(0)  # Default color
        
        if embeddings:
            return np.array(embeddings), node_labels, colors
        else:
            return np.array([]), [], []

