import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import StandardScaler
import os.path as osp

class ProcessData(Data):
    def __init__(self, x=None, edge_index=None, edge_attr=None, y=None):
        super().__init__()
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y
    
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return self.x.size(0)
        return super().__inc__(key, value, *args, **kwargs)

class ProcessDataset(InMemoryDataset):
    def __init__(self, data_list, transform=None):
        super().__init__('.', transform)
        self.data, self.slices = self.collate(data_list)
    
    def _download(self):
        pass  # No download required
    
    def _process(self):
        pass  # No processing required

class ProcessDataProcessor:
    def __init__(self, config_path='config.yaml'):
        self.config = self.load_config(config_path)
        self.node_scaler = StandardScaler()
        self.edge_scaler = StandardScaler()
        
    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def process_raw_data(self, nodes_df, edges_df):
        """
        Process raw dataframes into PyG Data objects
        """
        # Process node features
        node_features = nodes_df[self.config['data']['node_features']].values
        node_features = self.node_scaler.fit_transform(node_features)
        
        # Ensure edge indices are within bounds
        max_node_idx = len(nodes_df) - 1
        valid_edges_mask = (edges_df['source'] <= max_node_idx) & (edges_df['target'] <= max_node_idx)
        edges_df = edges_df[valid_edges_mask].copy()
        
        # Convert edge indices to numpy array
        edge_index = np.vstack((
            edges_df['source'].values,
            edges_df['target'].values
        ))
        
        # Process edge features
        edge_features = edges_df[self.config['data']['edge_features']].values
        edge_features = self.edge_scaler.fit_transform(edge_features)
        
        # Process target features
        target_features = nodes_df[self.config['data']['target_features']].values
        
        # Create PyG Data object
        data = ProcessData(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_features, dtype=torch.float),
            y=torch.tensor(target_features, dtype=torch.float)
        )
        
        return data
    
    def split_data(self, data):
        """
        Split a PyG Data object into train, validation, and test sets
        """
        num_nodes = data.x.size(0)
        indices = torch.randperm(num_nodes)
        
        # Calculate split sizes
        train_size = int(0.7 * num_nodes)
        val_size = int(0.15 * num_nodes)
        
        # Split indices
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # Create subsets
        train_data = self.create_subset(data, train_indices)
        val_data = self.create_subset(data, val_indices)
        test_data = self.create_subset(data, test_indices)
        
        # Convert to datasets
        train_dataset = ProcessDataset([train_data])
        val_dataset = ProcessDataset([val_data])
        test_dataset = ProcessDataset([test_data])
        
        return train_dataset, val_dataset, test_dataset
    
    def create_subset(self, data, indices):
        """
        Create a subset of the data using indices
        """
        # Create index mapping for edge index update
        idx_map = -torch.ones(data.x.size(0), dtype=torch.long)
        idx_map[indices] = torch.arange(len(indices))
        
        # Get edges where both nodes are in the subset
        edge_mask = torch.isin(data.edge_index[0], indices) & torch.isin(data.edge_index[1], indices)
        subset_edge_index = data.edge_index[:, edge_mask]
        
        # Update edge indices to match new node indices
        subset_edge_index = idx_map[subset_edge_index]
        
        return ProcessData(
            x=data.x[indices],
            edge_index=subset_edge_index,
            edge_attr=data.edge_attr[edge_mask] if data.edge_attr is not None else None,
            y=data.y[indices] if data.y is not None else None
        )
    
    def visualize_process_graph(self, data, output_file='process_graph.png'):
        """
        Visualize the process graph using networkx
        """
        G = nx.DiGraph()
        
        # Add nodes
        for i in range(data.x.size(0)):
            G.add_node(i)
        
        # Add edges
        edge_list = data.edge_index.t().numpy()
        for i in range(len(edge_list)):
            G.add_edge(edge_list[i][0], edge_list[i][1])
        
        # Plot
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, 
                             node_color='lightblue',
                             node_size=500)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, 
                             edge_color='gray',
                             arrows=True,
                             arrowsize=20)
        
        # Add labels
        nx.draw_networkx_labels(G, pos)
        
        plt.title("Process Flow Graph")
        plt.axis('off')
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()
