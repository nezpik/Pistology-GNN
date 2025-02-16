import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch

class ProcessGNN(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim, heads=4, p_norm=2):
        super().__init__()
        self.p_norm = p_norm  # L2 norm by default
        
        # Norm-based feature transformation (Section 3.3)
        self.norm_linear = nn.Linear(node_feat_dim, hidden_dim)
        
        # Multi-head attention layers (Section 3.2.3)
        self.attn1 = GATConv(hidden_dim, hidden_dim, heads=heads, edge_dim=edge_feat_dim)
        self.attn2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, edge_dim=edge_feat_dim)
        
        # Cycle-time regression head (Section 4.3.1)
        self.cycle_time_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Next-activity prediction head (Section 8.1.3)
        self.task_head = nn.Linear(hidden_dim, 10)  # Assuming 10 task classes

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Norm-based feature embedding (Section 3.3.1)
        x_norm = torch.norm(x, p=self.p_norm, dim=1, keepdim=True) + 1e-6
        x = self.norm_linear(x / x_norm)  # Normalized input
        
        # Message passing with attention (Section 3.2.1)
        x = F.relu(self.attn1(x, edge_index, edge_attr))
        x = F.relu(self.attn2(x, edge_index, edge_attr))
        
        # Node embeddings
        node_embeddings = x
        
        # Task-level predictions (next-activity)
        task_preds = self.task_head(node_embeddings)
        
        # Workflow-level predictions (cycle time)
        graph_embedding = global_mean_pool(node_embeddings, data.batch)
        cycle_time_pred = self.cycle_time_head(graph_embedding)
        
        return task_preds, cycle_time_pred


class ProcessMappingGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Node embedding layers
        self.node_embedding = nn.Linear(input_dim, hidden_dim)
        
        # GAT layers with residual connections
        self.gat_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gat_layers.append(GATConv(hidden_dim, hidden_dim))
        
        # Process prediction layers
        self.process_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Edge prediction layers
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3)  # 3 edge features: flow_rate, distance, cost
        )
    
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass of the GNN
        
        Args:
            x: Node features (N x F_in)
            edge_index: Edge indices (2 x E)
            batch: Batch assignment vector (N) or None
        
        Returns:
            node_pred: Node predictions (N x F_out)
            edge_pred: Edge predictions (E x F_edge)
            graph_emb: Graph embedding (B x F_hidden)
        """
        # Initial node embedding
        h = self.node_embedding(x)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # GAT layers with residual connections
        for gat_layer in self.gat_layers:
            h_new = gat_layer(h, edge_index)
            h = h + h_new  # Residual connection
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Process predictions
        node_pred = self.process_mlp(h)  # (N x F_out)
        
        # Edge predictions
        edge_features = []
        for i in range(edge_index.size(1)):
            # Get node features for both ends of each edge
            src_idx = edge_index[0, i]
            dst_idx = edge_index[1, i]
            edge_feature = torch.cat([h[src_idx], h[dst_idx]])
            edge_features.append(edge_feature)
        
        edge_features = torch.stack(edge_features)
        edge_pred = self.edge_mlp(edge_features)  # (E x F_edge)
        
        # Graph-level embedding
        if batch is not None:
            graph_emb = global_mean_pool(h, batch)  # (B x F_hidden)
        else:
            graph_emb = torch.mean(h, dim=0, keepdim=True)  # (1 x F_hidden)
        
        return node_pred, edge_pred, graph_emb