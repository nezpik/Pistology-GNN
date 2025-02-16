import torch
import torch.nn as nn
import torch.nn.functional as F

class ProcessMappingLoss(nn.Module):
    def __init__(self, node_loss_weight=1.0, edge_loss_weight=0.5, flow_loss_weight=0.1):
        super().__init__()
        self.node_loss_weight = node_loss_weight
        self.edge_loss_weight = edge_loss_weight
        self.flow_loss_weight = flow_loss_weight
        
        # MSE loss for continuous values
        self.mse_loss = nn.MSELoss(reduction='mean')
    
    def forward(self, node_pred, edge_pred, graph_emb, node_target, edge_target, batch=None):
        """
        Compute the combined loss for process mapping
        
        Args:
            node_pred: Predicted node features (N x F_out)
            edge_pred: Predicted edge features (E x F_edge)
            graph_emb: Graph embedding (B x F_hidden)
            node_target: Target node features (N x F_target)
            edge_target: Target edge features (E x F_edge)
            batch: Batch assignment vector (N) or None
        
        Returns:
            total_loss: Combined loss value
            loss_components: Dictionary of individual loss components
        """
        try:
            # Ensure all tensors are on the same device
            device = node_pred.device
            node_target = node_target.to(device)
            edge_target = edge_target.to(device)
            
            # Normalize predictions and targets
            node_pred = F.normalize(node_pred, dim=1)
            node_target = F.normalize(node_target, dim=1)
            edge_pred = F.normalize(edge_pred, dim=1)
            edge_target = F.normalize(edge_target, dim=1)
            
            # Node-level loss (MSE for throughput prediction)
            node_loss = self.mse_loss(node_pred, node_target)
            
            # Edge-level loss (MSE for flow prediction)
            edge_loss = self.mse_loss(edge_pred, edge_target)
            
            # Flow consistency loss
            flow_loss = self.compute_flow_consistency_loss(edge_pred, edge_target)
            
            # Scale losses to similar ranges
            node_loss = torch.clamp(node_loss, max=10.0)
            edge_loss = torch.clamp(edge_loss, max=10.0)
            flow_loss = torch.clamp(flow_loss, max=10.0)
            
            # Combine losses with weights
            total_loss = (
                self.node_loss_weight * node_loss +
                self.edge_loss_weight * edge_loss +
                self.flow_loss_weight * flow_loss
            )
            
            loss_components = {
                'node_loss': node_loss.item(),
                'edge_loss': edge_loss.item(),
                'flow_loss': flow_loss.item(),
                'total_loss': total_loss.item()
            }
            
            return total_loss, loss_components
            
        except Exception as e:
            print(f"Error in loss computation: {str(e)}")
            print(f"Shapes - node_pred: {node_pred.shape}, node_target: {node_target.shape}")
            print(f"Shapes - edge_pred: {edge_pred.shape}, edge_target: {edge_target.shape}")
            raise
    
    def compute_flow_consistency_loss(self, edge_pred, edge_target):
        """
        Compute flow consistency loss to ensure conservation of flow
        
        Args:
            edge_pred: Predicted edge features (E x F_edge)
            edge_target: Target edge features (E x F_edge)
        
        Returns:
            flow_loss: Flow consistency loss value
        """
        # Extract flow rates (first feature)
        pred_flows = edge_pred[:, 0]
        target_flows = edge_target[:, 0]
        
        # Normalize flows
        pred_flows = F.normalize(pred_flows, dim=0)
        target_flows = F.normalize(target_flows, dim=0)
        
        # Compute flow consistency loss
        flow_diff = torch.abs(pred_flows - target_flows)
        flow_loss = torch.mean(flow_diff)
        
        return flow_loss