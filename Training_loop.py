import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
import yaml
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
import logging

from GNN import ProcessMappingGNN
from Loss_function import ProcessMappingLoss

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ProcessMappingTrainer:
    def __init__(self, config_path='config.yaml'):
        self.config = self.load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {self.device}")
        
        self.setup_model()
        self.setup_training()
        self.history = {'train_loss': [], 'val_loss': [], 'metrics': {}}
    
    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            logging.info(f"Loaded config from {config_path}")
            return config
    
    def setup_model(self):
        try:
            model_config = self.config['model']
            self.model = ProcessMappingGNN(
                input_dim=len(self.config['data']['node_features']),
                hidden_dim=model_config['hidden_channels'],
                output_dim=len(self.config['data']['target_features']),
                num_layers=model_config['num_layers'],
                dropout=model_config['dropout']
            ).to(self.device)
            
            self.criterion = ProcessMappingLoss()
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=model_config['learning_rate'],
                weight_decay=model_config['weight_decay']
            )
            
            logging.info(f"Model architecture: {self.model}")
            logging.info(f"Total parameters: {sum(p.numel() for p in self.model.parameters())}")
            
        except Exception as e:
            logging.error(f"Error setting up model: {e}")
            raise
    
    def setup_training(self):
        self.train_loader = None
        self.val_loader = None
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(self.train_loader, desc='Training'):
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            try:
                # Log batch information
                logging.info(f"Processing batch {num_batches + 1}")
                logging.info(f"Batch sizes - x: {batch.x.shape}, edge_index: {batch.edge_index.shape}")
                
                # Forward pass
                node_pred, edge_pred, graph_emb = self.model(
                    batch.x,
                    batch.edge_index,
                    batch.batch if hasattr(batch, 'batch') else None
                )
                
                # Log predictions and targets
                logging.info(f"Predictions - Node: {node_pred.shape}, Edge: {edge_pred.shape}")
                logging.info(f"Targets - Node: {batch.y.shape}, Edge: {batch.edge_attr.shape}")
                
                # Compute loss
                loss, loss_components = self.criterion(
                    node_pred, edge_pred, graph_emb,
                    batch.y,
                    batch.edge_attr,
                    batch.batch if hasattr(batch, 'batch') else None
                )
                
                # Log loss information
                logging.info(f"Loss components: {loss_components}")
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
            except Exception as e:
                logging.error(f"Error in batch: {e}")
                logging.error(f"Batch info: x={batch.x.shape}, edge_index={batch.edge_index.shape}")
                logging.error(f"Full error: {str(e)}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        logging.info(f"Average training loss: {avg_loss:.4f}")
        return avg_loss
    
    def validate(self, loader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                
                try:
                    # Forward pass
                    node_pred, edge_pred, graph_emb = self.model(
                        batch.x,
                        batch.edge_index,
                        batch.batch if hasattr(batch, 'batch') else None
                    )
                    
                    # Compute loss
                    loss, loss_components = self.criterion(
                        node_pred, edge_pred, graph_emb,
                        batch.y,
                        batch.edge_attr,
                        batch.batch if hasattr(batch, 'batch') else None
                    )
                    
                    total_loss += loss.item()
                    all_preds.append(node_pred.cpu())
                    all_targets.append(batch.y.cpu())
                    num_batches += 1
                    
                    # Log validation metrics
                    if num_batches % 5 == 0:  # Log every 5 batches
                        logging.debug(f"Validation loss components: {loss_components}")
                    
                except Exception as e:
                    logging.error(f"Error in validation batch: {e}")
                    continue
        
        if num_batches == 0:
            return float('inf'), {'precision': 0, 'recall': 0, 'f1': 0}
        
        # Compute metrics
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = self.compute_metrics(all_preds, all_targets)
        
        return total_loss / num_batches, metrics
    
    def compute_metrics(self, predictions, targets):
        # Convert predictions to binary decisions if needed
        pred_labels = (predictions > 0.5).float()
        
        # Compute precision, recall, and F1 score
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets.numpy(), pred_labels.numpy(), average='weighted'
        )
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train(self, train_dataset, val_dataset, test_dataset=None):
        train_config = self.config['training']
        
        try:
            # Create data loaders
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=train_config['batch_size'],
                shuffle=True
            )
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=train_config['batch_size']
            )
            
            logging.info(f"Training on {len(train_dataset)} graphs")
            logging.info(f"Validating on {len(val_dataset)} graphs")
            
            # Training loop
            for epoch in range(train_config['epochs']):
                # Train
                train_loss = self.train_epoch()
                
                # Validate
                val_loss, metrics = self.validate(self.val_loader)
                
                # Update history
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['metrics'][epoch] = metrics
                
                # Early stopping
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self.save_checkpoint('best_model.pt')
                    logging.info(f"New best model saved with validation loss: {val_loss:.4f}")
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= train_config['early_stopping_patience']:
                        logging.info(f'Early stopping at epoch {epoch}')
                        break
                
                # Print progress
                logging.info(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, '
                           f'Val Loss = {val_loss:.4f}, F1 = {metrics["f1"]:.4f}')
            
            # Test if test data is provided
            if test_dataset is not None:
                self.test(test_dataset)
                
        except Exception as e:
            logging.error(f"Error during training: {e}")
            raise
    
    def test(self, test_dataset):
        try:
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.config['training']['batch_size']
            )
            test_loss, test_metrics = self.validate(test_loader)
            logging.info(f'Test Results: Loss = {test_loss:.4f}, '
                        f'F1 = {test_metrics["f1"]:.4f}')
            return test_metrics
            
        except Exception as e:
            logging.error(f"Error during testing: {e}")
            raise
    
    def save_checkpoint(self, filename):
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config,
                'history': self.history
            }, filename)
            logging.info(f"Checkpoint saved to {filename}")
            
        except Exception as e:
            logging.error(f"Error saving checkpoint: {e}")
            raise
    
    def load_checkpoint(self, filename):
        try:
            checkpoint = torch.load(filename)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.history = checkpoint['history']
            logging.info(f"Checkpoint loaded from {filename}")
            
        except Exception as e:
            logging.error(f"Error loading checkpoint: {e}")
            raise
    
    def plot_training_history(self):
        try:
            plt.figure(figsize=(12, 4))
            
            # Plot losses
            plt.subplot(1, 2, 1)
            plt.plot(self.history['train_loss'], label='Train Loss')
            plt.plot(self.history['val_loss'], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Training and Validation Loss')
            
            # Plot metrics
            plt.subplot(1, 2, 2)
            epochs = list(self.history['metrics'].keys())
            f1_scores = [metrics['f1'] for metrics in self.history['metrics'].values()]
            plt.plot(epochs, f1_scores, label='F1 Score')
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.legend()
            plt.title('F1 Score Evolution')
            
            plt.tight_layout()
            plt.savefig('training_history.png')
            plt.close()
            logging.info("Training history plot saved to training_history.png")
            
        except Exception as e:
            logging.error(f"Error plotting training history: {e}")
            raise