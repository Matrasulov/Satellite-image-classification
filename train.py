"""
Training script for satellite image classification.
"""

import os
import torch
import timm
import torchmetrics
from tqdm import tqdm


class Trainer:
    """Trainer class for satellite image classification."""
    
    def __init__(self, model, train_loader, val_loader, num_classes, 
                 device='cuda', learning_rate=3e-4, epochs=10, 
                 patience=5, save_dir='saved_models', model_name='satellite'):
        """
        Args:
            model: PyTorch model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            num_classes (int): Number of classes
            device (str): Device to use for training
            learning_rate (float): Learning rate
            epochs (int): Number of training epochs
            patience (int): Early stopping patience
            save_dir (str): Directory to save models
            model_name (str): Prefix for saved model files
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.patience = patience
        self.save_dir = save_dir
        self.model_name = model_name
        
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.f1_score = torchmetrics.F1Score(
            task="multiclass", 
            num_classes=num_classes
        ).to(device)
        
        # Tracking metrics
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.train_f1s = []
        self.val_f1s = []
        
        self.best_loss = float('inf')
        self.threshold = 0.01
        self.not_improved = 0
    
    def _to_device(self, batch):
        """Move batch to device."""
        return batch[0].to(self.device), batch[1].to(self.device)
    
    def _compute_metrics(self, images, labels):
        """Compute loss, accuracy, and F1 score."""
        preds = self.model(images)
        loss = self.criterion(preds, labels)
        acc = (torch.argmax(preds, dim=1) == labels).sum().item()
        f1 = self.f1_score(preds, labels)
        return loss, acc, f1
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0
        epoch_acc = 0
        epoch_f1 = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
        for batch in progress_bar:
            images, labels = self._to_device(batch)
            
            loss, acc, f1 = self._compute_metrics(images, labels)
            epoch_loss += loss.item()
            epoch_acc += acc
            epoch_f1 += f1
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / len(self.train_loader)
        avg_acc = epoch_acc / len(self.train_loader.dataset)
        avg_f1 = epoch_f1 / len(self.train_loader)
        
        self.train_losses.append(avg_loss)
        self.train_accs.append(avg_acc)
        self.train_f1s.append(avg_f1)
        
        return avg_loss, avg_acc, avg_f1
    
    def validate(self, epoch):
        """Validate the model."""
        self.model.eval()
        epoch_loss = 0
        epoch_acc = 0
        epoch_f1 = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                images, labels = self._to_device(batch)
                loss, acc, f1 = self._compute_metrics(images, labels)
                epoch_loss += loss.item()
                epoch_acc += acc
                epoch_f1 += f1
        
        avg_loss = epoch_loss / len(self.val_loader)
        avg_acc = epoch_acc / len(self.val_loader.dataset)
        avg_f1 = epoch_f1 / len(self.val_loader)
        
        self.val_losses.append(avg_loss)
        self.val_accs.append(avg_acc)
        self.val_f1s.append(avg_f1)
        
        return avg_loss, avg_acc, avg_f1
    
    def save_checkpoint(self):
        """Save model checkpoint."""
        os.makedirs(self.save_dir, exist_ok=True)
        save_path = f"{self.save_dir}/{self.model_name}_best_model.pth"
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
    
    def train(self):
        """Complete training loop."""
        print("Starting training...")
        print("=" * 60)
        
        for epoch in range(self.epochs):
            # Train
            train_loss, train_acc, train_f1 = self.train_epoch(epoch)
            print(f"\nEpoch {epoch+1}/{self.epochs} - Training:")
            print(f"  Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
            
            # Validate
            val_loss, val_acc, val_f1 = self.validate(epoch)
            print(f"Epoch {epoch+1}/{self.epochs} - Validation:")
            print(f"  Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
            
            # Early stopping and model saving
            if val_loss < (self.best_loss + self.threshold):
                self.best_loss = val_loss
                self.not_improved = 0
                self.save_checkpoint()
                print(f"  ✓ New best model saved!")
            else:
                self.not_improved += 1
                print(f"  No improvement for {self.not_improved} epoch(s)")
                
                if self.not_improved >= self.patience:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                    break
            
            print("-" * 60)
        
        print("\nTraining completed!")
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'train_f1s': self.train_f1s,
            'val_f1s': self.val_f1s
        }


def create_model(model_name='rexnet_150', num_classes=4, pretrained=True):
    """
    Create a model for satellite image classification.
    
    Args:
        model_name (str): Model architecture name
        num_classes (int): Number of output classes
        pretrained (bool): Use pretrained weights
    
    Returns:
        model: PyTorch model
    """
    model = timm.create_model(
        model_name, 
        pretrained=pretrained, 
        num_classes=num_classes
    )
    return model
