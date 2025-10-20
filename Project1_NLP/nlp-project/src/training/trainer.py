import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm
import os
from typing import Dict, Optional, List
import time


class SentimentTrainer:
    """
    Trainer class for sentiment analysis models.
    Handles training loop, validation, and metrics tracking.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        config: Dict,
        device: torch.device,
        use_wandb: bool = False,
        local_tracker: Optional[object] = None
    ):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.device = device
        self.use_wandb = use_wandb
        self.local_tracker = local_tracker
        
        # Training parameters
        self.num_epochs = config.get('num_epochs', 3)
        self.learning_rate = config.get('learning_rate', 2e-5)
        self.weight_decay = config.get('weight_decay', 0.01)
        self.warmup_steps = config.get('warmup_steps', 0)
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        self.output_dir = config.get('output_dir', 'outputs')
        
        # Setup optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        total_steps = len(self.train_dataloader) * self.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Tracking
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def train(self) -> Dict:
        """Main training loop."""
        print(f"\n{'='*60}")
        print(f"Starting Training")
        print(f"{'='*60}")
        print(f"Epochs: {self.num_epochs}")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Batch Size: {self.train_dataloader.batch_size}")
        print(f"Training Steps per Epoch: {len(self.train_dataloader)}")
        print(f"{'='*60}\n")
        
        best_val_acc = 0.0
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print("-" * 60)
            
            # Training phase
            train_loss, train_acc = self._train_epoch()
            
            # Validation phase
            val_loss, val_acc = self._validate()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            # Log to tracker
            if self.local_tracker:
                self.local_tracker.log({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'learning_rate': current_lr
                })
            
            # Log to wandb
            if self.use_wandb:
                import wandb
                wandb.log({
                    'epoch': epoch + 1,
                    'train/loss': train_loss,
                    'train/accuracy': train_acc,
                    'val/loss': val_loss,
                    'val/accuracy': val_acc,
                    'learning_rate': current_lr
                })
            
            # Print epoch summary
            print(f"\n📊 Epoch {epoch + 1} Summary:")
            print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            print(f"   Learning Rate: {current_lr:.2e}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self._save_checkpoint(epoch, val_acc, is_best=True)
                print(f"   ✅ New best model! Val Acc: {val_acc:.4f}")
            
            print("-" * 60)
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"{'='*60}")
        print(f"Best Validation Accuracy: {best_val_acc:.4f}")
        print(f"{'='*60}\n")
        
        return self.history
    
    def _train_epoch(self) -> tuple:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_dataloader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            
            # Calculate accuracy
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # Update metrics
            total_loss += loss.item()
            current_acc = correct / total
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_acc:.4f}'
            })
        
        avg_loss = total_loss / len(self.train_dataloader)
        avg_acc = correct / total
        
        return avg_loss, avg_acc
    
    def _validate(self) -> tuple:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_dataloader, desc="Validation")
            
            for batch in progress_bar:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                
                # Calculate accuracy
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
                # Update metrics
                total_loss += loss.item()
                current_acc = correct / total
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{current_acc:.4f}'
                })
        
        avg_loss = total_loss / len(self.val_dataloader)
        avg_acc = correct / total
        
        return avg_loss, avg_acc
    
    def _save_checkpoint(self, epoch: int, val_acc: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'history': self.history
        }
        
        if is_best:
            checkpoint_path = os.path.join(self.output_dir, 'best_model.pt')
        else:
            checkpoint_path = os.path.join(self.output_dir, f'checkpoint_epoch_{epoch+1}.pt')
        
        torch.save(checkpoint, checkpoint_path)
    
    def evaluate(self, test_dataloader: DataLoader) -> Dict:
        """Evaluate model on test set."""
        print(f"\n{'='*60}")
        print(f"Evaluating on Test Set")
        print(f"{'='*60}\n")
        
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        all_predictions = []
        all_labels = []
        all_logits = []
        
        with torch.no_grad():
            progress_bar = tqdm(test_dataloader, desc="Testing")
            
            for batch in progress_bar:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                
                # Calculate accuracy
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                total_loss += loss.item()
                
                # Store for detailed metrics
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_logits.extend(logits.cpu().numpy())
        
        test_loss = total_loss / len(test_dataloader)
        test_acc = correct / total
        
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'predictions': np.array(all_predictions),
            'labels': np.array(all_labels),
            'logits': np.array(all_logits)
        }
        
        print(f"\n📊 Test Results:")
        print(f"   Test Loss: {test_loss:.4f}")
        print(f"   Test Accuracy: {test_acc:.4f}")
        print(f"{'='*60}\n")
        
        return results