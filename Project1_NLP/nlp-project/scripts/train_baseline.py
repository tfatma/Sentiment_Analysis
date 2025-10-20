#!/usr/bin/env python3
"""
Baseline Model Training Script - Fixed Type Conversion

Trains a baseline transformer model with full fine-tuning.
"""

import os
import sys
import argparse
import yaml
import json
from pathlib import Path
from torch.utils.data import DataLoader
import torch

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.dataset import DataPreprocessor
from training.trainer import SentimentTrainer


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Get available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleBaselineModel(torch.nn.Module):
    """Simple baseline model wrapper."""
    
    def __init__(self, model_name: str, num_labels: int = 3):
        super().__init__()
        from transformers import AutoModel, AutoConfig
        
        config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=config)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # CLS token
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


def safe_float(value, default=2e-5):
    """Safely convert value to float."""
    try:
        if isinstance(value, str):
            return float(value)
        return float(value)
    except (ValueError, TypeError):
        print(f"Warning: Could not convert {value} to float, using default {default}")
        return default


def safe_int(value, default=3):
    """Safely convert value to int."""
    try:
        if isinstance(value, str):
            return int(value)
        return int(value)
    except (ValueError, TypeError):
        print(f"Warning: Could not convert {value} to int, using default {default}")
        return default


def main():
    parser = argparse.ArgumentParser(description="Train baseline sentiment model")
    parser.add_argument("--model_config", type=str, default="config/model_config.yaml")
    parser.add_argument("--training_config", type=str, default="config/training_config.yaml")
    parser.add_argument("--experiment_name", type=str, default="baseline_experiment")
    parser.add_argument("--output_dir", type=str, default=None)
    
    args = parser.parse_args()
    
    print("="*60)
    print(f"Baseline Model Training: {args.experiment_name}")
    print("="*60)
    
    # Load configs
    print("\nLoading configurations...")
    try:
        with open(args.model_config, 'r') as f:
            model_config = yaml.safe_load(f)
        print(f"✅ Loaded model config from {args.model_config}")
    except FileNotFoundError:
        print(f"⚠️ Model config not found, using defaults")
        model_config = {
            'model': {
                'name': 'roberta-base',
                'num_labels': 3,
                'max_length': 128
            }
        }
    
    try:
        with open(args.training_config, 'r') as f:
            training_config = yaml.safe_load(f)
        print(f"✅ Loaded training config from {args.training_config}")
    except FileNotFoundError:
        print(f"⚠️ Training config not found, using defaults")
        training_config = {
            'data': {
                'domains': ['electronics', 'books', 'movies', 'restaurants'],
                'train_size_per_domain': 50,
                'val_size_per_domain': 10,
                'test_size_per_domain': 10,
                'random_seed': 42
            },
            'training': {
                'num_epochs': 1,
                'batch_size': 8,
                'learning_rate': 2e-5,
                'weight_decay': 0.01,
                'warmup_steps': 0,
                'max_grad_norm': 1.0
            },
            'experiment': {
                'output_dir': 'outputs'
            }
        }
    
    # Set output directory
    if args.output_dir is None:
        output_base = training_config.get('experiment', {}).get('output_dir', 'outputs')
        args.output_dir = os.path.join(output_base, args.experiment_name)
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    
    print(f"✅ Output directory: {args.output_dir}")
    
    # Set seed for reproducibility
    seed = safe_int(training_config.get('data', {}).get('random_seed', 42))
    set_seed(seed)
    print(f"✅ Random seed set to: {seed}")
    
    # Prepare data
    print("\n" + "="*60)
    print("DATA PREPARATION")
    print("="*60)
    
    model_name = model_config.get('model', {}).get('name', 'roberta-base')
    preprocessor = DataPreprocessor(model_name)
    
    data_config = training_config.get('data', {})
    domains = data_config.get('domains', ['electronics', 'books', 'movies', 'restaurants'])
    train_size = safe_int(data_config.get('train_size_per_domain', 50))
    val_size = safe_int(data_config.get('val_size_per_domain', 10))
    test_size = safe_int(data_config.get('test_size_per_domain', 10))
    max_length = safe_int(model_config.get('model', {}).get('max_length', 128))
    
    train_dataset, val_dataset, test_dataset = preprocessor.create_datasets(
        domains=domains,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        max_length=max_length
    )
    
    print(f"\n✅ Datasets created successfully!")
    print(f"Data loaded - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create dataloaders
    train_config = training_config.get('training', {})
    batch_size = safe_int(train_config.get('batch_size', 8))
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Initialize model
    print("\n" + "="*60)
    print("MODEL INITIALIZATION")
    print("="*60)
    
    num_labels = safe_int(model_config.get('model', {}).get('num_labels', 3))
    
    print(f"Loading model: {model_name}")
    model = SimpleBaselineModel(model_name, num_labels)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal Parameters:     {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"Trainable Percentage: 100.00%")
    print(f"Model Size:           {total_params * 4 / 1024**2:.1f} MB")
    
    # Setup training config with proper type conversion
    training_config_clean = {
        'num_epochs': safe_int(train_config.get('num_epochs', 1)),
        'learning_rate': safe_float(train_config.get('learning_rate', 2e-5)),
        'weight_decay': safe_float(train_config.get('weight_decay', 0.01)),
        'warmup_steps': safe_int(train_config.get('warmup_steps', 0)),
        'max_grad_norm': safe_float(train_config.get('max_grad_norm', 1.0)),
        'output_dir': args.output_dir
    }
    
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Create trainer
    trainer = SentimentTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=training_config_clean,
        device=device,
        use_wandb=False,
        local_tracker=None
    )
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {training_config_clean['num_epochs']}")
    print(f"  Learning Rate: {training_config_clean['learning_rate']}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Weight Decay: {training_config_clean['weight_decay']}")
    
    # Train
    print("\nStarting training...")
    history = trainer.train()
    
    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    # Save training history
    history_path = os.path.join(args.output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✅ Training history saved to {history_path}")
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, 'final_model.pt')
    torch.save(model.state_dict(), final_model_path)
    print(f"✅ Final model saved to {final_model_path}")
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    test_results = trainer.evaluate(test_dataloader)
    
    # Save test results
    test_results_path = os.path.join(args.output_dir, 'test_results.json')
    test_results_save = {
        'test_loss': float(test_results['test_loss']),
        'test_accuracy': float(test_results['test_accuracy'])
    }
    with open(test_results_path, 'w') as f:
        json.dump(test_results_save, f, indent=2)
    print(f"✅ Test results saved to {test_results_path}")
    
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE!")
    print("="*60)
    print(f"Results saved to {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()