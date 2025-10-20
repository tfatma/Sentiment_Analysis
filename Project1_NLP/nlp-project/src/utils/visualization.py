# src/utils/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import torch


class VisualizationUtils:
    """
    Comprehensive visualization utilities for ML experiments.
    
    Why visualization matters:
    - Quickly spot training issues (overfitting, convergence)
    - Compare models visually
    - Create presentation-ready charts
    - Communicate results to stakeholders
    
    Interview talking point:
    "I created reusable visualization utilities to quickly analyze 
    experiments and communicate results. Good visualizations help 
    identify issues like overfitting and guide model improvements."
    """
    
    def __init__(self, style: str = 'default'):
        """
        Initialize visualization settings.
        
        Args:
            style: Matplotlib style to use
        """
        plt.style.use(style)
        sns.set_palette("husl")
        
        # Set default figure parameters for consistency
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['font.size'] = 10
    
    @staticmethod
    def plot_training_history(
        history: List[Dict],
        metrics: List[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot training history showing loss and metrics over epochs.
        
        This is THE most important plot for understanding training!
        
        What to look for:
        - Decreasing train loss: Good, model is learning
        - Decreasing val loss: Good, model generalizes
        - Train loss << Val loss: Overfitting! Need regularization
        - Both losses plateau: Converged (or need more capacity)
        - Val loss increases: Definite overfitting
        
        Args:
            history: List of dictionaries with metrics per epoch
            metrics: Which metrics to plot
            save_path: Where to save the figure
            
        Returns:
            Matplotlib figure object
            
        Example history:
        [
            {'epoch': 1, 'train/loss': 0.5, 'val/loss': 0.6, 'val/accuracy': 0.75},
            {'epoch': 2, 'train/loss': 0.3, 'val/loss': 0.4, 'val/accuracy': 0.85},
            ...
        ]
        
        Creates subplots showing:
        - Training vs validation loss
        - Accuracy over time
        - F1 score over time
        """
        if metrics is None:
            metrics = ['train/epoch_loss', 'val/loss', 'val/accuracy', 'val/f1_weighted']
        
        # Convert history to DataFrame for easier handling
        df = pd.DataFrame(history)
        epochs = range(1, len(df) + 1)
        
        # Create subplots
        n_metrics = len(metrics)
        n_cols = 2
        n_rows = (n_metrics + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        colors = sns.color_palette("husl", n_metrics)
        
        for i, metric in enumerate(metrics):
            if metric in df.columns:
                ax = axes[i]
                ax.plot(epochs, df[metric], marker='o', color=colors[i], 
                       linewidth=2, markersize=6, label=metric)
                
                # Formatting
                ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
                ax.set_xlabel('Epoch', fontsize=10)
                ax.set_ylabel(metric.split('/')[-1].replace('_', ' ').title(), fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.legend(loc='best')
                
                # Add min/max annotations for validation metrics
                if 'val' in metric:
                    best_idx = df[metric].idxmax() if 'accuracy' in metric or 'f1' in metric else df[metric].idxmin()
                    best_value = df[metric].iloc[best_idx]
                    ax.annotate(f'Best: {best_value:.4f}', 
                              xy=(best_idx + 1, best_value),
                              xytext=(10, 10), textcoords='offset points',
                              bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                              arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # Hide empty subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Training history saved to {save_path}")
        
        return fig
    
    @staticmethod
    def plot_parameter_comparison(
        model_params: Dict[str, Dict[str, int]],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize parameter efficiency comparison between models.
        
        This is CRUCIAL for demonstrating LoRA efficiency!
        
        Creates 3 charts:
        1. Total parameters (shows both models are similar size)
        2. Trainable parameters (LoRA has 200x fewer!)
        3. Trainable percentage (LoRA is <1%)
        
        Args:
            model_params: Dict with model names and their parameter counts
            save_path: Where to save the figure
            
        Example:
            model_params = {
                'Baseline': {
                    'total': 125000000,
                    'trainable': 125000000,
                    'trainable_percent': 100.0
                },
                'LoRA': {
                    'total': 125000000,
                    'trainable': 589824,
                    'trainable_percent': 0.47
                }
            }
            
        Interview gold:
        "This visualization clearly shows LoRA achieves 200x parameter 
        efficiency - only 0.47% trainable parameters vs 100% for baseline."
        """
        models = list(model_params.keys())
        
        # Extract data
        total_params = [model_params[model]['total'] for model in models]
        trainable_params = [model_params[model]['trainable'] for model in models]
        trainable_percent = [model_params[model]['trainable_percent'] for model in models]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. Total parameters
        bars1 = axes[0].bar(models, total_params, alpha=0.8, color='skyblue')
        axes[0].set_title('Total Parameters', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Number of Parameters', fontsize=11)
        axes[0].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height/1e6:.1f}M', ha='center', va='bottom', fontsize=10)
        
        # 2. Trainable parameters (LOG SCALE for better visualization!)
        bars2 = axes[1].bar(models, trainable_params, alpha=0.8, color='lightcoral')
        axes[1].set_title('Trainable Parameters', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Number of Parameters (log scale)', fontsize=11)
        axes[1].set_yscale('log')  # Log scale shows the huge difference!
        axes[1].tick_params(axis='x', rotation=45)
        
        for bar in bars2:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height/1e6:.2f}M' if height > 1e6 else f'{height/1e3:.1f}K',
                        ha='center', va='bottom', fontsize=10)
        
        # 3. Trainable percentage
        bars3 = axes[2].bar(models, trainable_percent, alpha=0.8, color='lightgreen')
        axes[2].set_title('Percentage of Trainable Parameters', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('Percentage (%)', fontsize=11)
        axes[2].set_ylim(0, 110)  # Slightly above 100 for labels
        axes[2].tick_params(axis='x', rotation=45)
        
        for bar in bars3:
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height + 2,
                        f'{height:.2f}%', ha='center', va='bottom', 
                        fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Parameter comparison saved to {save_path}")
        
        return fig
    
    @staticmethod
    def plot_domain_distribution(
        domains: List[str],
        labels: List[int],
        class_names: List[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot distribution of classes across domains.
        
        Why this matters:
        - Identifies domain imbalance
        - Shows class distribution per domain
        - Helps explain domain-specific performance
        
        Creates stacked bar chart showing:
        - How many samples per domain
        - Class breakdown within each domain
        
        Args:
            domains: List of domain names for each sample
            labels: List of sentiment labels
            class_names: Names of sentiment classes
            save_path: Where to save
            
        Example:
        Shows that Electronics has 300 samples (100 neg, 150 neu, 50 pos)
        while Movies has 250 samples (50 neg, 50 neu, 150 pos)
        
        This explains why model might perform better on Movies!
        """
        if class_names is None:
            class_names = ['Negative', 'Neutral', 'Positive']
        
        # Create DataFrame
        df = pd.DataFrame({'domain': domains, 'label': labels})
        domain_counts = df.groupby(['domain', 'label']).size().unstack(fill_value=0)
        
        # Create stacked bar plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = sns.color_palette("husl", len(class_names))
        domain_counts.plot(kind='bar', stacked=True, ax=ax, color=colors, alpha=0.8)
        
        ax.set_title('Class Distribution Across Domains', fontsize=14, fontweight='bold')
        ax.set_xlabel('Domain', fontsize=12)
        ax.set_ylabel('Number of Samples', fontsize=12)
        ax.legend(class_names, title='Sentiment', fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        
        # Add total counts on top of bars
        totals = domain_counts.sum(axis=1)
        for i, (domain, total) in enumerate(totals.items()):
            ax.text(i, total + max(totals) * 0.02, str(int(total)), 
                   ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Domain distribution saved to {save_path}")
        
        return fig
    
    @staticmethod
    def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str] = None,
        normalize: bool = True,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot confusion matrix to understand misclassifications.
        
        Confusion matrix shows:
        - Diagonal: Correct predictions
        - Off-diagonal: Confusion between classes
        
        Common patterns:
        - Neutral often confused with Positive/Negative
        - Strong diagonal = good performance
        - High off-diagonal values = systematic errors
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            normalize: Whether to show percentages
            save_path: Where to save
            
        Interview insight:
        "The confusion matrix revealed that 15% of Neutral samples 
        were misclassified as Positive, suggesting the model struggles 
        with borderline cases. This guided us to collect more Neutral 
        examples for training."
        """
        from sklearn.metrics import confusion_matrix
        
        if class_names is None:
            class_names = ['Negative', 'Neutral', 'Positive']
        
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='.2%' if normalize else 'd',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            cbar_kws={'label': 'Percentage' if normalize else 'Count'}
        )
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        title = 'Confusion Matrix (Normalized)' if normalize else 'Confusion Matrix'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Confusion matrix saved to {save_path}")
        
        return fig
    
    @staticmethod
    def plot_learning_curves(
        train_sizes: List[int],
        train_scores: List[float],
        val_scores: List[float],
        metric_name: str = "Accuracy",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot learning curves showing performance vs training data size.
        
        Learning curves help answer:
        - "Will more data help?" (if curves haven't converged)
        - "Is model too complex?" (if big gap between train/val)
        - "Is model too simple?" (if both scores are low)
        
        Args:
            train_sizes: Different training set sizes tested
            train_scores: Performance on training set
            val_scores: Performance on validation set
            metric_name: Name of the metric
            save_path: Where to save
            
        Interpretation:
        - Train score >> Val score: Overfitting
        - Both scores low: Underfitting
        - Curves converging: More data won't help much
        - Curves diverging: More data might help
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(train_sizes, train_scores, 'o-', label='Training Score', 
               linewidth=2, markersize=8)
        ax.plot(train_sizes, val_scores, 'o-', label='Validation Score',
               linewidth=2, markersize=8)
        
        ax.fill_between(train_sizes, train_scores, val_scores, alpha=0.1)
        
        ax.set_xlabel('Training Set Size', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f'Learning Curves ({metric_name})', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Learning curves saved to {save_path}")
        
        return fig