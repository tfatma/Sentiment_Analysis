# src/evaluation/metrics.py
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import pandas as pd


class MetricsCalculator:
    """
    Comprehensive metrics calculation for sentiment analysis.
    
    This class goes beyond simple accuracy to provide:
    - Domain-specific performance
    - Class-specific metrics
    - Calibration analysis (confidence reliability)
    - Confusion matrices
    
    Interview talking point:
    "I implemented a comprehensive evaluation framework that measures 
    not just accuracy, but also domain transfer, calibration, and 
    per-class performance to ensure model reliability in production."
    """
    
    def __init__(self, class_names: List[str] = None):
        """
        Initialize with class names for better reporting.
        
        Args:
            class_names: List of sentiment class names
                        Default: ['Negative', 'Neutral', 'Positive']
        """
        self.class_names = class_names or ['Negative', 'Neutral', 'Positive']
        self.domain_names = ['electronics', 'books', 'clothing', 'movies']
    
    def calculate_basic_metrics(
        self, 
        y_true: List[int], 
        y_pred: List[int]
    ) -> Dict[str, float]:
        """
        Calculate standard classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with accuracy, precision, recall, F1 scores
            
        Metrics explained:
        - accuracy: Overall correct predictions
        - f1_macro: Average F1 across classes (treats classes equally)
        - f1_weighted: Weighted F1 (accounts for class imbalance)
        - f1_micro: Global F1 (same as accuracy for multi-class)
        
        Why multiple F1 scores?
        - Macro: Good when classes are equally important
        - Weighted: Better for imbalanced datasets
        - Micro: Overall performance measure
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
            'f1_micro': f1_score(y_true, y_pred, average='micro'),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted')
        }
    
    def calculate_domain_specific_metrics(
        self, 
        y_true: List[int], 
        y_pred: List[int], 
        domains: List[int]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate metrics for each domain separately.
        
        This is crucial for:
        - Understanding domain adaptation
        - Identifying which domains need improvement
        - Domain transfer analysis
        
        Interview insight:
        "By measuring per-domain performance, I can identify if the model
        generalizes well across domains or if certain domains need 
        specialized attention. This is key for production deployment."
        
        Example output:
        {
            'electronics': {'accuracy': 0.87, 'f1_weighted': 0.86, ...},
            'books': {'accuracy': 0.91, 'f1_weighted': 0.90, ...},
            'movies': {'accuracy': 0.93, 'f1_weighted': 0.92, ...}
        }
        """
        domain_metrics = {}
        
        for domain_idx, domain_name in enumerate(self.domain_names):
            # Get samples for this domain
            domain_mask = np.array(domains) == domain_idx
            if not np.any(domain_mask):
                continue
            
            domain_true = np.array(y_true)[domain_mask]
            domain_pred = np.array(y_pred)[domain_mask]
            
            domain_metrics[domain_name] = self.calculate_basic_metrics(
                domain_true.tolist(), domain_pred.tolist()
            )
            domain_metrics[domain_name]['sample_count'] = len(domain_true)
        
        return domain_metrics
    
    def calculate_class_specific_metrics(
        self, 
        y_true: List[int], 
        y_pred: List[int]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate metrics for each sentiment class.
        
        Why this matters:
        - Identifies if model is biased toward certain sentiments
        - Helps understand error patterns
        - Important for balanced performance
        
        Example:
        Model might be great at detecting positive sentiment (F1=0.95)
        but poor at neutral (F1=0.65) - this needs fixing!
        
        Returns:
        {
            'Negative': {'precision': 0.85, 'recall': 0.83, 'f1': 0.84},
            'Neutral': {'precision': 0.72, 'recall': 0.68, 'f1': 0.70},
            'Positive': {'precision': 0.91, 'recall': 0.94, 'f1': 0.92}
        }
        """
        # Per-class precision, recall, f1
        precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        class_metrics = {}
        for i, class_name in enumerate(self.class_names):
            if i < len(precision):
                class_metrics[class_name] = {
                    'precision': precision[i],
                    'recall': recall[i],
                    'f1': f1[i]
                }
        
        return class_metrics
    
    def calculate_calibration_metrics(
        self, 
        y_true: List[int], 
        y_probs: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate calibration metrics (confidence reliability).
        
        What is calibration?
        - If model says 80% confidence, it should be right 80% of the time
        - Uncalibrated models might say 95% but only be right 70%
        
        ECE (Expected Calibration Error):
        - Measures gap between confidence and accuracy
        - Lower is better (0 = perfect calibration)
        
        Why calibration matters:
        - Production systems need reliable confidence scores
        - Helps with decision thresholds
        - Critical for risk-sensitive applications
        
        Interview point:
        "I measure calibration to ensure the model's confidence scores
        are reliable for production use. A well-calibrated model with
        80% confidence should be correct ~80% of the time."
        """
        try:
            # Convert to binary classification for calibration
            y_true_binary = np.array(y_true) > 1  # positive vs negative/neutral
            y_probs_positive = y_probs[:, 2] if y_probs.shape[1] > 2 else y_probs[:, 1]
            
            # Calculate calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true_binary, y_probs_positive, n_bins=10
            )
            
            # Expected Calibration Error (ECE)
            bin_boundaries = np.linspace(0, 1, 11)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_probs_positive > bin_lower) & (y_probs_positive <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = y_true_binary[in_bin].mean()
                    avg_confidence_in_bin = y_probs_positive[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            return {
                'ece': ece,
                'max_calibration_error': np.max(np.abs(fraction_of_positives - mean_predicted_value))
            }
        
        except Exception as e:
            print(f"Warning: Could not calculate calibration metrics: {e}")
            return {'ece': 0.0, 'max_calibration_error': 0.0}
    
    def calculate_all_metrics(
        self, 
        y_true: List[int], 
        y_pred: List[int], 
        y_probs: Optional[np.ndarray] = None,
        domains: Optional[List[int]] = None
    ) -> Dict[str, any]:
        """
        Calculate all metrics in one go.
        
        This is the main method used during training validation.
        
        Returns comprehensive metrics:
        - Basic metrics (accuracy, F1, etc.)
        - Domain-specific performance
        - Class-specific performance  
        - Calibration metrics
        
        All keys are prefixed with 'val/' for validation metrics
        to distinguish from training metrics in logs.
        """
        metrics = {}
        
        # Add val/ prefix for validation metrics
        basic_metrics = self.calculate_basic_metrics(y_true, y_pred)
        for key, value in basic_metrics.items():
            metrics[f'val/{key}'] = value
        
        # Domain-specific metrics
        if domains is not None:
            domain_metrics = self.calculate_domain_specific_metrics(y_true, y_pred, domains)
            for domain, domain_vals in domain_metrics.items():
                for metric, value in domain_vals.items():
                    metrics[f'val/{domain}_{metric}'] = value
        
        # Class-specific metrics
        class_metrics = self.calculate_class_specific_metrics(y_true, y_pred)
        for class_name, class_vals in class_metrics.items():
            for metric, value in class_vals.items():
                metrics[f'val/{class_name.lower()}_{metric}'] = value
        
        # Calibration metrics
        if y_probs is not None:
            calibration_metrics = self.calculate_calibration_metrics(y_true, y_probs)
            for key, value in calibration_metrics.items():
                metrics[f'val/{key}'] = value
        
        return metrics
    
    def generate_classification_report(
        self, 
        y_true: List[int], 
        y_pred: List[int]
    ) -> str:
        """
        Generate detailed sklearn classification report.
        
        This gives a nice formatted table showing:
        - Precision, recall, F1 for each class
        - Support (number of samples per class)
        - Macro and weighted averages
        
        Perfect for including in experiment reports!
        """
        return classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            digits=4
        )