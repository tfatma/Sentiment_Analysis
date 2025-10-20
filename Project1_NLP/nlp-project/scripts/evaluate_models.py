#!/usr/bin/env python3
"""
Model Evaluation and Comparison Script

This script provides comprehensive evaluation of trained models:
1. Performance metrics (accuracy, F1, precision, recall)
2. Domain-specific analysis
3. Error analysis with confidence breakdown
4. Model comparison (baseline vs LoRA)
5. Domain transfer analysis
6. Visualizations and reports

Why comprehensive evaluation matters:
- Beyond accuracy: Understand WHERE and WHY model fails
- Domain transfer: Does model generalize?
- Confidence calibration: Are predictions reliable?
- Model comparison: Quantify efficiency vs performance trade-off

Interview talking point:
"I implemented a comprehensive evaluation framework that goes beyond 
simple accuracy. It includes error analysis, domain transfer testing, 
and confidence calibration - all critical for production deployment. 
This helps identify edge cases and guides model improvements."

Usage examples:
# Evaluate single model
python scripts/evaluate_models.py --model_configs lora:results/lora/best_model

# Compare baseline vs LoRA
python scripts/evaluate_models.py --model_configs \\
  baseline:results/baseline/best_model \\
  lora:results/lora/best_model

# With domain transfer analysis
python scripts/evaluate_models.py --model_configs \\
  lora:results/lora/best_model \\
  --test_domain_transfer
"""

import os
import sys
import argparse
import yaml
import json
from pathlib import Path
from torch.utils.data import DataLoader, Subset
import torch

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.dataset import DataPreprocessor
from models.base_model import BaseSentimentModel
from models.lora_model import LoRASentimentModel
from evaluation.evaluator import ModelEvaluator
from training.utils import set_seed, get_device
from utils.logging_utils import get_experiment_logger


def load_model(model_type: str, model_path: str, model_config: dict, device):
    """
    Load a trained model from checkpoint.
    
    This handles both baseline and LoRA models with proper error handling.
    
    Args:
        model_type: 'baseline' or 'lora'
        model_path: Path to model checkpoint
        model_config: Model configuration dict
        device: Device to load model on
        
    Returns:
        Loaded model
        
    Why this is tricky:
    - LoRA models save adapters differently
    - Need to handle missing files gracefully
    - Must move model to correct device
    """
    logger = get_experiment_logger('evaluation')
    
    if model_type == 'baseline':
        logger.info(f"Loading baseline model from {model_path}")
        model = BaseSentimentModel(
            model_name=model_config['model']['name'],
            num_labels=model_config['model']['num_labels']
        )
        
        # Load state dict if available
        checkpoint_path = os.path.join(model_path, 'pytorch_model.bin')
        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict)
            logger.info("✅ Baseline model loaded successfully")
        else:
            logger.warning(f"No checkpoint found at {checkpoint_path}, using untrained model")
    
    elif model_type == 'lora':
        logger.info(f"Loading LoRA model from {model_path}")
        model = LoRASentimentModel(
            model_name=model_config['model']['name'],
            num_labels=model_config['model']['num_labels'],
            lora_config=model_config['lora']
        )
        
        # Load LoRA adapters if available
        adapter_config_path = os.path.join(model_path, 'adapter_config.json')
        if os.path.exists(adapter_config_path):
            try:
                model.load_adapter(model_path)
                logger.info("✅ LoRA adapters loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load LoRA adapters: {e}")
        else:
            logger.warning(f"No LoRA adapters found at {model_path}, using untrained model")
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'baseline' or 'lora'")
    
    model.to(device)
    model.eval()
    
    return model


def print_evaluation_results(model_name: str, results: dict):
    """
    Print evaluation results in a nice formatted way.
    
    Args:
        model_name: Name of the model
        results: Evaluation results dictionary
    """
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS: {model_name}")
    print(f"{'='*60}")
    
    metrics = results['metrics']
    
    # Overall metrics
    print(f"\n📊 Overall Performance:")
    print(f"  Accuracy:        {metrics.get('val/accuracy', 0):.4f}")
    print(f"  F1 (Weighted):   {metrics.get('val/f1_weighted', 0):.4f}")
    print(f"  F1 (Macro):      {metrics.get('val/f1_macro', 0):.4f}")
    print(f"  Precision:       {metrics.get('val/precision_weighted', 0):.4f}")
    print(f"  Recall:          {metrics.get('val/recall_weighted', 0):.4f}")
    print(f"  Loss:            {results['loss']:.4f}")
    
    # Per-class metrics
    print(f"\n📈 Per-Class Performance:")
    for class_name in ['negative', 'neutral', 'positive']:
        f1_key = f'val/{class_name}_f1'
        precision_key = f'val/{class_name}_precision'
        recall_key = f'val/{class_name}_recall'
        
        if f1_key in metrics:
            print(f"  {class_name.capitalize():10s} - F1: {metrics[f1_key]:.4f}, "
                  f"Precision: {metrics.get(precision_key, 0):.4f}, "
                  f"Recall: {metrics.get(recall_key, 0):.4f}")
    
    # Domain-specific metrics
    print(f"\n🌐 Domain-Specific Performance:")
    domains = ['electronics', 'books', 'clothing', 'movies']
    for domain in domains:
        acc_key = f'val/{domain}_accuracy'
        f1_key = f'val/{domain}_f1_weighted'
        
        if acc_key in metrics:
            print(f"  {domain.capitalize():12s} - Accuracy: {metrics[acc_key]:.4f}, "
                  f"F1: {metrics.get(f1_key, 0):.4f}")
    
    # Calibration metrics
    if 'val/ece' in metrics:
        print(f"\n🎯 Calibration Metrics:")
        print(f"  ECE (Expected Calibration Error): {metrics['val/ece']:.4f}")
        print(f"  Max Calibration Error:            {metrics.get('val/max_calibration_error', 0):.4f}")
        print(f"  (Lower is better - 0 is perfect calibration)")
    
    print(f"\n{'='*60}\n")


def main():
    """
    Main evaluation function.
    
    Flow:
    1. Parse arguments and load configs
    2. Load test dataset
    3. Load all models to evaluate
    4. Run evaluation on each model
    5. Compare models if multiple provided
    6. Perform domain transfer analysis if requested
    7. Save all results and visualizations
    """
    # ============================================
    # STEP 1: Parse Arguments
    # ============================================
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--model_configs", nargs='+', required=True,
                       help="List of model configs in format 'model_type:model_path'")
    parser.add_argument("--data_config", type=str, default="config/training_config.yaml",
                       help="Path to training config (for data settings)")
    parser.add_argument("--model_config", type=str, default="config/model_config.yaml",
                       help="Path to model config")
    parser.add_argument("--output_dir", type=str, default="./results/evaluation",
                       help="Output directory for evaluation results")
    parser.add_argument("--test_domain_transfer", action="store_true",
                       help="Test domain transfer performance")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for evaluation")
    
    args = parser.parse_args()
    
    print("="*70)
    print("MODEL EVALUATION AND COMPARISON")
    print("="*70)
    
    # ============================================
    # STEP 2: Load Configurations
    # ============================================
    print("\n📋 Loading configurations...")
    
    try:
        with open(args.model_config, 'r') as f:
            model_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ Model config not found: {args.model_config}")
        return
    
    try:
        with open(args.data_config, 'r') as f:
            data_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ Data config not found: {args.data_config}")
        return
    
    # Set seed for reproducibility
    set_seed(data_config['data']['random_seed'])
    
    # Setup logging
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    logger = get_experiment_logger('evaluation', os.path.join(args.output_dir, 'logs'))
    
    # ============================================
    # STEP 3: Prepare Test Dataset
    # ============================================
    print("\n📊 Preparing test dataset...")
    logger.info("Preparing test dataset...")
    
    preprocessor = DataPreprocessor(model_config['model']['name'])
    
    try:
        train_dataset, val_dataset, test_dataset = preprocessor.create_datasets(
            domains=data_config['data']['domains'],
            train_size=data_config['data']['train_size_per_domain'],
            val_size=data_config['data']['val_size_per_domain'],
            test_size=data_config['data']['test_size_per_domain'],
            max_length=model_config['model']['max_length']
        )
        print(f"✅ Test dataset prepared: {len(test_dataset)} samples")
    except Exception as e:
        logger.error(f"Failed to prepare test dataset: {e}")
        return
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # ============================================
    # STEP 4: Load Models
    # ============================================
    print(f"\n🤖 Loading models...")
    
    device = get_device()
    print(f"Using device: {device}")
    
    models = {}
    for model_config_str in args.model_configs:
        parts = model_config_str.split(':')
        if len(parts) != 2:
            print(f"⚠️  Invalid model config format: {model_config_str}")
            print("   Use format: 'model_type:model_path'")
            continue
        
        model_type, model_path = parts
        model_name = f"{model_type}_{os.path.basename(model_path)}"
        
        print(f"Loading {model_type} model from {model_path}...")
        try:
            models[model_name] = load_model(model_type, model_path, model_config, device)
            print(f"✅ {model_name} loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {e}")
            logger.error(f"Failed to load {model_name}: {e}")
            continue
    
    if not models:
        print("❌ No models were successfully loaded!")
        return
    
    print(f"\n✅ Loaded {len(models)} model(s) for evaluation")
    
    # ============================================
    # STEP 5: Evaluate Each Model
    # ============================================
    print(f"\n{'='*70}")
    print("INDIVIDUAL MODEL EVALUATION")
    print(f"{'='*70}")
    
    evaluator = ModelEvaluator(device=device)
    individual_results = {}
    
    for model_name, model in models.items():
        print(f"\n🔍 Evaluating {model_name}...")
        logger.info(f"Evaluating {model_name}...")
        
        try:
            results = evaluator.evaluate_model(model, test_dataloader, return_predictions=True)
            individual_results[model_name] = results
            
            # Print results
            print_evaluation_results(model_name, results)
            
            # Save individual results
            model_output_dir = os.path.join(args.output_dir, model_name)
            os.makedirs(model_output_dir, exist_ok=True)
            
            # Save metrics
            with open(os.path.join(model_output_dir, 'evaluation_results.json'), 'w') as f:
                json_results = {
                    'metrics': results['metrics'],
                    'loss': results['loss'],
                    'classification_report': results['classification_report']
                }
                json.dump(json_results, f, indent=2)
            
            # Save classification report
            with open(os.path.join(model_output_dir, 'classification_report.txt'), 'w') as f:
                f.write(results['classification_report'])
            
            # Error analysis
            print(f"📊 Performing error analysis for {model_name}...")
            error_analysis = evaluator.analyze_errors(model, test_dataloader, model_output_dir)
            
            with open(os.path.join(model_output_dir, 'error_analysis.json'), 'w') as f:
                json.dump(error_analysis, f, indent=2, default=str)
            
            print(f"✅ Error analysis completed for {model_name}")
            
        except Exception as e:
            print(f"❌ Evaluation failed for {model_name}: {e}")
            logger.error(f"Evaluation failed for {model_name}: {e}")
            continue
    
    # ============================================
    # STEP 6: Compare Models
    # ============================================
    if len(models) > 1:
        print(f"\n{'='*70}")
        print("MODEL COMPARISON")
        print(f"{'='*70}")
        
        logger.info("Comparing models...")
        
        try:
            comparison_results = evaluator.compare_models(models, test_dataloader, args.output_dir)
            
            # Display comparison
            comparison_df = comparison_results['comparison']
            print(f"\n{comparison_df.to_string(index=False)}")
            
            # Save comparison
            comparison_df.to_csv(os.path.join(args.output_dir, 'model_comparison.csv'), index=False)
            print(f"\n✅ Model comparison saved to {args.output_dir}/model_comparison.csv")
            
        except Exception as e:
            print(f"❌ Model comparison failed: {e}")
            logger.error(f"Model comparison failed: {e}")
    
    # ============================================
    # STEP 7: Domain Transfer Analysis
    # ============================================
    if args.test_domain_transfer:
        print(f"\n{'='*70}")
        print("DOMAIN TRANSFER ANALYSIS")
        print(f"{'='*70}")
        
        logger.info("Performing domain transfer analysis...")
        
        # Create domain-specific dataloaders
        domain_dataloaders = {}
        for domain in data_config['data']['domains']:
            # Filter test dataset by domain
            domain_indices = [
                i for i in range(len(test_dataset))
                if test_dataset.domains[test_dataset[i]['domain'].item()] == domain
            ]
            
            if domain_indices:
                domain_subset = Subset(test_dataset, domain_indices)
                domain_dataloaders[domain] = DataLoader(
                    domain_subset, batch_size=args.batch_size, shuffle=False
                )
                print(f"  {domain}: {len(domain_indices)} samples")
        
        # Analyze each model
        for model_name, model in models.items():
            print(f"\n🌐 Domain transfer analysis for {model_name}...")
            
            try:
                transfer_results = evaluator.domain_transfer_analysis(
                    model, domain_dataloaders,
                    os.path.join(args.output_dir, model_name)
                )
                
                with open(os.path.join(args.output_dir, model_name, 'domain_transfer.json'), 'w') as f:
                    json.dump(transfer_results, f, indent=2, default=str)
                
                print(f"✅ Domain transfer analysis completed for {model_name}")
                
            except Exception as e:
                print(f"❌ Domain transfer analysis failed for {model_name}: {e}")
                logger.error(f"Domain transfer analysis failed: {e}")
    
    # ============================================
    # STEP 8: Summary
    # ============================================
    print(f"\n{'='*70}")
    print("EVALUATION COMPLETED!")
    print(f"{'='*70}")
    print(f"\n📁 Results saved to: {args.output_dir}")
    print(f"\n📊 Generated files:")
    print(f"  • Individual results: {args.output_dir}/[model_name]/")
    print(f"  • Comparison table:   {args.output_dir}/model_comparison.csv")
    print(f"  • Visualizations:     {args.output_dir}/[model_name]/*.png")
    print(f"  • Error analysis:     {args.output_dir}/[model_name]/error_analysis.json")
    
    if args.test_domain_transfer:
        print(f"  • Domain transfer:    {args.output_dir}/[model_name]/domain_transfer.json")
    
    print(f"\n🎯 Key insights:")
    for model_name, results in individual_results.items():
        metrics = results['metrics']
        print(f"  {model_name}:")
        print(f"    - Accuracy: {metrics.get('val/accuracy', 0):.4f}")
        print(f"    - F1 Score: {metrics.get('val/f1_weighted', 0):.4f}")


if __name__ == "__main__":
    main()