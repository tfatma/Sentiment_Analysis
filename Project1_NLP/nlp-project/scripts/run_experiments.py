#!/usr/bin/env python3
"""
Complete Experiment Pipeline Runner

This script orchestrates the entire ML pipeline from data prep to evaluation.
It's the "one-click" solution to run complete experiments.

What it runs:
1. Data preparation (optional, can skip if already done)
2. Baseline model training
3. LoRA model training
4. Model comparison and evaluation
5. Generates all visualizations and reports

Why this matters:
- Reproducibility: Same command = same results
- Efficiency: Don't repeat yourself
- Documentation: Clear experiment flow
- CI/CD ready: Can be automated

Interview talking point:
"I created an experiment orchestration script that runs the entire
pipeline with a single command. This ensures reproducibility and 
makes it easy to run ablation studies with different hyperparameters."

Usage examples:
# Quick test with small dataset
python scripts/run_experiments.py --quick_mode --experiment_name quick_test

# Full experiment
python scripts/run_experiments.py --experiment_name production_run

# Skip baseline, only train LoRA
python scripts/run_experiments.py --skip_baseline --experiment_name lora_only

# Skip evaluation (just training)
python scripts/run_experiments.py --skip_evaluation --experiment_name train_only
"""

import os
import sys
import subprocess
import argparse
import yaml
from pathlib import Path
import time


def run_command(command: list, description: str) -> bool:
    """
    Run a command and handle errors gracefully.
    
    Args:
        command: List of command parts (e.g., ['python', 'script.py', '--arg'])
        description: Human-readable description
        
    Returns:
        True if successful, False otherwise
        
    Why this function:
    - Centralized error handling
    - Clear output formatting
    - Easy to track which step failed
    """
    print(f"\n{'='*70}")
    print(f"🚀 {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(command)}")
    print(f"{'='*70}\n")
    
    try:
        result = subprocess.run(command, check=True, capture_output=False)
        print(f"\n✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with error code {e.returncode}")
        print(f"Check logs above for details")
        return False
    except FileNotFoundError:
        print(f"\n❌ Command not found: {command[0]}")
        print(f"Make sure all dependencies are installed")
        return False


def update_config_for_quick_mode(config_path: str):
    """
    Modify config for quick testing mode.
    
    Quick mode changes:
    - Smaller dataset (100 samples per domain vs 1000)
    - Fewer epochs (2 vs 3)
    - Faster iteration for testing
    
    Args:
        config_path: Path to training config file
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Reduce dataset sizes
    config['data']['train_size_per_domain'] = 100
    config['data']['val_size_per_domain'] = 20
    config['data']['test_size_per_domain'] = 20
    
    # Reduce epochs
    config['training']['num_epochs'] = 2
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, indent=2)
    
    print("✅ Config updated for quick mode:")
    print(f"  - Dataset: 100 train / 20 val / 20 test per domain")
    print(f"  - Epochs: 2")


def restore_config(backup_path: str, config_path: str):
    """Restore original config from backup."""
    if os.path.exists(backup_path):
        with open(backup_path, 'r') as f:
            config = yaml.safe_load(f)
        with open(config_path, 'w') as f:
            yaml.dump(config, f, indent=2)


def main():
    """
    Main orchestration function.
    
    This coordinates the entire ML pipeline with proper error handling
    and progress tracking.
    """
    # ============================================
    # STEP 1: Parse Arguments
    # ============================================
    parser = argparse.ArgumentParser(
        description="Run complete sentiment analysis experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test run (small dataset, 10 minutes)
  python scripts/run_experiments.py --quick_mode --experiment_name test
  
  # Full production run (1-2 hours)
  python scripts/run_experiments.py --experiment_name production
  
  # Only train LoRA (skip baseline)
  python scripts/run_experiments.py --skip_baseline --experiment_name lora_only
  
  # Train without evaluation
  python scripts/run_experiments.py --skip_evaluation --experiment_name train_only
        """
    )
    
    parser.add_argument("--experiment_name", type=str, default="full_experiment",
                       help="Base name for experiments")
    parser.add_argument("--skip_data_prep", action="store_true",
                       help="Skip data preparation (use if data already prepared)")
    parser.add_argument("--skip_baseline", action="store_true",
                       help="Skip baseline model training")
    parser.add_argument("--skip_lora", action="store_true",
                       help="Skip LoRA model training")
    parser.add_argument("--skip_evaluation", action="store_true",
                       help="Skip model evaluation and comparison")
    parser.add_argument("--quick_mode", action="store_true",
                       help="Use reduced dataset for quick testing (100 samples/domain)")
    
    args = parser.parse_args()
    
    # ============================================
    # STEP 2: Setup
    # ============================================
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    print("="*70)
    print("SENTIMENT ANALYSIS WITH PEFT - COMPLETE EXPERIMENT PIPELINE")
    print("="*70)
    print(f"Experiment name: {args.experiment_name}")
    print(f"Working directory: {project_root}")
    print(f"Quick mode: {'Yes' if args.quick_mode else 'No'}")
    print("="*70)
    
    start_time = time.time()
    
    # Handle quick mode
    config_path = "config/training_config.yaml"
    backup_path = "config/training_config.yaml.backup"
    
    if args.quick_mode:
        print("\n⚡ Quick mode enabled - using reduced dataset")
        # Backup original config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        with open(backup_path, 'w') as f:
            yaml.dump(config, f, indent=2)
        # Update for quick mode
        update_config_for_quick_mode(config_path)
    
    # Track which steps succeeded
    results = {
        'data_prep': None,
        'baseline': None,
        'lora': None,
        'evaluation': None
    }
    
    # ============================================
    # STEP 3: Data Preparation
    # ============================================
    if not args.skip_data_prep:
        success = run_command(
            [sys.executable, "scripts/prepare_data.py",
             "--config", config_path],
            "Data Preparation"
        )
        results['data_prep'] = success
        
        if not success:
            print("\n❌ Data preparation failed. Cannot continue.")
            if args.quick_mode:
                restore_config(backup_path, config_path)
            return
    else:
        print("\n⏭️  Skipping data preparation (assuming data already prepared)")
        results['data_prep'] = 'skipped'
    
    # ============================================
    # STEP 4: Baseline Model Training
    # ============================================
    if not args.skip_baseline:
        success = run_command(
            [sys.executable, "scripts/train_baseline.py",
             "--experiment_name", f"{args.experiment_name}_baseline",
             "--model_config", "config/model_config.yaml",
             "--training_config", config_path],
            "Baseline Model Training"
        )
        results['baseline'] = success
        
        if not success:
            print("\n⚠️  Baseline training failed, but continuing with LoRA...")
    else:
        print("\n⏭️  Skipping baseline training")
        results['baseline'] = 'skipped'
    
    # ============================================
    # STEP 5: LoRA Model Training
    # ============================================
    if not args.skip_lora:
        success = run_command(
            [sys.executable, "scripts/train_lora.py",
             "--experiment_name", f"{args.experiment_name}_lora",
             "--model_config", "config/model_config.yaml",
             "--training_config", config_path],
            "LoRA Model Training"
        )
        results['lora'] = success
        
        if not success:
            print("\n⚠️  LoRA training failed, but continuing...")
    else:
        print("\n⏭️  Skipping LoRA training")
        results['lora'] = 'skipped'
    
    # ============================================
    # STEP 6: Model Evaluation
    # ============================================
    if not args.skip_evaluation:
        # Build model configs for evaluation
        model_configs = []
        baseline_path = f"results/{args.experiment_name}_baseline/best_model"
        lora_path = f"results/{args.experiment_name}_lora/best_model"
        
        if os.path.exists(baseline_path) and results['baseline'] != 'skipped':
            model_configs.append(f"baseline:{baseline_path}")
        
        if os.path.exists(lora_path) and results['lora'] != 'skipped':
            model_configs.append(f"lora:{lora_path}")
        
        if model_configs:
            eval_command = [
                sys.executable, "scripts/evaluate_models.py",
                "--model_configs"] + model_configs + [
                "--output_dir", f"results/{args.experiment_name}_evaluation",
                "--test_domain_transfer"
            ]
            
            success = run_command(eval_command, "Model Evaluation and Comparison")
            results['evaluation'] = success
            
            if not success:
                print("\n⚠️  Model evaluation failed")
        else:
            print("\n⚠️  No trained models found for evaluation")
            print(f"     Baseline path: {baseline_path} (exists: {os.path.exists(baseline_path)})")
            print(f"     LoRA path: {lora_path} (exists: {os.path.exists(lora_path)})")
            results['evaluation'] = 'no_models'
    else:
        print("\n⏭️  Skipping model evaluation")
        results['evaluation'] = 'skipped'
    
    # ============================================
    # STEP 7: Cleanup and Summary
    # ============================================
    
    # Restore original config if in quick mode
    if args.quick_mode:
        restore_config(backup_path, config_path)
        if os.path.exists(backup_path):
            os.remove(backup_path)
        print("\n✅ Original config restored")
    
    # Calculate duration
    total_duration = time.time() - start_time
    hours = int(total_duration // 3600)
    minutes = int((total_duration % 3600) // 60)
    seconds = int(total_duration % 60)
    
    # Print comprehensive summary
    print("\n" + "="*70)
    print("EXPERIMENT PIPELINE COMPLETED!")
    print("="*70)
    
    print(f"\n⏱️  Total Duration: ", end="")
    if hours > 0:
        print(f"{hours}h {minutes}m {seconds}s")
    elif minutes > 0:
        print(f"{minutes}m {seconds}s")
    else:
        print(f"{seconds}s")
    
    print(f"\n📋 Experiment Summary:")
    print(f"   Experiment name: {args.experiment_name}")
    print(f"   Quick mode: {'Yes' if args.quick_mode else 'No'}")
    
    # Status of each step
    def status_icon(status):
        if status == True:
            return "✅"
        elif status == False:
            return "❌"
        elif status == 'skipped':
            return "⏭️ "
        else:
            return "⚠️ "
    
    print(f"\n📊 Pipeline Steps:")
    print(f"   {status_icon(results['data_prep'])} Data Preparation")
    print(f"   {status_icon(results['baseline'])} Baseline Training")
    print(f"   {status_icon(results['lora'])} LoRA Training")
    print(f"   {status_icon(results['evaluation'])} Evaluation")
    
    # Show where results are
    print(f"\n📁 Results Location:")
    print(f"   Base directory: ./results/")
    
    if results['baseline'] == True:
        print(f"   Baseline model: results/{args.experiment_name}_baseline/")
        print(f"                   - Training curves, model checkpoint, logs")
    
    if results['lora'] == True:
        print(f"   LoRA model:     results/{args.experiment_name}_lora/")
        print(f"                   - Training curves, LoRA adapters, logs")
    
    if results['evaluation'] == True:
        print(f"   Evaluation:     results/{args.experiment_name}_evaluation/")
        print(f"                   - Model comparison, domain analysis, error analysis")
    
    # Next steps suggestions
    print(f"\n🚀 Next Steps:")
    
    if results['baseline'] == True or results['lora'] == True:
        print(f"   1. Review training curves:")
        if results['baseline'] == True:
            print(f"      - results/{args.experiment_name}_baseline/training_curves.png")
        if results['lora'] == True:
            print(f"      - results/{args.experiment_name}_lora/training_curves.png")
    
    if results['evaluation'] == True:
        print(f"   2. Check model comparison:")
        print(f"      - results/{args.experiment_name}_evaluation/model_comparison.csv")
        print(f"      - results/{args.experiment_name}_evaluation/model_comparison.png")
    
    if results['baseline'] == True and results['lora'] == True:
        print(f"   3. Compare efficiency:")
        print(f"      cat results/{args.experiment_name}_baseline/results_summary.yaml")
        print(f"      cat results/{args.experiment_name}_lora/results_summary.yaml")
    
    # Final status
    success_count = sum(1 for v in results.values() if v == True)
    total_count = sum(1 for v in results.values() if v not in ['skipped', None, 'no_models'])
    
    if success_count == total_count and total_count > 0:
        print(f"\n🎉 All {total_count} steps completed successfully!")
    elif success_count > 0:
        print(f"\n⚠️  {success_count}/{total_count} steps completed successfully")
        print(f"   Check logs above for failed steps")
    else:
        print(f"\n❌ No steps completed successfully")
        print(f"   Please check the error messages above")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()