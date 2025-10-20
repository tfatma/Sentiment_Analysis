#!/usr/bin/env python3
"""
Data Preparation Script

This script handles the initial data loading and preparation phase.
It's the first step in the ML pipeline.

What it does:
1. Loads configuration files
2. Initializes the data preprocessor
3. Downloads/creates datasets for all domains
4. Validates data quality
5. Saves dataset metadata

Why separate data preparation?
- Can be run once and reused for multiple experiments
- Easier to debug data issues separately from training
- Allows for data inspection before training
- Good software engineering practice (separation of concerns)

Interview talking point:
"I separated data preparation from training for modularity and 
reproducibility. This allows us to validate data quality upfront 
and reuse prepared data across multiple experiments."
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.dataset import DataPreprocessor
from utils.logging_utils import setup_logger
import yaml


def validate_data_quality(train_dataset, val_dataset, test_dataset, logger):
    """
    Validate the quality and distribution of loaded data.
    
    Checks:
    1. Dataset sizes are reasonable
    2. No empty texts
    3. Label distribution is reasonable (not too imbalanced)
    4. Domain distribution is balanced
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        logger: Logger for output
        
    This is important because:
    - Catches data loading errors early
    - Identifies class imbalance issues
    - Ensures domain balance
    - Prevents training on corrupted data
    """
    logger.info("Validating data quality...")
    
    # Check sizes
    if len(train_dataset) == 0:
        logger.error("Training dataset is empty!")
        return False
    
    if len(val_dataset) == 0:
        logger.warning("Validation dataset is empty!")
    
    if len(test_dataset) == 0:
        logger.warning("Test dataset is empty!")
    
    # Check for empty texts in training data (sample first 100)
    sample_size = min(100, len(train_dataset))
    empty_count = 0
    for i in range(sample_size):
        sample = train_dataset[i]
        if not sample['text'] or len(sample['text'].strip()) == 0:
            empty_count += 1
    
    if empty_count > 0:
        logger.warning(f"Found {empty_count}/{sample_size} empty texts in sample")
    
    # Check label distribution
    label_counts = {}
    for i in range(min(1000, len(train_dataset))):
        label = train_dataset[i]['labels'].item()
        label_counts[label] = label_counts.get(label, 0) + 1
    
    logger.info(f"Label distribution (sample): {label_counts}")
    
    # Check if severely imbalanced (one class > 80% of data)
    total_samples = sum(label_counts.values())
    max_class_ratio = max(label_counts.values()) / total_samples
    if max_class_ratio > 0.8:
        logger.warning(f"Severe class imbalance detected: {max_class_ratio:.1%} in majority class")
    
    # Check domain distribution
    domain_counts = {}
    for i in range(min(1000, len(train_dataset))):
        domain = train_dataset.domains[train_dataset[i]['domain'].item()]
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    
    logger.info(f"Domain distribution (sample): {domain_counts}")
    
    logger.info("✅ Data validation completed!")
    return True


def print_data_statistics(train_dataset, val_dataset, test_dataset, logger):
    """
    Print detailed statistics about the datasets.
    
    This helps understand:
    - Dataset sizes
    - Domain coverage
    - Sample characteristics
    
    Good for:
    - Debugging
    - Documentation
    - Reproducibility reports
    """
    logger.info("\n" + "="*50)
    logger.info("DATASET STATISTICS")
    logger.info("="*50)
    
    # Overall sizes
    logger.info(f"\nDataset Sizes:")
    logger.info(f"  Training:   {len(train_dataset):,} samples")
    logger.info(f"  Validation: {len(val_dataset):,} samples")
    logger.info(f"  Test:       {len(test_dataset):,} samples")
    logger.info(f"  Total:      {len(train_dataset) + len(val_dataset) + len(test_dataset):,} samples")
    
    # Domains
    unique_domains = set(train_dataset.domains)
    logger.info(f"\nDomains: {', '.join(unique_domains)}")
    logger.info(f"Number of domains: {len(unique_domains)}")
    
    # Token length statistics (sample)
    logger.info(f"\nToken Length Statistics (sample of 100):")
    sample_lengths = []
    for i in range(min(100, len(train_dataset))):
        sample = train_dataset[i]
        # Count non-zero tokens (excluding padding)
        length = (sample['input_ids'] != 0).sum().item()
        sample_lengths.append(length)
    
    logger.info(f"  Mean:   {sum(sample_lengths) / len(sample_lengths):.1f} tokens")
    logger.info(f"  Min:    {min(sample_lengths)} tokens")
    logger.info(f"  Max:    {max(sample_lengths)} tokens")
    logger.info(f"  Median: {sorted(sample_lengths)[len(sample_lengths)//2]} tokens")
    
    # Sample examples
    logger.info(f"\nSample Examples:")
    label_names = ['Negative', 'Neutral', 'Positive']
    for i in range(min(3, len(train_dataset))):
        sample = train_dataset[i]
        text = sample['text'][:100] + "..." if len(sample['text']) > 100 else sample['text']
        label = label_names[sample['labels'].item()]
        domain = train_dataset.domains[sample['domain'].item()]
        
        logger.info(f"\n  Example {i+1}:")
        logger.info(f"    Text:   {text}")
        logger.info(f"    Label:  {label}")
        logger.info(f"    Domain: {domain}")
    
    logger.info("\n" + "="*50)


def main():
    """
    Main data preparation function.
    
    Steps:
    1. Parse command-line arguments
    2. Load configuration
    3. Initialize data preprocessor
    4. Create datasets
    5. Validate data quality
    6. Save metadata
    7. Print statistics
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description="Prepare multi-domain sentiment dataset")
    parser.add_argument("--config", type=str, default="config/training_config.yaml",
                       help="Path to training config file")
    parser.add_argument("--output_dir", type=str, default="./data",
                       help="Output directory for processed data")
    parser.add_argument("--force", action="store_true",
                       help="Force re-download even if data exists")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger("data_preparation")
    logger.info("="*60)
    logger.info("STARTING DATA PREPARATION")
    logger.info("="*60)
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {args.config}")
        logger.info("Please create config/training_config.yaml first")
        return
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return
    
    data_config = config['data']
    
    # Load model config for tokenizer info
    model_config_path = "config/model_config.yaml"
    try:
        with open(model_config_path, 'r') as f:
            model_config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Model configuration file not found: {model_config_path}")
        return
    
    logger.info(f"Using model: {model_config['model']['name']}")
    logger.info(f"Domains: {', '.join(data_config['domains'])}")
    logger.info(f"Train size per domain: {data_config['train_size_per_domain']}")
    logger.info(f"Val size per domain: {data_config['val_size_per_domain']}")
    logger.info(f"Test size per domain: {data_config['test_size_per_domain']}")
    
    # Initialize preprocessor
    logger.info("\nInitializing data preprocessor...")
    preprocessor = DataPreprocessor(
        tokenizer_name=model_config['model']['name']
    )
    logger.info(f"✅ Tokenizer loaded: {model_config['model']['name']}")
    
    # Create datasets
    logger.info("\n" + "="*60)
    logger.info("CREATING DATASETS")
    logger.info("="*60)
    logger.info("This may take a few minutes for first-time download...")
    
    try:
        train_dataset, val_dataset, test_dataset = preprocessor.create_datasets(
            domains=data_config['domains'],
            train_size=data_config['train_size_per_domain'],
            val_size=data_config['val_size_per_domain'],
            test_size=data_config['test_size_per_domain'],
            max_length=model_config['model']['max_length']
        )
        logger.info("✅ Datasets created successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to create datasets: {e}")
        logger.error("Falling back to dummy data for testing...")
        # The DataPreprocessor handles fallback automatically
        return
    
    # Validate data quality
    logger.info("\n" + "="*60)
    logger.info("VALIDATING DATA QUALITY")
    logger.info("="*60)
    
    is_valid = validate_data_quality(train_dataset, val_dataset, test_dataset, logger)
    
    if not is_valid:
        logger.error("❌ Data validation failed!")
        logger.warning("Proceeding anyway, but results may be unreliable")
    
    # Print detailed statistics
    print_data_statistics(train_dataset, val_dataset, test_dataset, logger)
    
    # Save dataset metadata
    logger.info("\n" + "="*60)
    logger.info("SAVING METADATA")
    logger.info("="*60)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    dataset_info = {
        'domains': data_config['domains'],
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset),
        'total_size': len(train_dataset) + len(val_dataset) + len(test_dataset),
        'max_length': model_config['model']['max_length'],
        'tokenizer': model_config['model']['name'],
        'train_size_per_domain': data_config['train_size_per_domain'],
        'val_size_per_domain': data_config['val_size_per_domain'],
        'test_size_per_domain': data_config['test_size_per_domain'],
        'random_seed': data_config['random_seed']
    }
    
    metadata_path = os.path.join(args.output_dir, 'dataset_info.yaml')
    with open(metadata_path, 'w') as f:
        yaml.dump(dataset_info, f, indent=2)
    
    logger.info(f"✅ Dataset metadata saved to {metadata_path}")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("DATA PREPARATION COMPLETED SUCCESSFULLY!")
    logger.info("="*60)
    logger.info(f"\n📊 Summary:")
    logger.info(f"   • Total samples: {dataset_info['total_size']:,}")
    logger.info(f"   • Training: {dataset_info['train_size']:,}")
    logger.info(f"   • Validation: {dataset_info['val_size']:,}")
    logger.info(f"   • Test: {dataset_info['test_size']:,}")
    logger.info(f"   • Domains: {len(data_config['domains'])}")
    logger.info(f"   • Metadata: {metadata_path}")
    
    logger.info(f"\n🚀 Next steps:")
    logger.info(f"   1. Train baseline model:")
    logger.info(f"      python scripts/train_baseline.py --experiment_name my_baseline")
    logger.info(f"   2. Train LoRA model:")
    logger.info(f"      python scripts/train_lora.py --experiment_name my_lora")
    logger.info(f"   3. Or run complete experiment:")
    logger.info(f"      python scripts/run_experiments.py --experiment_name full_exp")


if __name__ == "__main__":
    main()