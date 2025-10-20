#!/usr/bin/env python3
"""
test_dataset_loading.py
-----------------------
Test script to verify dataset loading works with the fixed dataset.py
"""

from dataset import DataPreprocessor

print("="*60)
print("Testing Dataset Loading")
print("="*60)

# Initialize preprocessor
print("\n1. Initializing preprocessor with RoBERTa tokenizer...")
preprocessor = DataPreprocessor(tokenizer_name="roberta-base")
print("   ✅ Tokenizer loaded successfully")

# Test with multiple domains
domains = ["electronics", "books", "movies", "restaurants"]
print(f"\n2. Loading datasets for domains: {domains}")

# Create small datasets for testing (small sizes for quick testing)
try:
    train, val, test = preprocessor.create_datasets(
        domains=domains,
        train_size=100,      # Small size for quick testing
        val_size=20,
        test_size=20,
        max_length=128
    )
    
    print("\n" + "="*60)
    print("✅ SUCCESS! All datasets created successfully!")
    print("="*60)
    print(f"\nDataset Sizes:")
    print(f"   • Train: {len(train)} samples")
    print(f"   • Validation: {len(val)} samples")
    print(f"   • Test: {len(test)} samples")
    
    # Test a sample
    print(f"\n📊 Sample from training set:")
    sample = train[0]
    print(f"   • Text (first 100 chars): {sample['text'][:100]}...")
    print(f"   • Label: {sample['labels'].item()}")
    print(f"   • Domain: {sample['domain'].item()}")
    print(f"   • Input shape: {sample['input_ids'].shape}")
    
    print("\n" + "="*60)
    print("🎉 Everything is working! Ready for model training!")
    print("="*60)
    
except Exception as e:
    print(f"\n❌ Error occurred: {e}")
    print("\nPlease check the error message above and fix any issues.")
    import traceback
    traceback.print_exc()