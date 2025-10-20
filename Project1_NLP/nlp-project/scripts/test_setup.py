#!/usr/bin/env python3
"""
Installation and Setup Verification Script

This script tests that everything is installed correctly before you start.
It's the first thing you should run after installation.

What it tests:
1. Python version (3.8+)
2. All required packages
3. CUDA/GPU availability
4. Data loading functionality
5. Model initialization
6. Basic training loop

Why this is important:
- Catches installation issues early
- Saves time debugging later
- Validates environment setup
- Gives confidence before starting experiments

Interview talking point:
"I created a comprehensive setup verification script that tests all
dependencies and core functionality. This catches issues early and
ensures the environment is properly configured before training."

Usage:
  python scripts/test_setup.py
"""

import sys
import importlib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def print_header(text):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def print_test(name, passed, details=""):
    """Print test result with formatting."""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} - {name}")
    if details:
        print(f"         {details}")


def test_python_version():
    """Test if Python version is 3.8 or higher."""
    print_header("PYTHON VERSION CHECK")
    
    version = sys.version_info
    is_valid = version.major == 3 and version.minor >= 8
    
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    print_test("Python Version", is_valid, f"Version: {version_str}")
    
    if not is_valid:
        print("    ⚠️  Python 3.8+ required. Please upgrade Python.")
    
    return is_valid


def test_imports():
    """Test if all required packages are installed."""
    print_header("PACKAGE IMPORTS CHECK")
    
    required_packages = {
        'torch': 'PyTorch',
        'transformers': 'Hugging Face Transformers',
        'datasets': 'Hugging Face Datasets',
        'peft': 'Parameter-Efficient Fine-Tuning',
        'sklearn': 'Scikit-learn',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'tqdm': 'TQDM',
        'yaml': 'PyYAML'
    }
    
    failed = []
    
    for package, name in required_packages.items():
        try:
            mod = importlib.import_module(package)
            version = getattr(mod, '__version__', 'unknown')
            print_test(name, True, f"Version: {version}")
        except ImportError as e:
            print_test(name, False, f"Not installed")
            failed.append(package)
    
    if failed:
        print(f"\n    ⚠️  Missing packages: {', '.join(failed)}")
        print(f"    💡 Install with: pip install {' '.join(failed)}")
    
    return len(failed) == 0


def test_cuda():
    """Test CUDA/GPU availability."""
    print_header("GPU/CUDA CHECK")
    
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print_test("CUDA Available", True, f"Device: {device_name}")
            print(f"         Memory: {memory_gb:.1f} GB")
            
            # Test actual GPU operation
            try:
                x = torch.randn(1000, 1000).cuda()
                y = x @ x
                print_test("GPU Operations", True, "Matrix multiplication successful")
            except Exception as e:
                print_test("GPU Operations", False, str(e))
        else:
            print_test("CUDA Available", False, "Will use CPU (slower)")
            print("         💡 Training will be slower without GPU")
            print("         💡 Consider using Google Colab or cloud GPU")
        
        # Check MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print_test("MPS (Apple Silicon)", True, "Available")
        
        return True  # Not critical if GPU unavailable
        
    except Exception as e:
        print_test("CUDA Test", False, str(e))
        return False


def test_data_loading():
    """Test basic data loading functionality."""
    print_header("DATA LOADING CHECK")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from data.dataset import DataPreprocessor
        
        print("  Testing data preprocessor initialization...")
        
        # Use smaller model for testing
        preprocessor = DataPreprocessor("distilbert-base-uncased")
        print_test("DataPreprocessor Init", True)
        
        print("  Creating tiny test dataset...")
        
        # Create minimal dataset
        train_ds, val_ds, test_ds = preprocessor.create_datasets(
            domains=["electronics", "books"],
            train_size=10, 
            val_size=5, 
            test_size=5,
            max_length=64
        )
        
        sizes = f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}"
        print_test("Dataset Creation", True, sizes)
        
        # Test data loading
        sample = train_ds[0]
        has_required_keys = all(key in sample for key in ['input_ids', 'attention_mask', 'labels'])
        print_test("Dataset Format", has_required_keys, "All required keys present")
        
        # Test tokenization
        valid_shape = len(sample['input_ids'].shape) == 1
        print_test("Tokenization", valid_shape, f"Shape: {sample['input_ids'].shape}")
        
        return True
        
    except Exception as e:
        print_test("Data Loading", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_model_initialization():
    """Test model initialization."""
    print_header("MODEL INITIALIZATION CHECK")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from models.base_model import BaseSentimentModel
        from models.lora_model import LoRASentimentModel
        import torch
        
        print("  Testing baseline model...")
        
        # Test baseline model
        baseline_model = BaseSentimentModel(
            model_name="distilbert-base-uncased",
            num_labels=3
        )
        
        baseline_params = baseline_model.get_trainable_parameters()
        print_test("Baseline Model", True, f"{baseline_params:,} parameters")
        
        print("  Testing LoRA model...")
        
        # Test LoRA model
        lora_model = LoRASentimentModel(
            model_name="distilbert-base-uncased",
            num_labels=3,
            lora_config={'r': 8, 'lora_alpha': 16, 'lora_dropout': 0.1}
        )
        
        lora_params = lora_model.get_trainable_parameters()
        efficiency = 100 * lora_params / baseline_params
        print_test("LoRA Model", True, f"{lora_params:,} parameters ({efficiency:.2f}%)")
        
        # Test forward pass
        print("  Testing forward pass...")
        batch_size = 2
        seq_length = 32
        
        dummy_input = {
            'input_ids': torch.randint(0, 1000, (batch_size, seq_length)),
            'attention_mask': torch.ones(batch_size, seq_length),
            'labels': torch.randint(0, 3, (batch_size,))
        }
        
        with torch.no_grad():
            baseline_output = baseline_model(**dummy_input)
            lora_output = lora_model(**dummy_input)
        
        baseline_ok = baseline_output['loss'] is not None
        lora_ok = lora_output['loss'] is not None
        
        print_test("Baseline Forward Pass", baseline_ok, f"Loss: {baseline_output['loss']:.4f}")
        print_test("LoRA Forward Pass", lora_ok, f"Loss: {lora_output['loss']:.4f}")
        
        return baseline_ok and lora_ok
        
    except Exception as e:
        print_test("Model Initialization", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_training_loop():
    """Test a minimal training loop."""
    print_header("TRAINING LOOP CHECK")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from data.dataset import DataPreprocessor
        from models.lora_model import LoRASentimentModel
        import torch
        from torch.utils.data import DataLoader
        
        print("  Setting up minimal training test...")
        
        # Create tiny dataset
        preprocessor = DataPreprocessor("distilbert-base-uncased")
        train_ds, _, _ = preprocessor.create_datasets(
            domains=["electronics"],
            train_size=5, val_size=2, test_size=2,
            max_length=32
        )
        
        dataloader = DataLoader(train_ds, batch_size=2, shuffle=True)
        
        # Create model
        model = LoRASentimentModel(
            model_name="distilbert-base-uncased",
            num_labels=3,
            lora_config={'r': 4, 'lora_alpha': 8}
        )
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        print("  Running one training step...")
        
        model.train()
        batch = next(iter(dataloader))
        
        # Forward pass
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        
        loss = outputs['loss']
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        print_test("Training Step", True, f"Loss: {loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print_test("Training Loop", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests and provide summary."""
    
    print("\n" + "="*60)
    print("  SENTIMENT ANALYSIS PEFT - SETUP VERIFICATION")
    print("="*60)
    print("\n  Testing your environment and dependencies...")
    print("  This may take a minute...\n")
    
    # Run all tests
    tests = [
        ("Python Version", test_python_version),
        ("Package Imports", test_imports),
        ("CUDA/GPU", test_cuda),
        ("Data Loading", test_data_loading),
        ("Model Initialization", test_model_initialization),
        ("Training Loop", test_training_loop)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ Unexpected error in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print_header("SUMMARY")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for name, passed in results:
        status = "✅" if passed else "❌"
        print(f"  {status} {name}")
    
    print(f"\n  Results: {passed_count}/{total_count} tests passed")
    
    # Final verdict
    if passed_count == total_count:
        print("\n  🎉 All tests passed! You're ready to run experiments.")
        print("\n  🚀 Next steps:")
        print("     1. Run quick demo:")
        print("        python scripts/minimal_demo.py")
        print("     2. Run full experiment:")
        print("        python scripts/run_experiments.py --quick_mode")
    else:
        print("\n  ⚠️  Some tests failed. Please fix the issues above.")
        print("\n  💡 Common fixes:")
        print("     • Missing packages: pip install -r requirements.txt")
        print("     • Python version: Upgrade to Python 3.8+")
        print("     • GPU issues: Training will work on CPU (slower)")
    
    print("\n" + "="*60 + "\n")
    
    return passed_count == total_count


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)