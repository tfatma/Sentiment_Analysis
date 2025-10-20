# src/training/utils.py
import torch
import numpy as np
import random
from typing import Dict, Any
import yaml
import os


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    
    Why this matters:
    - Ensures reproducible results across runs
    - Critical for comparing different models fairly
    - Required for scientific validity of experiments
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # These make training deterministic but slower
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Dictionary containing configuration
        
    Example:
        config = load_config('config/model_config.yaml')
        model_name = config['model']['name']
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path where to save the YAML file
        
    Usage:
        Saves experiment configurations for reproducibility
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, indent=2)


def get_device() -> torch.device:
    """
    Get the best available device for training.
    
    Returns:
        torch.device object (cuda, mps, or cpu)
        
    Priority:
        1. CUDA (NVIDIA GPUs) - fastest for training
        2. MPS (Apple Silicon) - good for M1/M2 Macs
        3. CPU - slowest but universally available
        
    Example:
        device = get_device()
        model.to(device)
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters (total, trainable, frozen).
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
        
    This is crucial for:
    - Understanding model size
    - Comparing efficiency (especially for LoRA)
    - Memory estimation
    
    Example:
        params = count_parameters(model)
        print(f"Trainable: {params['trainable']:,}")
        # Output: Trainable: 589,824
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params
    }


def calculate_model_size(model: torch.nn.Module) -> float:
    """
    Calculate model size in megabytes.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in MB
        
    Breakdown:
    - param_size: weights and biases
    - buffer_size: batch norm stats, etc.
    
    Why this matters:
    - Deployment constraints (mobile, edge devices)
    - Storage and transfer costs
    - Memory requirements during inference
    
    Example:
        size = calculate_model_size(lora_model)
        print(f"Model size: {size:.1f}MB")
        # LoRA adapters: ~2.4MB vs Full model: ~500MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted string (e.g., "1h 23m 45s")
        
    Example:
        time_str = format_time(5025.5)
        print(time_str)  # "1h 23m 45s"
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def log_gpu_memory():
    """
    Log current GPU memory usage.
    
    Useful for:
    - Debugging out-of-memory errors
    - Optimizing batch sizes
    - Comparing memory efficiency of different models
    
    Example output:
        GPU Memory: 4.2GB / 16.0GB (26.3% used)
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, "
              f"Reserved: {reserved:.2f}GB, "
              f"Total: {total:.2f}GB")
    else:
        print("CUDA not available")