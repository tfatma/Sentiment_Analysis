# src/utils/logging_utils.py
import logging
import os
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console_output: bool = True
) -> logging.Logger:
    """
    Setup logger with both file and console handlers.
    
    Why proper logging matters:
    - Debugging: Track what went wrong during training
    - Monitoring: See training progress in real-time
    - Reproducibility: Have a record of all experiments
    - Production: Essential for deployed systems
    
    Args:
        name: Logger name (e.g., 'trainer', 'evaluator')
        log_file: Path to log file (optional)
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        console_output: Whether to print to console
        
    Returns:
        Configured logger instance
        
    Example:
        logger = setup_logger('training', 'logs/train.log')
        logger.info('Starting epoch 1')
        logger.warning('GPU memory at 90%')
        logger.error('Training failed!')
        
    Interview point:
    "I implemented comprehensive logging to track experiments, 
    debug issues, and maintain reproducibility. All experiments 
    are logged to both console and file for easy review."
    """
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicates
    # This is important when logger is called multiple times
    logger.handlers.clear()
    
    # Create formatter with timestamp, level, and message
    # Format: "2024-01-15 10:30:45 - trainer - INFO - Starting training..."
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler (prints to terminal)
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler (saves to file)
    if log_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to parent loggers
    # This avoids duplicate log messages
    logger.propagate = False
    
    return logger


def get_experiment_logger(
    experiment_name: str, 
    output_dir: str = './results/logs'
) -> logging.Logger:
    """
    Get a logger specifically for an experiment with timestamped filename.
    
    Why timestamped logs?
    - Prevents overwriting previous experiment logs
    - Easy to find logs for specific run
    - Automatic organization
    
    Args:
        experiment_name: Name of the experiment
        output_dir: Directory to save logs
        
    Returns:
        Configured logger
        
    Example:
        logger = get_experiment_logger('lora_r16_experiment')
        # Creates: results/logs/lora_r16_experiment_20240115_103045.log
        
    File naming convention:
        {experiment_name}_{YYYYMMDD_HHMMSS}.log
        
    This helps with:
    - Tracking multiple runs of same experiment
    - Comparing hyperparameter changes over time
    - Debugging specific failed runs
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"{experiment_name}_{timestamp}.log")
    
    return setup_logger(
        f"experiment_{experiment_name}",
        log_file=log_file,
        level=logging.INFO
    )


def log_model_info(logger: logging.Logger, model, model_name: str = "Model"):
    """
    Log detailed model information.
    
    What it logs:
    - Model architecture name
    - Total parameters
    - Trainable parameters  
    - Frozen parameters
    - Trainable percentage
    - Model size in MB
    
    Args:
        logger: Logger instance
        model: PyTorch model
        model_name: Descriptive name for the model
        
    Example output:
        Model Information: LoRA Model
        ├── Total Parameters: 125,234,432 (125.2M)
        ├── Trainable Parameters: 589,824 (0.59M)
        ├── Frozen Parameters: 124,644,608 (124.6M)
        ├── Trainable %: 0.47%
        └── Model Size: 477.8 MB
        
    Why this is valuable:
    - Quick sanity check (LoRA should be <1% trainable)
    - Memory estimation
    - Deployment planning
    - Documentation
    """
    from training.utils import count_parameters, calculate_model_size
    
    params = count_parameters(model)
    size = calculate_model_size(model)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"{model_name} Information")
    logger.info(f"{'='*60}")
    logger.info(f"Total Parameters:      {params['total']:,} ({params['total']/1e6:.1f}M)")
    logger.info(f"Trainable Parameters:  {params['trainable']:,} ({params['trainable']/1e6:.2f}M)")
    logger.info(f"Frozen Parameters:     {params['frozen']:,} ({params['frozen']/1e6:.1f}M)")
    logger.info(f"Trainable Percentage:  {100.0 * params['trainable'] / params['total']:.2f}%")
    logger.info(f"Model Size:            {size:.1f} MB")
    logger.info(f"{'='*60}\n")


def log_training_config(logger: logging.Logger, config: dict):
    """
    Log training configuration in a readable format.
    
    Why log config?
    - Reproducibility: Know exact settings used
    - Debugging: Identify misconfigured hyperparameters
    - Documentation: Auto-generates experiment documentation
    
    Args:
        logger: Logger instance
        config: Configuration dictionary
        
    Example output:
        Training Configuration:
        ├── Learning Rate: 2e-05
        ├── Batch Size: 16
        ├── Epochs: 3
        ├── Optimizer: AdamW
        ├── Weight Decay: 0.01
        └── Warmup Steps: 500
    """
    logger.info(f"\n{'='*60}")
    logger.info("Training Configuration")
    logger.info(f"{'='*60}")
    
    def log_dict(d, indent=0):
        """Recursively log nested dictionaries."""
        for key, value in d.items():
            if isinstance(value, dict):
                logger.info(f"{'  ' * indent}{key}:")
                log_dict(value, indent + 1)
            else:
                logger.info(f"{'  ' * indent}{key}: {value}")
    
    log_dict(config)
    logger.info(f"{'='*60}\n")


def log_metrics(logger: logging.Logger, metrics: dict, prefix: str = ""):
    """
    Log metrics in a clean, readable format.
    
    Perfect for logging:
    - Training metrics (loss, accuracy, F1)
    - Validation metrics
    - Test results
    
    Args:
        logger: Logger instance
        metrics: Dictionary of metric names and values
        prefix: Optional prefix (e.g., "Epoch 1", "Final Results")
        
    Example:
        metrics = {
            'train/loss': 0.324,
            'train/accuracy': 0.876,
            'val/loss': 0.412,
            'val/f1': 0.854
        }
        log_metrics(logger, metrics, "Epoch 1")
        
    Output:
        Epoch 1 Metrics:
        ├── train/loss: 0.3240
        ├── train/accuracy: 0.8760
        ├── val/loss: 0.4120
        └── val/f1: 0.8540
    """
    header = f"{prefix} Metrics" if prefix else "Metrics"
    logger.info(f"\n{header}:")
    
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
        else:
            logger.info(f"  {key}: {value}")


class TensorBoardLogger:
    """
    Optional TensorBoard integration for advanced visualization.
    
    TensorBoard provides:
    - Interactive training curves
    - Hyperparameter comparison
    - Model graph visualization
    - Embedding projections
    
    Note: This is optional and requires TensorBoard installation
    Usage:
        tb_logger = TensorBoardLogger('runs/experiment1')
        tb_logger.log_scalar('loss', loss, step)
        tb_logger.log_scalars('metrics', {'acc': 0.9, 'f1': 0.88}, step)
    """
    
    def __init__(self, log_dir: str):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory for TensorBoard logs
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
            self.enabled = True
        except ImportError:
            print("TensorBoard not available. Install with: pip install tensorboard")
            self.enabled = False
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value."""
        if self.enabled:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int):
        """Log multiple scalars."""
        if self.enabled:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def close(self):
        """Close the TensorBoard writer."""
        if self.enabled:
            self.writer.close()


# Example usage in scripts:
"""
# Basic logging
logger = setup_logger('training', 'logs/train.log')
logger.info('Starting training...')

# Experiment logging
logger = get_experiment_logger('lora_experiment')
logger.info('Experiment initialized')

# Model info logging
log_model_info(logger, model, 'LoRA Model')

# Config logging
log_training_config(logger, config)

# Metrics logging
log_metrics(logger, metrics, 'Epoch 1')
"""