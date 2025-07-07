"""
Model utilities for ViBidLQA-AQA system.
Provides model inspection, memory management, and utility functions.
"""

import torch
import gc
import random
import numpy as np
from typing import Tuple, Optional, Any, Dict
from .logging_utils import get_logger

logger = get_logger(__name__)


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducible results.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Additional settings for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to: {seed}")


def get_model_size(model: torch.nn.Module) -> Tuple[int, int]:
    """
    Get model parameter counts.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def format_parameter_count(param_count: int) -> str:
    """
    Format parameter count in human readable format.
    
    Args:
        param_count: Number of parameters
        
    Returns:
        Formatted parameter count string
    """
    if param_count >= 1e9:
        return f"{param_count/1e9:.2f}B"
    elif param_count >= 1e6:
        return f"{param_count/1e6:.2f}M"
    elif param_count >= 1e3:
        return f"{param_count/1e3:.2f}K"
    else:
        return str(param_count)


def estimate_memory_usage(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    dtype: torch.dtype = torch.float32
) -> Dict[str, float]:
    """
    Estimate model memory usage.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (batch_size, ...)
        dtype: Data type for estimation
        
    Returns:
        Dictionary with memory estimates in MB
    """
    # Parameter memory
    total_params, _ = get_model_size(model)
    
    # Bytes per parameter (float32 = 4 bytes, float16 = 2 bytes)
    bytes_per_param = torch.tensor(0, dtype=dtype).element_size()
    param_memory_mb = (total_params * bytes_per_param) / (1024 ** 2)
    
    # Activation memory (rough estimate)
    batch_size = input_shape[0] if input_shape else 1
    activation_memory_mb = (batch_size * 1024 * 1024 * bytes_per_param) / (1024 ** 2)  # Rough estimate
    
    # Gradient memory (same as parameters for training)
    gradient_memory_mb = param_memory_mb
    
    # Total training memory
    total_training_mb = param_memory_mb + activation_memory_mb + gradient_memory_mb
    
    return {
        "parameters_mb": param_memory_mb,
        "activations_mb": activation_memory_mb,
        "gradients_mb": gradient_memory_mb,
        "total_training_mb": total_training_mb,
        "inference_mb": param_memory_mb + activation_memory_mb
    }


def cleanup_memory() -> None:
    """
    Clean up GPU and CPU memory.
    """
    # Clear Python garbage collection
    gc.collect()
    
    # Clear PyTorch cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    logger.debug("Memory cleanup completed")


def get_device_info() -> Dict[str, Any]:
    """
    Get information about available devices.
    
    Returns:
        Dictionary with device information
    """
    device_info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": None,
        "device_names": [],
        "total_memory": [],
        "allocated_memory": [],
        "cached_memory": []
    }
    
    if torch.cuda.is_available():
        device_info["current_device"] = torch.cuda.current_device()
        
        for i in range(torch.cuda.device_count()):
            device_info["device_names"].append(torch.cuda.get_device_name(i))
            device_info["total_memory"].append(torch.cuda.get_device_properties(i).total_memory)
            device_info["allocated_memory"].append(torch.cuda.memory_allocated(i))
            device_info["cached_memory"].append(torch.cuda.memory_reserved(i))
    
    return device_info


def log_device_info() -> None:
    """Log device information."""
    device_info = get_device_info()
    
    logger.info("=" * 50)
    logger.info("DEVICE INFORMATION")
    logger.info("=" * 50)
    logger.info(f"CUDA Available: {device_info['cuda_available']}")
    
    if device_info['cuda_available']:
        logger.info(f"Device Count: {device_info['cuda_device_count']}")
        logger.info(f"Current Device: {device_info['current_device']}")
        
        for i, name in enumerate(device_info['device_names']):
            total_mb = device_info['total_memory'][i] / (1024 ** 2)
            allocated_mb = device_info['allocated_memory'][i] / (1024 ** 2)
            cached_mb = device_info['cached_memory'][i] / (1024 ** 2)
            
            logger.info(f"Device {i}: {name}")
            logger.info(f"  Total Memory: {total_mb:.1f} MB")
            logger.info(f"  Allocated: {allocated_mb:.1f} MB")
            logger.info(f"  Cached: {cached_mb:.1f} MB")
    else:
        logger.info("Using CPU")
    
    logger.info("=" * 50)


def get_optimal_device() -> torch.device:
    """
    Get the optimal device for training/inference.
    
    Returns:
        Optimal torch device
    """
    if torch.cuda.is_available():
        # Use the device with most free memory
        max_free_memory = 0
        best_device = 0
        
        for i in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(i).total_memory
            allocated = torch.cuda.memory_allocated(i)
            free = total - allocated
            
            if free > max_free_memory:
                max_free_memory = free
                best_device = i
        
        device = torch.device(f"cuda:{best_device}")
        logger.info(f"Selected device: {device} (Free memory: {max_free_memory / (1024**2):.1f} MB)")
    else:
        device = torch.device("cpu")
        logger.info("Selected device: CPU")
    
    return device


def move_to_device(obj: Any, device: torch.device) -> Any:
    """
    Move object to device safely.
    
    Args:
        obj: Object to move (tensor, model, etc.)
        device: Target device
        
    Returns:
        Object moved to device
    """
    try:
        if hasattr(obj, 'to'):
            return obj.to(device)
        elif isinstance(obj, dict):
            return {k: move_to_device(v, device) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [move_to_device(item, device) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(move_to_device(item, device) for item in obj)
        else:
            return obj
    except Exception as e:
        logger.warning(f"Could not move object to device {device}: {e}")
        return obj


def safe_load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Optional[str] = None
) -> Dict[str, Any]:
    """
    Safely load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state
        map_location: Device mapping for loading
        
    Returns:
        Checkpoint data dictionary
    """
    try:
        if map_location is None:
            map_location = 'cpu' if not torch.cuda.is_available() else None
        
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        # Load model state
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Assume the checkpoint is just the state dict
            model.load_state_dict(checkpoint)
        
        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Checkpoint loaded successfully from: {checkpoint_path}")
        return checkpoint
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint from {checkpoint_path}: {e}")
        raise


def save_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    loss: Optional[float] = None,
    metrics: Optional[Dict[str, float]] = None,
    config: Optional[Any] = None
) -> None:
    """
    Save model checkpoint safely.
    
    Args:
        checkpoint_path: Path to save checkpoint
        model: Model to save
        optimizer: Optional optimizer state
        epoch: Current epoch
        loss: Current loss value
        metrics: Current metrics
        config: Configuration object
    """
    try:
        from .file_utils import ensure_dir
        from pathlib import Path
        
        checkpoint_path = Path(checkpoint_path)
        ensure_dir(checkpoint_path.parent)
        
        checkpoint_data = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'loss': loss,
            'metrics': metrics or {}
        }
        
        if optimizer is not None:
            checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
        
        if config is not None:
            checkpoint_data['config'] = config.__dict__ if hasattr(config, '__dict__') else config
        
        torch.save(checkpoint_data, checkpoint_path)
        logger.info(f"Checkpoint saved to: {checkpoint_path}")
        
    except Exception as e:
        logger.error(f"Failed to save checkpoint to {checkpoint_path}: {e}")
        raise


def freeze_model_layers(
    model: torch.nn.Module,
    freeze_patterns: Optional[list] = None,
    unfreeze_patterns: Optional[list] = None
) -> int:
    """
    Freeze/unfreeze model layers based on name patterns.
    
    Args:
        model: Model to modify
        freeze_patterns: List of patterns for layers to freeze
        unfreeze_patterns: List of patterns for layers to unfreeze
        
    Returns:
        Number of frozen parameters
    """
    frozen_params = 0
    
    for name, param in model.named_parameters():
        should_freeze = False
        
        # Check freeze patterns
        if freeze_patterns:
            should_freeze = any(pattern in name for pattern in freeze_patterns)
        
        # Check unfreeze patterns (overrides freeze)
        if unfreeze_patterns:
            should_unfreeze = any(pattern in name for pattern in unfreeze_patterns)
            if should_unfreeze:
                should_freeze = False
        
        if should_freeze:
            param.requires_grad = False
            frozen_params += param.numel()
            logger.debug(f"Frozen layer: {name}")
        else:
            param.requires_grad = True
    
    total_params, trainable_params = get_model_size(model)
    logger.info(f"Frozen {frozen_params:,} parameters")
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    return frozen_params


def count_model_flops(model: torch.nn.Module, input_shape: Tuple[int, ...]) -> int:
    """
    Estimate model FLOPs (rough approximation).
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        
    Returns:
        Estimated FLOPs
    """
    try:
        # This is a very rough estimation
        total_params, _ = get_model_size(model)
        
        # Rough FLOP estimation: 2 * params * sequence_length
        sequence_length = input_shape[1] if len(input_shape) > 1 else 1
        estimated_flops = 2 * total_params * sequence_length
        
        return estimated_flops
        
    except Exception as e:
        logger.warning(f"Could not estimate FLOPs: {e}")
        return 0


def print_model_summary(
    model: torch.nn.Module,
    input_shape: Optional[Tuple[int, ...]] = None,
    show_layers: bool = False
) -> None:
    """
    Print comprehensive model summary.
    
    Args:
        model: PyTorch model
        input_shape: Optional input shape for memory estimation
        show_layers: Whether to show individual layers
    """
    total_params, trainable_params = get_model_size(model)
    
    logger.info("=" * 60)
    logger.info("MODEL SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Total parameters: {format_parameter_count(total_params)} ({total_params:,})")
    logger.info(f"Trainable parameters: {format_parameter_count(trainable_params)} ({trainable_params:,})")
    logger.info(f"Non-trainable parameters: {format_parameter_count(total_params - trainable_params)} ({total_params - trainable_params:,})")
    
    if input_shape:
        memory_info = estimate_memory_usage(model, input_shape)
        logger.info(f"Estimated memory (inference): {memory_info['inference_mb']:.1f} MB")
        logger.info(f"Estimated memory (training): {memory_info['total_training_mb']:.1f} MB")
        
        flops = count_model_flops(model, input_shape)
        if flops > 0:
            logger.info(f"Estimated FLOPs: {format_parameter_count(flops)}")
    
    if show_layers:
        logger.info("\nLayer Details:")
        logger.info("-" * 60)
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                params = sum(p.numel() for p in module.parameters())
                if params > 0:
                    logger.info(f"{name}: {module.__class__.__name__} ({format_parameter_count(params)})")
    
    logger.info("=" * 60)