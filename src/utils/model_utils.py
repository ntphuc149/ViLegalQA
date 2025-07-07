"""
Model utilities for model information and operations.
"""

import torch
from typing import Optional


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        trainable_only: Only count trainable parameters
        
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_model_size(model_name: str) -> Optional[float]:
    """
    Extract model size from model name.
    
    Args:
        model_name: Model name
        
    Returns:
        Model size in billions of parameters, or None if unknown
    """
    model_lower = model_name.lower()
    
    size_patterns = {
        "0.5b": 0.5,
        "1.5b": 1.5,
        "2.7b": 2.7,
        "3b": 3.0,
        "7b": 7.0,
        "13b": 13.0,
        "base": 0.3,  # Typical base model size
        "large": 0.8,  # Typical large model size
    }
    
    for pattern, size in size_patterns.items():
        if pattern in model_lower:
            return size
    
    return None