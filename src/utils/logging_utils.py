"""
Logging utilities for ViBidLQA-AQA system.
Provides centralized logging configuration and utilities.
"""

import logging
import sys
import os
from pathlib import Path
from typing import Optional, Union
from datetime import datetime


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    format_string: Optional[str] = None,
    include_timestamp: bool = True
) -> None:
    """
    Setup centralized logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs
        format_string: Custom format string for log messages
        include_timestamp: Whether to include timestamp in logs
    """
    # Set logging level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatter
    if format_string is None:
        if include_timestamp:
            format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        else:
            format_string = '%(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # Remove existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(numeric_level)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(numeric_level)
        root_logger.addHandler(file_handler)
    
    # Set root logger level
    root_logger.setLevel(numeric_level)
    
    # Suppress some noisy loggers
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("wandb").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # If no handlers are configured, setup basic logging
    if not logging.getLogger().handlers:
        setup_logging()
    
    return logger


class LoggerContextManager:
    """Context manager for temporary logging level changes."""
    
    def __init__(self, logger: logging.Logger, level: str):
        self.logger = logger
        self.new_level = getattr(logging, level.upper())
        self.old_level = None
    
    def __enter__(self):
        self.old_level = self.logger.level
        self.logger.setLevel(self.new_level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.old_level)


def log_execution_time(func):
    """Decorator to log function execution time."""
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} completed in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.2f} seconds: {e}")
            raise
    
    return wrapper


def create_experiment_logger(
    experiment_name: str,
    output_dir: Union[str, Path],
    log_level: str = "INFO"
) -> logging.Logger:
    """
    Create a logger for a specific experiment.
    
    Args:
        experiment_name: Name of the experiment
        output_dir: Directory to save logs
        log_level: Logging level
        
    Returns:
        Configured experiment logger
    """
    # Create experiment-specific log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(output_dir) / "logs" / f"{experiment_name}_{timestamp}.log"
    
    # Setup logging with file output
    setup_logging(
        log_level=log_level,
        log_file=log_file,
        include_timestamp=True
    )
    
    # Get experiment logger
    logger = get_logger(f"experiment.{experiment_name}")
    logger.info(f"Experiment '{experiment_name}' started")
    logger.info(f"Logs will be saved to: {log_file}")
    
    return logger


# Convenience functions for common logging patterns
def log_config(config: object, logger: Optional[logging.Logger] = None) -> None:
    """Log configuration object in a readable format."""
    if logger is None:
        logger = get_logger(__name__)
    
    logger.info("=" * 50)
    logger.info("CONFIGURATION")
    logger.info("=" * 50)
    
    if hasattr(config, '__dict__'):
        for key, value in config.__dict__.items():
            logger.info(f"{key}: {value}")
    else:
        logger.info(str(config))
    
    logger.info("=" * 50)


def log_model_info(model, logger: Optional[logging.Logger] = None) -> None:
    """Log model information including parameters count."""
    if logger is None:
        logger = get_logger(__name__)
    
    try:
        from .model_utils import get_model_size
        
        total_params, trainable_params = get_model_size(model)
        
        logger.info("=" * 50)
        logger.info("MODEL INFORMATION")
        logger.info("=" * 50)
        logger.info(f"Model class: {model.__class__.__name__}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Trainable ratio: {trainable_params/total_params*100:.2f}%")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.warning(f"Could not log model info: {e}")


def log_dataset_info(dataset, logger: Optional[logging.Logger] = None) -> None:
    """Log dataset information."""
    if logger is None:
        logger = get_logger(__name__)
    
    logger.info("=" * 50)
    logger.info("DATASET INFORMATION")
    logger.info("=" * 50)
    
    if hasattr(dataset, 'keys'):
        for split_name in dataset.keys():
            split_size = len(dataset[split_name])
            logger.info(f"{split_name}: {split_size:,} samples")
    else:
        logger.info(f"Dataset size: {len(dataset):,} samples")
    
    # Log sample structure
    if hasattr(dataset, 'column_names'):
        logger.info(f"Columns: {dataset.column_names}")
    elif hasattr(dataset, 'features'):
        logger.info(f"Features: {list(dataset.features.keys())}")
    
    logger.info("=" * 50)