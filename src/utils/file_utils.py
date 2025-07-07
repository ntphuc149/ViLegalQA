"""
File handling utilities for ViBidLQA-AQA system.
Provides safe file operations, JSON/pickle handling, and directory management.
"""

import json
import pickle
import shutil
from pathlib import Path
from typing import Any, Union, Dict, List, Optional
import yaml
from .logging_utils import get_logger

logger = get_logger(__name__)


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path to create
        
    Returns:
        Path object of the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_file_operation(operation: str):
    """Decorator for safe file operations with error handling."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except FileNotFoundError as e:
                logger.error(f"{operation} failed - File not found: {e}")
                raise
            except PermissionError as e:
                logger.error(f"{operation} failed - Permission denied: {e}")
                raise
            except Exception as e:
                logger.error(f"{operation} failed - Unexpected error: {e}")
                raise
        return wrapper
    return decorator


@safe_file_operation("Save JSON")
def save_json(
    data: Any, 
    file_path: Union[str, Path],
    indent: int = 2,
    ensure_ascii: bool = False
) -> None:
    """
    Save data to JSON file safely.
    
    Args:
        data: Data to save
        file_path: Output file path
        indent: JSON indentation
        ensure_ascii: Whether to ensure ASCII encoding
    """
    file_path = Path(file_path)
    ensure_dir(file_path.parent)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
    
    logger.debug(f"JSON saved to: {file_path}")


@safe_file_operation("Load JSON")
def load_json(file_path: Union[str, Path]) -> Any:
    """
    Load data from JSON file safely.
    
    Args:
        file_path: Input file path
        
    Returns:
        Loaded data
    """
    file_path = Path(file_path)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.debug(f"JSON loaded from: {file_path}")
    return data


@safe_file_operation("Save YAML")
def save_yaml(
    data: Any,
    file_path: Union[str, Path],
    default_flow_style: bool = False
) -> None:
    """
    Save data to YAML file safely.
    
    Args:
        data: Data to save
        file_path: Output file path
        default_flow_style: YAML flow style
    """
    file_path = Path(file_path)
    ensure_dir(file_path.parent)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=default_flow_style, allow_unicode=True)
    
    logger.debug(f"YAML saved to: {file_path}")


@safe_file_operation("Load YAML")
def load_yaml(file_path: Union[str, Path]) -> Any:
    """
    Load data from YAML file safely.
    
    Args:
        file_path: Input file path
        
    Returns:
        Loaded data
    """
    file_path = Path(file_path)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    logger.debug(f"YAML loaded from: {file_path}")
    return data


@safe_file_operation("Save pickle")
def save_pickle(data: Any, file_path: Union[str, Path]) -> None:
    """
    Save data to pickle file safely.
    
    Args:
        data: Data to save
        file_path: Output file path
    """
    file_path = Path(file_path)
    ensure_dir(file_path.parent)
    
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    
    logger.debug(f"Pickle saved to: {file_path}")


@safe_file_operation("Load pickle")
def load_pickle(file_path: Union[str, Path]) -> Any:
    """
    Load data from pickle file safely.
    
    Args:
        file_path: Input file path
        
    Returns:
        Loaded data
    """
    file_path = Path(file_path)
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    logger.debug(f"Pickle loaded from: {file_path}")
    return data


def copy_file(src: Union[str, Path], dst: Union[str, Path]) -> None:
    """
    Copy file from source to destination safely.
    
    Args:
        src: Source file path
        dst: Destination file path
    """
    src, dst = Path(src), Path(dst)
    ensure_dir(dst.parent)
    
    shutil.copy2(src, dst)
    logger.debug(f"File copied from {src} to {dst}")


def move_file(src: Union[str, Path], dst: Union[str, Path]) -> None:
    """
    Move file from source to destination safely.
    
    Args:
        src: Source file path
        dst: Destination file path
    """
    src, dst = Path(src), Path(dst)
    ensure_dir(dst.parent)
    
    shutil.move(str(src), str(dst))
    logger.debug(f"File moved from {src} to {dst}")


def list_files(
    directory: Union[str, Path],
    pattern: str = "*",
    recursive: bool = False
) -> List[Path]:
    """
    List files in directory with pattern matching.
    
    Args:
        directory: Directory to search
        pattern: File pattern (glob style)
        recursive: Whether to search recursively
        
    Returns:
        List of matching file paths
    """
    directory = Path(directory)
    
    if recursive:
        files = list(directory.rglob(pattern))
    else:
        files = list(directory.glob(pattern))
    
    # Filter only files (not directories)
    files = [f for f in files if f.is_file()]
    
    logger.debug(f"Found {len(files)} files matching '{pattern}' in {directory}")
    return files


def get_file_size(file_path: Union[str, Path]) -> int:
    """
    Get file size in bytes.
    
    Args:
        file_path: File path
        
    Returns:
        File size in bytes
    """
    return Path(file_path).stat().st_size


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def backup_file(file_path: Union[str, Path], backup_suffix: str = ".bak") -> Path:
    """
    Create a backup of a file.
    
    Args:
        file_path: Original file path
        backup_suffix: Suffix for backup file
        
    Returns:
        Path to backup file
    """
    file_path = Path(file_path)
    backup_path = file_path.with_suffix(file_path.suffix + backup_suffix)
    
    if file_path.exists():
        copy_file(file_path, backup_path)
        logger.info(f"Backup created: {backup_path}")
    
    return backup_path


def create_experiment_dir(
    base_dir: Union[str, Path],
    experiment_name: str,
    timestamp: bool = True
) -> Path:
    """
    Create directory structure for experiments.
    
    Args:
        base_dir: Base directory for experiments
        experiment_name: Name of the experiment
        timestamp: Whether to add timestamp to directory name
        
    Returns:
        Path to experiment directory
    """
    from datetime import datetime
    
    base_dir = Path(base_dir)
    
    if timestamp:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = base_dir / f"{experiment_name}_{timestamp_str}"
    else:
        exp_dir = base_dir / experiment_name
    
    # Create subdirectories
    subdirs = ["model", "logs", "results", "configs"]
    for subdir in subdirs:
        ensure_dir(exp_dir / subdir)
    
    logger.info(f"Experiment directory created: {exp_dir}")
    return exp_dir


def save_experiment_config(
    config: Any,
    experiment_dir: Union[str, Path],
    filename: str = "config.yaml"
) -> None:
    """
    Save experiment configuration to file.
    
    Args:
        config: Configuration object or dictionary
        experiment_dir: Experiment directory
        filename: Configuration file name
    """
    experiment_dir = Path(experiment_dir)
    config_path = experiment_dir / "configs" / filename
    
    # Convert config object to dictionary if needed
    if hasattr(config, '__dict__'):
        config_dict = config.__dict__.copy()
    elif hasattr(config, 'to_dict'):
        config_dict = config.to_dict()
    else:
        config_dict = config
    
    # Convert Path objects to strings for YAML serialization
    def convert_paths(obj):
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: convert_paths(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_paths(item) for item in obj]
        return obj
    
    config_dict = convert_paths(config_dict)
    save_yaml(config_dict, config_path)
    logger.info(f"Configuration saved to: {config_path}")


def cleanup_temp_files(temp_dir: Union[str, Path], pattern: str = "tmp_*") -> None:
    """
    Clean up temporary files in directory.
    
    Args:
        temp_dir: Temporary directory
        pattern: Pattern for temp files to remove
    """
    temp_dir = Path(temp_dir)
    
    if temp_dir.exists():
        temp_files = list_files(temp_dir, pattern, recursive=True)
        
        for temp_file in temp_files:
            try:
                temp_file.unlink()
                logger.debug(f"Removed temp file: {temp_file}")
            except Exception as e:
                logger.warning(f"Could not remove temp file {temp_file}: {e}")
        
        logger.info(f"Cleaned up {len(temp_files)} temporary files")