"""
File utilities for common file operations.
"""

import json
from pathlib import Path
from typing import Any, Dict


def ensure_dir(path: str) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def save_json(data: Dict[str, Any], path: str) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        path: File path
    """
    ensure_dir(Path(path).parent)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> Dict[str, Any]:
    """
    Load data from JSON file.
    
    Args:
        path: File path
        
    Returns:
        Loaded data
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)