"""
Training-specific configuration classes for ViBidLQA-AQA
"""

from .finetune_config import FinetuneConfig
from .instruct_config import InstructConfig

__all__ = ["FinetuneConfig", "InstructConfig"]