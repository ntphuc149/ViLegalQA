"""
Configuration factory for creating appropriate config instances
based on model type and training method.
"""

import argparse
import yaml
from pathlib import Path
from typing import Union, Dict, Any

from .base_config import BaseConfig
from .model_configs.plm_config import PLMConfig
from .model_configs.llm_config import LLMConfig
from .training_configs.finetune_config import FinetuneConfig
from .training_configs.instruct_config import InstructConfig


class ConfigFactory:
    """Factory class for creating appropriate configuration instances."""
    
    @staticmethod
    def create_config(
        model_type: str,
        training_method: str,
        config_path: str = None,
        **kwargs
    ) -> Union[PLMConfig, LLMConfig, FinetuneConfig, InstructConfig]:
        """
        Create appropriate configuration based on model type and training method.
        
        Args:
            model_type: 'plm' or 'llm'
            training_method: 'finetune' or 'instruct'
            config_path: Path to YAML config file (optional)
            **kwargs: Additional arguments to override config values
            
        Returns:
            Appropriate configuration instance
        """
        # Load base config from file if provided
        base_config = {}
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                base_config = yaml.safe_load(f)
        
        # Merge with kwargs
        base_config.update(kwargs)
        
        # Ensure model_type and training_method are set
        base_config['model_type'] = model_type
        base_config['training_method'] = training_method
        
        # Create appropriate config instance
        if training_method == "finetune":
            if model_type != "plm":
                raise ValueError("Fine-tuning is only supported for PLM models")
            return FinetuneConfig(**base_config)
        
        elif training_method == "instruct":
            return InstructConfig(**base_config)
        
        else:
            raise ValueError(f"Unsupported training method: {training_method}")
    
    @staticmethod
    def from_args(args: argparse.Namespace) -> Union[PLMConfig, LLMConfig, FinetuneConfig, InstructConfig]:
        """Create config from command line arguments."""
        # Convert args to dict, excluding None values
        args_dict = {k: v for k, v in vars(args).items() if v is not None}
        
        # Extract model_type and training_method
        model_type = args_dict.get('model_type', 'plm')
        training_method = args_dict.get('training_method', 'finetune')
        config_path = args_dict.get('config', None)
        
        return ConfigFactory.create_config(
            model_type=model_type,
            training_method=training_method,
            config_path=config_path,
            **args_dict
        )
    
    @staticmethod
    def get_default_config(model_name: str) -> Union[PLMConfig, LLMConfig, FinetuneConfig, InstructConfig]:
        """
        Get default configuration based on model name.
        
        Args:
            model_name: Model name (e.g., 'VietAI/vit5-base', 'Qwen/Qwen2-7B')
            
        Returns:
            Appropriate default configuration
        """
        model_name_lower = model_name.lower()
        
        # Determine model type from model name
        if any(name in model_name_lower for name in ['vit5', 'bartpho']):
            model_type = 'plm'
            # PLMs can use either fine-tuning or instruction tuning
            if 'instruct' in model_name_lower or 'chat' in model_name_lower:
                training_method = 'instruct'
            else:
                training_method = 'finetune'
        
        elif any(name in model_name_lower for name in ['qwen', 'seallm', 'vinallama', 'llama']):
            model_type = 'llm'
            training_method = 'instruct'  # LLMs typically use instruction tuning
        
        else:
            # Default to PLM with fine-tuning for unknown models
            model_type = 'plm'
            training_method = 'finetune'
            print(f"Warning: Unknown model {model_name}, defaulting to PLM fine-tuning")
        
        return ConfigFactory.create_config(
            model_type=model_type,
            training_method=training_method,
            model_name=model_name
        )
    
    @staticmethod
    def save_config(config: BaseConfig, path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = config.to_dict()
        
        # Convert any Path objects to strings
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    
    @staticmethod
    def load_config(path: str) -> Union[PLMConfig, LLMConfig, FinetuneConfig, InstructConfig]:
        """Load configuration from YAML file."""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        model_type = config_dict.get('model_type', 'plm')
        training_method = config_dict.get('training_method', 'finetune')
        
        return ConfigFactory.create_config(
            model_type=model_type,
            training_method=training_method,
            **config_dict
        )


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add configuration arguments to argument parser."""
    
    # ===== BASIC CONFIGURATION =====
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")
    parser.add_argument("--model_name", type=str, default="VietAI/vit5-base", 
                       help="Model name or path")
    parser.add_argument("--model_type", type=str, choices=["plm", "llm"], default="plm",
                       help="Model type: plm (encoder-decoder) or llm (decoder-only)")
    parser.add_argument("--training_method", type=str, choices=["finetune", "instruct"], default="finetune",
                       help="Training method: finetune or instruct")
    
    # ===== STAGE CONTROL =====
    parser.add_argument("--do_finetune", action="store_true", help="Run fine-tuning stage")
    parser.add_argument("--do_infer", action="store_true", help="Run inference stage")
    parser.add_argument("--do_eval", action="store_true", help="Run evaluation stage")
    parser.add_argument("--do_end2end", action="store_true", help="Run all stages end-to-end")
    
    # ===== DATASET CONFIGURATION =====
    parser.add_argument("--dataset_name", type=str, default="Truong-Phuc/ViBidLQA",
                       help="Dataset name or path")
    parser.add_argument("--data_split_mode", type=str, choices=["auto", "predefined"], default="auto",
                       help="Data splitting mode")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Training set ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation set ratio")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test set ratio")
    
    # ===== TRAINING PARAMETERS =====
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Training batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2, help="Evaluation batch size per device")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="Gradient accumulation steps")
    
    # ===== SEQUENCE LENGTH CONFIGURATION =====
    parser.add_argument("--max_source_length", type=int, default=1024, help="Maximum source sequence length")
    parser.add_argument("--max_target_length", type=int, default=256, help="Maximum target sequence length")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length (for LLMs)")
    
    # ===== LORA CONFIGURATION (for LLMs) =====
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    
    # ===== EVALUATION CONFIGURATION =====
    parser.add_argument("--checkpoint_path", type=str, help="Path to checkpoint for inference")
    parser.add_argument("--results_file", type=str, help="Path to results CSV file for evaluation")
    
    # ===== GENERATION CONFIGURATION =====
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum new tokens for generation")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.75, help="Top-p for generation")
    
    # ===== LOGGING =====
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="vibidlqa-aqa", help="W&B project name")
    
    return parser