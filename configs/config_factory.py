"""
Configuration factory for ViBidLQA-AQA system.
Dynamically creates configurations based on model type and training method.
"""

import argparse
import yaml
from pathlib import Path
from typing import Union, Dict, Any, Optional

from base_config import BaseConfig, load_config_from_env
from .model_configs.plm_config import PLMConfig, get_plm_config
from .model_configs.llm_config import LLMConfig, get_llm_config
from .training_configs.finetune_config import FinetuneConfig
from .training_configs.instruct_config import InstructConfig


class ConfigFactory:
    """
    Factory class for creating configurations dynamically.
    """
    
    @staticmethod
    def create_config(
        model_type: str,
        training_method: str,
        model_name: Optional[str] = None,
        **kwargs
    ) -> Union[PLMConfig, LLMConfig]:
        """
        Create configuration based on model type and training method.
        
        Args:
            model_type: Type of model ("plm" or "llm")
            training_method: Training method ("finetune" or "instruct") 
            model_name: Optional model name for specific configs
            **kwargs: Additional configuration parameters
            
        Returns:
            Appropriate configuration object
            
        Raises:
            ValueError: If invalid model_type or training_method
        """
        # Load environment variables first
        env_config = load_config_from_env()
        
        # Merge with provided kwargs (kwargs take precedence)
        config_kwargs = {**env_config, **kwargs}
        
        # Add model_name if provided
        if model_name:
            config_kwargs["model_name"] = model_name
        
        # Set training method
        config_kwargs["training_method"] = training_method
        
        if model_type == "plm":
            # Use specific PLM config if model_name is provided
            if model_name:
                config = get_plm_config(model_name, training_method)
                # Update with any additional kwargs
                if config_kwargs:
                    config.update_from_dict(config_kwargs)
                return config
            else:
                return PLMConfig(training_method=training_method, **config_kwargs)
                
        elif model_type == "llm":
            # Use specific LLM config if model_name is provided
            if model_name:
                config = get_llm_config(model_name)
                # Update with any additional kwargs
                if config_kwargs:
                    config.update_from_dict(config_kwargs)
                return config
            else:
                return LLMConfig(**config_kwargs)
        else:
            raise ValueError(
                f"Invalid model_type: {model_type}. Must be 'plm' or 'llm'"
            )
    
    @staticmethod
    def create_config_from_model_name(
        model_name: str,
        training_method: str = "auto",
        **kwargs
    ) -> Union[PLMConfig, LLMConfig]:
        """
        Create configuration automatically from model name.
        
        Args:
            model_name: Model name to infer configuration from
            training_method: Training method or "auto" to infer
            **kwargs: Additional configuration parameters
            
        Returns:
            Appropriate configuration object
        """
        # Auto-detect model type from model name
        model_type = ConfigFactory._infer_model_type(model_name)
        
        # Auto-detect training method if not specified
        if training_method == "auto":
            training_method = ConfigFactory._infer_training_method(model_name, model_type)
        
        return ConfigFactory.create_config(
            model_type=model_type,
            training_method=training_method,
            model_name=model_name,
            **kwargs
        )
    
    @staticmethod
    def _infer_model_type(model_name: str) -> str:
        """
        Infer model type from model name.
        
        Args:
            model_name: Model name
            
        Returns:
            Model type ("plm" or "llm")
        """
        model_name_lower = model_name.lower()
        
        # PLM (encoder-decoder) models
        plm_indicators = ["vit5", "bartpho", "bart", "t5", "pegasus"]
        if any(indicator in model_name_lower for indicator in plm_indicators):
            return "plm"
        
        # LLM (decoder-only) models  
        llm_indicators = ["qwen", "llama", "mistral", "sea", "vina", "gpt", "claude"]
        if any(indicator in model_name_lower for indicator in llm_indicators):
            return "llm"
        
        # Default fallback based on common patterns
        if "instruct" in model_name_lower or "chat" in model_name_lower:
            return "llm"
        
        # Default to PLM for Vietnamese models
        return "plm"
    
    @staticmethod
    def _infer_training_method(model_name: str, model_type: str) -> str:
        """
        Infer training method from model name and type.
        
        Args:
            model_name: Model name
            model_type: Model type
            
        Returns:
            Training method ("finetune" or "instruct")
        """
        model_name_lower = model_name.lower()
        
        # LLMs typically use instruction tuning
        if model_type == "llm":
            return "instruct"
        
        # Check for instruction indicators in PLM names
        instruct_indicators = ["instruct", "chat", "tuned"]
        if any(indicator in model_name_lower for indicator in instruct_indicators):
            return "instruct"
        
        # Default to fine-tuning for PLMs
        return "finetune"
    
    @staticmethod
    def load_config_from_file(config_path: str) -> Union[PLMConfig, LLMConfig]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Loaded configuration object
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        # Extract model type and training method
        model_type = config_dict.get('model_type', 'plm')
        training_method = config_dict.get('training_method', 'finetune')
        model_name = config_dict.get('model_name')
        
        return ConfigFactory.create_config(
            model_type=model_type,
            training_method=training_method,
            model_name=model_name,
            **config_dict
        )
    
    @staticmethod
    def save_config_to_file(
        config: Union[PLMConfig, LLMConfig],
        config_path: str
    ) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            config: Configuration object to save
            config_path: Output file path
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert config to dictionary
        config_dict = config.to_dict()
        
        # Convert any Path objects to strings
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    
    @staticmethod
    def create_config_from_args(args: argparse.Namespace) -> Union[PLMConfig, LLMConfig]:
        """
        Create configuration from command line arguments.
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            Configuration object
        """
        # Load from file if specified
        if hasattr(args, 'config') and args.config:
            config = ConfigFactory.load_config_from_file(args.config)
            
            # Override with command line arguments
            cli_overrides = ConfigFactory._extract_cli_overrides(args)
            if cli_overrides:
                config.update_from_dict(cli_overrides)
            
            return config
        
        # Create from CLI arguments
        model_type = getattr(args, 'model_type', 'plm')
        training_method = getattr(args, 'training_method', 'finetune')
        model_name = getattr(args, 'model_name', None)
        
        # Extract all CLI arguments
        cli_config = ConfigFactory._extract_cli_overrides(args)
        
        return ConfigFactory.create_config(
            model_type=model_type,
            training_method=training_method,
            model_name=model_name,
            **cli_config
        )
    
    @staticmethod
    def _extract_cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
        """
        Extract configuration overrides from CLI arguments.
        
        Args:
            args: Parsed arguments
            
        Returns:
            Dictionary of configuration overrides
        """
        # Define mapping from CLI args to config fields
        cli_mappings = {
            'model_name': 'model_name',
            'model_type': 'model_type', 
            'training_method': 'training_method',
            'dataset_name': 'dataset_name',
            'output_dir': 'output_dir',
            'num_train_epochs': 'num_train_epochs',
            'per_device_train_batch_size': 'per_device_train_batch_size',
            'per_device_eval_batch_size': 'per_device_eval_batch_size',
            'gradient_accumulation_steps': 'gradient_accumulation_steps',
            'learning_rate': 'learning_rate',
            'warmup_ratio': 'warmup_ratio',
            'weight_decay': 'weight_decay',
            'max_source_length': 'max_source_length',
            'max_target_length': 'max_target_length',
            'max_seq_length': 'max_seq_length',
            'max_new_tokens': 'max_new_tokens',
            'eval_steps': 'eval_steps',
            'save_steps': 'save_steps',
            'logging_steps': 'logging_steps',
            'seed': 'seed',
            'fp16': 'fp16',
            'bf16': 'bf16',
            'use_wandb': 'use_wandb',
            'wandb_project': 'wandb_project',
            'wandb_entity': 'wandb_entity',
            'lora_r': 'lora_r',
            'lora_alpha': 'lora_alpha',
            'lora_dropout': 'lora_dropout',
            'use_qlora': 'use_qlora',
            'load_in_4bit': 'load_in_4bit',
            'data_split_mode': 'data_split_mode',
            'train_ratio': 'train_ratio',
            'val_ratio': 'val_ratio',
            'test_ratio': 'test_ratio',
        }
        
        overrides = {}
        for cli_arg, config_field in cli_mappings.items():
            if hasattr(args, cli_arg):
                value = getattr(args, cli_arg)
                if value is not None:
                    overrides[config_field] = value
        
        return overrides
    
    @staticmethod
    def get_supported_models() -> Dict[str, Dict[str, Any]]:
        """
        Get information about supported models.
        
        Returns:
            Dictionary with supported model information
        """
        return {
            "plm_models": {
                "VietAI/vit5-base": {
                    "type": "plm",
                    "architecture": "encoder-decoder",
                    "size": "223M",
                    "supported_methods": ["finetune", "instruct"]
                },
                "VietAI/vit5-large": {
                    "type": "plm", 
                    "architecture": "encoder-decoder",
                    "size": "737M",
                    "supported_methods": ["finetune", "instruct"]
                },
                "vinai/bartpho-base": {
                    "type": "plm",
                    "architecture": "encoder-decoder", 
                    "size": "132M",
                    "supported_methods": ["finetune", "instruct"]
                },
                "vinai/bartpho-large": {
                    "type": "plm",
                    "architecture": "encoder-decoder",
                    "size": "409M", 
                    "supported_methods": ["finetune", "instruct"]
                }
            },
            "llm_models": {
                "Qwen/Qwen2-7B-Instruct": {
                    "type": "llm",
                    "architecture": "decoder-only",
                    "size": "7.6B",
                    "supported_methods": ["instruct"]
                },
                "Qwen/Qwen2.5-7B-Instruct": {
                    "type": "llm",
                    "architecture": "decoder-only",
                    "size": "7.6B", 
                    "supported_methods": ["instruct"]
                },
                "SeaLLMs/SeaLLMs-v3-7B-Chat": {
                    "type": "llm",
                    "architecture": "decoder-only",
                    "size": "7.2B",
                    "supported_methods": ["instruct"]
                },
                "vilm/vinallama-7b-chat": {
                    "type": "llm",
                    "architecture": "decoder-only", 
                    "size": "6.9B",
                    "supported_methods": ["instruct"]
                }
            }
        }
    
    @staticmethod
    def validate_model_compatibility(
        model_name: str,
        model_type: str,
        training_method: str
    ) -> bool:
        """
        Validate if model name is compatible with specified type and method.
        
        Args:
            model_name: Model name
            model_type: Model type
            training_method: Training method
            
        Returns:
            True if compatible, False otherwise
        """
        supported = ConfigFactory.get_supported_models()
        
        # Check in PLM models
        if model_type == "plm":
            model_info = supported["plm_models"].get(model_name)
            if model_info:
                return training_method in model_info["supported_methods"]
        
        # Check in LLM models
        elif model_type == "llm":
            model_info = supported["llm_models"].get(model_name)
            if model_info:
                return training_method in model_info["supported_methods"]
        
        # If not in predefined list, use heuristics
        inferred_type = ConfigFactory._infer_model_type(model_name)
        return inferred_type == model_type


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
    parser.add_argument("--train_ratio", type=float, default=0.8,
                       help="Training data ratio (for auto split)")
    parser.add_argument("--val_ratio", type=float, default=0.1, 
                       help="Validation data ratio (for auto split)")
    parser.add_argument("--test_ratio", type=float, default=0.1,
                       help="Test data ratio (for auto split)")
    
    # ===== TRAINING PARAMETERS =====
    parser.add_argument("--output_dir", type=str, default="./outputs",
                       help="Output directory")
    parser.add_argument("--num_train_epochs", type=int, default=5,
                       help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2,
                       help="Training batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2,
                       help="Evaluation batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=3e-5,
                       help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.05,
                       help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    
    # ===== SEQUENCE LENGTHS =====
    parser.add_argument("--max_source_length", type=int, default=1024,
                       help="Maximum source sequence length")
    parser.add_argument("--max_target_length", type=int, default=256,
                       help="Maximum target sequence length")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                       help="Maximum sequence length (for LLMs)")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                       help="Maximum new tokens to generate")
    
    # ===== EVALUATION & LOGGING =====
    parser.add_argument("--eval_steps", type=int, default=500,
                       help="Evaluation frequency in steps")
    parser.add_argument("--save_steps", type=int, default=500,
                       help="Save frequency in steps")
    parser.add_argument("--logging_steps", type=int, default=100,
                       help="Logging frequency in steps")
    
    # ===== HARDWARE & OPTIMIZATION =====
    parser.add_argument("--fp16", action="store_true",
                       help="Use 16-bit floating point precision")
    parser.add_argument("--bf16", action="store_true",
                       help="Use bfloat16 precision")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # ===== LORA PARAMETERS =====
    parser.add_argument("--lora_r", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                       help="LoRA dropout")
    
    # ===== QUANTIZATION =====
    parser.add_argument("--use_qlora", action="store_true",
                       help="Use QLoRA quantization")
    parser.add_argument("--load_in_4bit", action="store_true",
                       help="Load model in 4-bit")
    parser.add_argument("--load_in_8bit", action="store_true",
                       help="Load model in 8-bit")
    
    # ===== MONITORING =====
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="vibidlqa-aqa",
                       help="W&B project name")
    parser.add_argument("--wandb_entity", type=str,
                       help="W&B entity/team name")
    
    # ===== PATHS =====
    parser.add_argument("--checkpoint_path", type=str,
                       help="Path to model checkpoint for inference")
    parser.add_argument("--results_file", type=str,
                       help="Path to results file for evaluation")
    parser.add_argument("--cache_dir", type=str,
                       help="Cache directory for models and datasets")
    
    return parser