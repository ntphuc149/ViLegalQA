"""
PLM (Pre-trained Language Model) configuration for encoder-decoder models.
Supports ViT5, BARTPho, and other encoder-decoder architectures.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from ..base_config import BaseConfig


@dataclass
class PLMConfig(BaseConfig):
    """
    Configuration for PLM (encoder-decoder) models like ViT5, BARTPho.
    """
    
    # ===== MODEL TYPE OVERRIDE =====
    model_type: str = "plm"  # Fixed for PLM models
    
    # ===== PLM SPECIFIC PARAMETERS =====
    model_name: str = "VietAI/vit5-base"  # Default to ViT5
    tokenizer_name: Optional[str] = None  # Use model_name if None
    
    # ===== SEQUENCE LENGTHS =====
    max_source_length: int = 1024  # Input sequence length
    max_target_length: int = 256   # Output sequence length
    
    # ===== TRAINING SPECIFIC =====
    predict_with_generate: bool = True
    generation_max_length: int = 256
    generation_num_beams: int = 1
    early_stopping: bool = True
    
    # ===== INSTRUCTION TUNING =====
    use_instruction_format: bool = False
    instruction_template: str = "vietnamese_legal"
    system_message: str = "Bạn là chuyên gia về lĩnh vực pháp luật tại Việt Nam."
    
    # ===== DATA PROCESSING =====
    padding: str = "max_length"  # "max_length", "longest", "do_not_pad"
    truncation: bool = True
    pad_to_max_length: bool = True
    ignore_pad_token_for_loss: bool = True
    
    # ===== OPTIMIZATION =====
    learning_rate: float = 3e-5
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # ===== EVALUATION =====
    eval_strategy: str = "steps"
    eval_steps: int = 500
    metric_for_best_model: str = "eval_rouge1"
    greater_is_better: bool = True
    
    # ===== GENERATION PARAMETERS =====
    do_sample: bool = True
    temperature: float = 0.1
    top_p: float = 0.75
    top_k: int = 50
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    
    # ===== MODEL ARCHITECTURE SPECIFIC =====
    dropout_rate: Optional[float] = None
    attention_dropout: Optional[float] = None
    
    def __post_init__(self):
        """Post-initialization for PLM-specific validation."""
        super().__post_init__()
        self.validate_plm_config()
        self.setup_instruction_format()
    
    def validate_plm_config(self) -> None:
        """Validate PLM-specific configuration."""
        # Ensure model type is correct
        if self.model_type != "plm":
            raise ValueError(f"PLMConfig requires model_type='plm', got '{self.model_type}'")
        
        # Validate generation parameters
        if self.generation_max_length > self.max_target_length:
            raise ValueError(
                f"generation_max_length ({self.generation_max_length}) "
                f"cannot exceed max_target_length ({self.max_target_length})"
            )
        
        # Validate beam search
        if self.generation_num_beams < 1:
            raise ValueError("generation_num_beams must be >= 1")
        
        # Validate padding strategy
        if self.padding not in ["max_length", "longest", "do_not_pad"]:
            raise ValueError(f"Invalid padding strategy: {self.padding}")
        
        # Validate supported models
        supported_models = [
            "VietAI/vit5", "VietAI/vit5-base", "VietAI/vit5-large",
            "vinai/bartpho-base", "vinai/bartpho-large",
            "facebook/bart-base", "facebook/bart-large",
            "google/t5-small", "google/t5-base", "google/t5-large"
        ]
        
        if not any(model in self.model_name for model in ["vit5", "bart", "t5"]):
            print(f"Warning: {self.model_name} may not be supported. "
                  f"Supported models include ViT5, BARTPho, T5 variants.")
    
    def setup_instruction_format(self) -> None:
        """Setup instruction formatting if enabled."""
        if self.use_instruction_format or self.training_method == "instruct":
            self.use_instruction_format = True
            
            # Adjust max lengths for instruction format
            instruction_overhead = 100  # Estimated tokens for instruction formatting
            self.max_source_length = max(512, self.max_source_length - instruction_overhead)
    
    def get_model_config_updates(self) -> Dict[str, Any]:
        """Get model configuration updates."""
        config_updates = {}
        
        if self.dropout_rate is not None:
            config_updates["dropout_rate"] = self.dropout_rate
        
        if self.attention_dropout is not None:
            config_updates["attention_dropout"] = self.attention_dropout
        
        return config_updates
    
    def get_generation_config(self) -> Dict[str, Any]:
        """Get generation configuration for inference."""
        return {
            "max_length": self.generation_max_length,
            "max_new_tokens": self.max_new_tokens,
            "num_beams": self.generation_num_beams,
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "length_penalty": self.length_penalty,
            "early_stopping": self.early_stopping,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "pad_token_id": None,  # Will be set by tokenizer
            "eos_token_id": None,  # Will be set by tokenizer
        }
    
    def get_training_args_updates(self) -> Dict[str, Any]:
        """Get training arguments specific to PLM training."""
        return {
            "predict_with_generate": self.predict_with_generate,
            "generation_max_length": self.generation_max_length,
            "generation_num_beams": self.generation_num_beams,
            "include_inputs_for_metrics": False,
        }
    
    def is_vietnamese_model(self) -> bool:
        """Check if model is Vietnamese-specific."""
        vietnamese_models = ["vit5", "bartpho", "vitext2sql"]
        return any(model in self.model_name.lower() for model in vietnamese_models)
    
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Get preprocessing configuration."""
        return {
            "max_source_length": self.max_source_length,
            "max_target_length": self.max_target_length,
            "padding": self.padding,
            "truncation": self.truncation,
            "pad_to_max_length": self.pad_to_max_length,
            "ignore_pad_token_for_loss": self.ignore_pad_token_for_loss,
            "use_instruction_format": self.use_instruction_format,
            "instruction_template": self.instruction_template,
            "system_message": self.system_message,
        }


# Predefined configurations for common PLM models
@dataclass 
class ViT5BaseConfig(PLMConfig):
    """Configuration for ViT5-base model."""
    model_name: str = "VietAI/vit5-base"
    max_source_length: int = 1024
    max_target_length: int = 256
    learning_rate: float = 3e-5
    per_device_train_batch_size: int = 2
    generation_num_beams: int = 4


@dataclass
class ViT5LargeConfig(PLMConfig):
    """Configuration for ViT5-large model."""
    model_name: str = "VietAI/vit5-large"
    max_source_length: int = 1024
    max_target_length: int = 256
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    generation_num_beams: int = 4


@dataclass
class BARTPhoBaseConfig(PLMConfig):
    """Configuration for BARTPho-base model."""
    model_name: str = "vinai/bartpho-base"
    max_source_length: int = 1024
    max_target_length: int = 256
    learning_rate: float = 3e-5
    per_device_train_batch_size: int = 2
    generation_num_beams: int = 4


@dataclass
class BARTPhoLargeConfig(PLMConfig):
    """Configuration for BARTPho-large model."""
    model_name: str = "vinai/bartpho-large"
    max_source_length: int = 1024
    max_target_length: int = 256
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    generation_num_beams: int = 4


# Instruction tuning configurations
@dataclass
class ViT5InstructConfig(ViT5BaseConfig):
    """Configuration for ViT5 instruction tuning."""
    training_method: str = "instruct"
    use_instruction_format: bool = True
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.1
    eval_steps: int = 250


@dataclass
class BARTPhoInstructConfig(BARTPhoBaseConfig):
    """Configuration for BARTPho instruction tuning."""
    training_method: str = "instruct"
    use_instruction_format: bool = True
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.1
    eval_steps: int = 250


def get_plm_config(model_name: str, training_method: str = "finetune") -> PLMConfig:
    """
    Get appropriate PLM configuration based on model name and training method.
    
    Args:
        model_name: Model name/identifier
        training_method: Training method ("finetune" or "instruct")
        
    Returns:
        Appropriate PLMConfig instance
    """
    model_configs = {
        ("VietAI/vit5-base", "finetune"): ViT5BaseConfig,
        ("VietAI/vit5-base", "instruct"): ViT5InstructConfig,
        ("VietAI/vit5-large", "finetune"): ViT5LargeConfig,
        ("vinai/bartpho-base", "finetune"): BARTPhoBaseConfig,
        ("vinai/bartpho-base", "instruct"): BARTPhoInstructConfig,
        ("vinai/bartpho-large", "finetune"): BARTPhoLargeConfig,
    }
    
    config_class = model_configs.get((model_name, training_method))
    
    if config_class:
        return config_class()
    else:
        # Fallback to base config with model name
        config = PLMConfig(model_name=model_name, training_method=training_method)
        
        # Apply instruction tuning settings if needed
        if training_method == "instruct":
            config.use_instruction_format = True
            config.learning_rate = 5e-5
            config.warmup_ratio = 0.1
        
        return config