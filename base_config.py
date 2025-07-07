"""
Base configuration classes for ViBidLQA-AQA system.
Provides shared configuration parameters and validation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import os


@dataclass
class BaseConfig:
    """
    Base configuration class with common parameters.
    All other configs inherit from this class.
    """
    
    # ===== BASIC INFORMATION =====
    experiment_name: str = "vibidlqa_experiment"
    description: str = "ViBidLQA Abstractive QA Experiment"
    
    # ===== DATASET CONFIGURATION =====
    dataset_name: str = "Truong-Phuc/ViBidLQA"
    use_auth_token: bool = True
    data_split_mode: str = "auto"  # "auto" or "predefined"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    max_samples: Optional[int] = None  # Limit dataset size for testing
    
    # ===== MODEL CONFIGURATION =====
    model_name: str = "VietAI/vit5-base"
    model_type: str = "plm"  # "plm" (encoder-decoder) or "llm" (decoder-only)
    training_method: str = "finetune"  # "finetune" or "instruct"
    
    # ===== TRAINING PARAMETERS =====
    output_dir: str = "./outputs"
    num_train_epochs: int = 5
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-5
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # ===== EVALUATION & LOGGING =====
    eval_strategy: str = "steps"  # "steps", "epoch", "no"
    eval_steps: int = 500
    save_strategy: str = "steps"  # "steps", "epoch", "no"
    save_steps: int = 500
    save_total_limit: int = 3
    logging_steps: int = 100
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # ===== INFERENCE PARAMETERS =====
    max_new_tokens: int = 256
    do_sample: bool = True
    temperature: float = 0.1
    top_p: float = 0.75
    top_k: int = 50
    repetition_penalty: float = 1.0
    
    # ===== EVALUATION METRICS =====
    metrics: List[str] = field(default_factory=lambda: [
        "rouge", "bleu", "meteor", "bertscore"
    ])
    
    # ===== PREPROCESSING PARAMETERS =====
    max_source_length: int = 1024
    max_target_length: int = 256
    preprocessing_num_workers: int = 4
    
    # ===== HARDWARE & OPTIMIZATION =====
    device: str = "auto"  # "auto", "cpu", "cuda", "cuda:0", etc.
    fp16: bool = False
    bf16: bool = False
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True
    
    # ===== REPRODUCIBILITY =====
    seed: int = 42
    
    # ===== MONITORING & LOGGING =====
    use_wandb: bool = False
    wandb_project: str = "vibidlqa-aqa"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    log_level: str = "INFO"
    
    # ===== PATHS =====
    cache_dir: Optional[str] = None
    checkpoint_path: Optional[str] = None
    results_file: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        self.validate()
        self.setup_paths()
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        # Validate ratios
        if self.data_split_mode == "auto":
            total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
            if not (0.99 <= total_ratio <= 1.01):  # Allow small floating point errors
                raise ValueError(f"Data split ratios must sum to 1.0, got {total_ratio}")
        
        # Validate model type
        if self.model_type not in ["plm", "llm"]:
            raise ValueError(f"model_type must be 'plm' or 'llm', got {self.model_type}")
        
        # Validate training method
        if self.training_method not in ["finetune", "instruct"]:
            raise ValueError(f"training_method must be 'finetune' or 'instruct', got {self.training_method}")
        
        # Validate eval strategy
        if self.eval_strategy not in ["steps", "epoch", "no"]:
            raise ValueError(f"eval_strategy must be 'steps', 'epoch', or 'no', got {self.eval_strategy}")
        
        # Validate batch sizes
        if self.per_device_train_batch_size <= 0:
            raise ValueError("per_device_train_batch_size must be positive")
        if self.per_device_eval_batch_size <= 0:
            raise ValueError("per_device_eval_batch_size must be positive")
        
        # Validate learning rate
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        
        # Validate sequence lengths
        if self.max_source_length <= 0:
            raise ValueError("max_source_length must be positive")
        if self.max_target_length <= 0:
            raise ValueError("max_target_length must be positive")
    
    def setup_paths(self) -> None:
        """Setup and ensure output paths exist."""
        # Convert string paths to Path objects
        self.output_dir = Path(self.output_dir)
        
        if self.cache_dir:
            self.cache_dir = Path(self.cache_dir)
        
        if self.checkpoint_path:
            self.checkpoint_path = Path(self.checkpoint_path)
        
        if self.results_file:
            self.results_file = Path(self.results_file)
    
    def get_effective_batch_size(self) -> int:
        """Get effective batch size including gradient accumulation."""
        return self.per_device_train_batch_size * self.gradient_accumulation_steps
    
    def get_output_paths(self) -> Dict[str, Path]:
        """Get all output paths for the experiment."""
        base_dir = Path(self.output_dir)
        
        return {
            "model": base_dir / "model",
            "logs": base_dir / "logs", 
            "results": base_dir / "results",
            "configs": base_dir / "configs",
            "checkpoints": base_dir / "checkpoints"
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        config_dict = {}
        
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
            else:
                config_dict[key] = value
        
        return config_dict
    
    def update_from_dict(self, updates: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
        
        # Re-validate after updates
        self.validate()
        self.setup_paths()
    
    def get_model_cache_dir(self) -> Optional[Path]:
        """Get model cache directory."""
        if self.cache_dir:
            return Path(self.cache_dir) / "models"
        return None
    
    def get_dataset_cache_dir(self) -> Optional[Path]:
        """Get dataset cache directory."""
        if self.cache_dir:
            return Path(self.cache_dir) / "datasets"
        return None
    
    def is_encoder_decoder_model(self) -> bool:
        """Check if model is encoder-decoder type."""
        return self.model_type == "plm"
    
    def is_decoder_only_model(self) -> bool:
        """Check if model is decoder-only type."""
        return self.model_type == "llm"
    
    def is_instruction_tuning(self) -> bool:
        """Check if using instruction tuning."""
        return self.training_method == "instruct"
    
    def get_wandb_config(self) -> Dict[str, Any]:
        """Get configuration for W&B logging."""
        wandb_config = self.to_dict()
        
        # Add computed fields
        wandb_config["effective_batch_size"] = self.get_effective_batch_size()
        wandb_config["total_train_steps_estimate"] = "computed_during_training"
        
        return wandb_config
    
    def __repr__(self) -> str:
        """String representation of config."""
        return f"{self.__class__.__name__}(model={self.model_name}, method={self.training_method})"


def load_config_from_env() -> Dict[str, Any]:
    """Load configuration values from environment variables."""
    env_config = {}
    
    # Map environment variables to config fields
    env_mappings = {
        "VIBIDLQA_MODEL_NAME": "model_name",
        "VIBIDLQA_DATASET_NAME": "dataset_name", 
        "VIBIDLQA_OUTPUT_DIR": "output_dir",
        "VIBIDLQA_BATCH_SIZE": "per_device_train_batch_size",
        "VIBIDLQA_LEARNING_RATE": "learning_rate",
        "VIBIDLQA_EPOCHS": "num_train_epochs",
        "VIBIDLQA_SEED": "seed",
        "WANDB_PROJECT": "wandb_project",
        "WANDB_ENTITY": "wandb_entity",
        "HF_TOKEN": "use_auth_token"
    }
    
    for env_var, config_key in env_mappings.items():
        value = os.getenv(env_var)
        if value is not None:
            # Type conversion based on config key
            if config_key in ["per_device_train_batch_size", "num_train_epochs", "seed"]:
                env_config[config_key] = int(value)
            elif config_key in ["learning_rate"]:
                env_config[config_key] = float(value)
            elif config_key in ["use_auth_token"]:
                env_config[config_key] = value.lower() in ["true", "1", "yes"]
            else:
                env_config[config_key] = value
    
    return env_config