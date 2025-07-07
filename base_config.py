"""
Base configuration class for ViBidLQA-AQA
Contains common parameters shared across all model types and training methods.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any
import os
from pathlib import Path


@dataclass
class BaseConfig:
    """
    Base configuration class containing common parameters for all AQA tasks.
    
    This class serves as the foundation for more specific configurations
    (PLM, LLM, fine-tuning, instruction-tuning specific configs).
    """
    
    # ===== DATASET CONFIGURATION =====
    dataset_name: str = "Truong-Phuc/ViBidLQA"
    """Dataset name or path. Can be HuggingFace dataset name or local path."""
    
    use_auth_token: bool = True
    """Whether to use authentication token for private datasets."""
    
    data_split_mode: str = "auto"
    """Data splitting mode: 'auto' (custom split) or 'predefined' (use existing splits)."""
    
    train_ratio: float = 0.8
    """Training set ratio when using auto split mode."""
    
    val_ratio: float = 0.1
    """Validation set ratio when using auto split mode."""
    
    test_ratio: float = 0.1
    """Test set ratio when using auto split mode."""
    
    max_samples: Optional[int] = None
    """Maximum number of samples to use (for debugging). None means use all."""
    
    # ===== MODEL CONFIGURATION =====
    model_name: str = "VietAI/vit5-base"
    """Model name or path. Can be HuggingFace model name or local path."""
    
    model_type: str = "plm"
    """Model type: 'plm' (encoder-decoder) or 'llm' (decoder-only)."""
    
    training_method: str = "finetune"
    """Training method: 'finetune' (traditional) or 'instruct' (instruction-tuning)."""
    
    # ===== TRAINING CONFIGURATION =====
    output_dir: str = "./outputs"
    """Output directory for saving models, logs, and results."""
    
    num_train_epochs: int = 5
    """Number of training epochs."""
    
    per_device_train_batch_size: int = 2
    """Training batch size per device."""
    
    per_device_eval_batch_size: int = 2
    """Evaluation batch size per device."""
    
    learning_rate: float = 3e-5
    """Initial learning rate for training."""
    
    warmup_ratio: float = 0.05
    """Ratio of total training steps used for warmup."""
    
    weight_decay: float = 0.01
    """Weight decay for regularization."""
    
    adam_epsilon: float = 1e-8
    """Epsilon value for Adam optimizer."""
    
    max_grad_norm: float = 1.0
    """Maximum gradient norm for clipping."""
    
    # ===== LOGGING AND EVALUATION =====
    logging_steps: int = 100
    """Number of steps between logging."""
    
    eval_steps: int = 500
    """Number of steps between evaluations."""
    
    save_steps: int = 500
    """Number of steps between model saves."""
    
    save_total_limit: int = 2
    """Maximum number of checkpoints to keep."""
    
    evaluation_strategy: str = "steps"
    """Evaluation strategy: 'steps', 'epoch', or 'no'."""
    
    load_best_model_at_end: bool = True
    """Whether to load the best model at the end of training."""
    
    metric_for_best_model: str = "eval_rouge1"
    """Metric to use for selecting the best model."""
    
    greater_is_better: bool = True
    """Whether higher metric values are better."""
    
    # ===== INFERENCE CONFIGURATION =====
    max_new_tokens: int = 256
    """Maximum number of new tokens to generate during inference."""
    
    do_sample: bool = True
    """Whether to use sampling during generation."""
    
    temperature: float = 0.1
    """Temperature for sampling. Lower values make output more deterministic."""
    
    top_p: float = 0.75
    """Top-p (nucleus) sampling parameter."""
    
    top_k: int = 50
    """Top-k sampling parameter."""
    
    num_beams: int = 4
    """Number of beams for beam search (when do_sample=False)."""
    
    early_stopping: bool = True
    """Whether to use early stopping in beam search."""
    
    # ===== EVALUATION CONFIGURATION =====
    metrics: List[str] = field(default_factory=lambda: ["rouge", "bleu", "meteor", "bertscore"])
    """List of metrics to compute during evaluation."""
    
    bertscore_lang: str = "vi"
    """Language code for BERTScore evaluation."""
    
    # ===== COMPUTE CONFIGURATION =====
    fp16: bool = False
    """Whether to use 16-bit floating point precision."""
    
    bf16: bool = False
    """Whether to use bfloat16 precision (requires newer GPUs)."""
    
    dataloader_num_workers: int = 4
    """Number of workers for data loading."""
    
    dataloader_pin_memory: bool = True
    """Whether to pin memory in data loaders."""
    
    gradient_checkpointing: bool = False
    """Whether to use gradient checkpointing to save memory."""
    
    # ===== REPRODUCIBILITY =====
    seed: int = 42
    """Random seed for reproducibility."""
    
    # ===== RUNTIME CONFIGURATION =====
    do_finetune: bool = False
    """Whether to run fine-tuning stage."""
    
    do_infer: bool = False
    """Whether to run inference stage."""
    
    do_eval: bool = False
    """Whether to run evaluation stage."""
    
    do_end2end: bool = False
    """Whether to run all stages end-to-end."""
    
    checkpoint_path: Optional[str] = None
    """Path to model checkpoint for inference/evaluation."""
    
    results_file: Optional[str] = None
    """Path to results CSV file for evaluation."""
    
    # ===== WANDB CONFIGURATION =====
    use_wandb: bool = False
    """Whether to use Weights & Biases for logging."""
    
    wandb_project: str = "vibidlqa-aqa"
    """W&B project name."""
    
    wandb_run_name: Optional[str] = None
    """W&B run name. If None, will be auto-generated."""
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Validate data split ratios
        if self.data_split_mode == "auto":
            total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
            if abs(total_ratio - 1.0) > 1e-6:
                raise ValueError(f"Data split ratios must sum to 1.0, got {total_ratio}")
        
        # Validate training method and model type compatibility
        if self.model_type not in ["plm", "llm"]:
            raise ValueError(f"model_type must be 'plm' or 'llm', got {self.model_type}")
        
        if self.training_method not in ["finetune", "instruct"]:
            raise ValueError(f"training_method must be 'finetune' or 'instruct', got {self.training_method}")
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Validate stage flags
        stage_flags = [self.do_finetune, self.do_infer, self.do_eval, self.do_end2end]
        if sum(stage_flags) == 0:
            raise ValueError("At least one stage flag must be True")
        
        if self.do_end2end and any([self.do_finetune, self.do_infer, self.do_eval]):
            raise ValueError("Cannot use --do_end2end with other stage flags")
        
        # Validate inference/eval specific requirements
        if self.do_infer and not self.checkpoint_path:
            raise ValueError("--checkpoint_path is required when using --do_infer")
        
        if self.do_eval and not self.results_file:
            raise ValueError("--results_file is required when using --do_eval")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items()}
    
    def save_config(self, path: str) -> None:
        """Save configuration to JSON file."""
        import json
        config_dict = self.to_dict()
        
        # Convert Path objects to strings for JSON serialization
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BaseConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, path: str) -> 'BaseConfig':
        """Load configuration from JSON file."""
        import json
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)