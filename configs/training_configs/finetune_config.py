"""
Configuration for traditional fine-tuning of PLMs
Used for sequence-to-sequence training without instruction formatting.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from ..model_configs.plm_config import PLMConfig


@dataclass 
class FinetuneConfig(PLMConfig):
    """
    Configuration for traditional fine-tuning of PLM models.
    
    This config is specifically for standard seq2seq fine-tuning where
    the model learns to map (context, question) -> answer directly.
    """
    
    # Override training method
    training_method: str = "finetune"
    use_instruction_format: bool = False
    
    # ===== FINE-TUNING SPECIFIC PARAMETERS =====
    
    # Data processing
    source_prefix: str = ""
    """Prefix to add to source sequences (e.g., 'translate: ' for T5)."""
    
    target_prefix: str = ""
    """Prefix to add to target sequences."""
    
    ignore_pad_token_for_loss: bool = True
    """Whether to ignore padding tokens in loss calculation."""
    
    # Training dynamics
    warmup_steps: Optional[int] = None
    """Number of warmup steps. If None, uses warmup_ratio."""
    
    lr_scheduler_type: str = "linear"
    """Learning rate scheduler type: 'linear', 'cosine', 'polynomial', etc."""
    
    # Data collation
    max_train_samples: Optional[int] = None
    """Maximum number of training samples to use."""
    
    max_eval_samples: Optional[int] = None  
    """Maximum number of evaluation samples to use."""
    
    preprocessing_num_workers: int = 4
    """Number of processes for data preprocessing."""
    
    # ===== GENERATION PARAMETERS FOR EVALUATION =====
    eval_with_generate: bool = True
    """Whether to use generation for evaluation instead of teacher forcing."""
    
    # ===== MEMORY OPTIMIZATION =====
    remove_unused_columns: bool = True
    """Whether to remove unused columns from dataset."""
    
    include_inputs_for_metrics: bool = False
    """Whether to include inputs when computing metrics."""
    
    # ===== MODEL-SPECIFIC OPTIMIZATIONS =====
    model_specific_settings: Dict[str, Any] = field(default_factory=dict)
    """Model-specific settings that override defaults."""
    
    def __post_init__(self):
        """Post-initialization validation and setup for fine-tuning."""
        super().__post_init__()
        
        # Ensure fine-tuning specific settings
        if self.use_instruction_format:
            print("Warning: use_instruction_format=True but training_method='finetune'")
            print("Consider using InstructConfig for instruction-based training")
        
        # Set warmup steps if not provided
        if self.warmup_steps is None and hasattr(self, 'num_train_epochs'):
            # Estimate total steps for warmup calculation
            estimated_steps = 1000  # Rough estimate, will be calculated properly during training
            self.warmup_steps = int(estimated_steps * self.warmup_ratio)
        
        # Apply model-specific optimizations
        self._apply_model_specific_settings()
        
        # Validate generation settings for evaluation
        if self.eval_with_generate and not self.predict_with_generate:
            print("Setting predict_with_generate=True for evaluation with generation")
            self.predict_with_generate = True
    
    def _apply_model_specific_settings(self):
        """Apply model-specific optimizations."""
        model_family = self.get_model_family()
        
        # ViT5 specific settings
        if model_family == "vit5":
            if not self.model_specific_settings:
                self.model_specific_settings = {
                    "source_prefix": "",  # ViT5 doesn't need prefix
                    "early_stopping": True,
                    "generation_num_beams": 4
                }
        
        # BARTPho specific settings  
        elif model_family == "bartpho":
            if not self.model_specific_settings:
                self.model_specific_settings = {
                    "source_prefix": "",  # BARTPho doesn't need prefix
                    "early_stopping": True,
                    "generation_num_beams": 4,
                    "length_penalty": 1.0
                }
        
        # Apply the settings
        for key, value in self.model_specific_settings.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def get_training_args_dict(self) -> Dict[str, Any]:
        """Get training arguments as dictionary for Seq2SeqTrainingArguments."""
        return {
            "output_dir": self.output_dir,
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "learning_rate": self.learning_rate,
            "warmup_ratio": self.warmup_ratio,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "adam_epsilon": self.adam_epsilon,
            "max_grad_norm": self.max_grad_norm,
            "lr_scheduler_type": self.lr_scheduler_type,
            "logging_steps": self.logging_steps,
            "eval_steps": self.eval_steps,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "evaluation_strategy": self.evaluation_strategy,
            "load_best_model_at_end": self.load_best_model_at_end,
            "metric_for_best_model": self.metric_for_best_model,
            "greater_is_better": self.greater_is_better,
            "predict_with_generate": self.predict_with_generate,
            "generation_max_length": self.generation_max_length,
            "generation_num_beams": self.generation_num_beams,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "dataloader_num_workers": self.dataloader_num_workers,
            "dataloader_pin_memory": self.dataloader_pin_memory,
            "gradient_checkpointing": self.gradient_checkpointing,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "group_by_length": self.group_by_length,
            "remove_unused_columns": self.remove_unused_columns,
            "include_inputs_for_metrics": self.include_inputs_for_metrics,
            "seed": self.seed,
            "report_to": "wandb" if self.use_wandb else "none"
        }