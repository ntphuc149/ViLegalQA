"""
Configuration for Pre-trained Language Models (PLMs) - Encoder-Decoder models
Supports ViT5, BARTPho and similar seq2seq models for Abstractive QA.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from ..base_config import BaseConfig


@dataclass
class PLMConfig(BaseConfig):
    """
    Configuration for PLM (Encoder-Decoder) models like ViT5, BARTPho.
    
    These models use sequence-to-sequence architecture and can be trained with
    either traditional fine-tuning or instruction-based fine-tuning.
    """
    
    # Override base model type
    model_type: str = "plm"
    
    # ===== PLM-SPECIFIC MODEL CONFIGURATION =====
    model_name: str = "VietAI/vit5-base"
    """PLM model name. Supported: VietAI/vit5-base, VietAI/vit5-large, vinai/bartpho-*"""
    
    # ===== TOKENIZATION CONFIGURATION =====
    max_source_length: int = 1024
    """Maximum length of input sequence (context + question)."""
    
    max_target_length: int = 256
    """Maximum length of target sequence (answer)."""
    
    padding: str = "max_length"
    """Padding strategy: 'max_length', 'longest', or 'do_not_pad'."""
    
    truncation: bool = True
    """Whether to truncate sequences that exceed max_length."""
    
    # ===== GENERATION CONFIGURATION =====
    predict_with_generate: bool = True
    """Whether to use generation mode during evaluation."""
    
    generation_max_length: Optional[int] = None
    """Max length for generation. If None, uses max_target_length."""
    
    generation_num_beams: int = 4
    """Number of beams for beam search during generation."""
    
    # ===== INSTRUCTION TUNING CONFIGURATION =====
    use_instruction_format: bool = False
    """Whether to use instruction format for input processing."""
    
    instruction_template: str = "vietnamese_legal"
    """Template name for instruction formatting."""
    
    # ===== TRAINING SPECIFIC =====
    gradient_accumulation_steps: int = 16
    """Number of steps to accumulate gradients before updating."""
    
    group_by_length: bool = True
    """Whether to group samples by length for efficient training."""
    
    # ===== PLM MODEL VARIANTS =====
    supported_models: List[str] = field(default_factory=lambda: [
        "VietAI/vit5-base",
        "VietAI/vit5-large", 
        "vinai/bartpho-syllable",
        "vinai/bartpho-word",
        "vinai/bartpho-syllable-base",
        "vinai/bartpho-word-base"
    ])
    """List of supported PLM models."""
    
    def __post_init__(self):
        """Post-initialization validation for PLM-specific parameters."""
        super().__post_init__()
        
        # Validate model name
        if self.model_name not in self.supported_models:
            print(f"Warning: {self.model_name} not in officially supported models.")
            print(f"Supported models: {self.supported_models}")
        
        # Set generation max length if not specified
        if self.generation_max_length is None:
            self.generation_max_length = self.max_target_length
        
        # Validate instruction settings
        if self.training_method == "instruct" and not self.use_instruction_format:
            print("Warning: training_method='instruct' but use_instruction_format=False")
        
        # Adjust batch size for memory efficiency
        if "large" in self.model_name.lower() and self.per_device_train_batch_size > 1:
            print(f"Warning: Large model detected. Consider reducing batch size from {self.per_device_train_batch_size}")
        
        # Validate length settings
        if self.max_source_length + self.max_target_length > 2048:
            print(f"Warning: Total sequence length ({self.max_source_length + self.max_target_length}) is very large")
    
    def get_model_family(self) -> str:
        """Get the model family name (vit5, bartpho, etc.)."""
        model_lower = self.model_name.lower()
        if "vit5" in model_lower:
            return "vit5"
        elif "bartpho" in model_lower:
            return "bartpho"
        else:
            return "unknown"
    
    def is_large_model(self) -> bool:
        """Check if this is a large model variant."""
        return "large" in self.model_name.lower()
    
    def get_recommended_batch_size(self) -> int:
        """Get recommended batch size based on model size."""
        if self.is_large_model():
            return 1  # Large models need smaller batch size
        else:
            return 2  # Base models can handle larger batch size