"""
Configuration for Large Language Models (LLMs) - Decoder-only models
Supports Qwen2, SeaLLM, VinaLLaMa and similar models for Abstractive QA with instruction tuning.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from ..base_config import BaseConfig


@dataclass
class LLMConfig(BaseConfig):
    """
    Configuration for LLM (Decoder-only) models like Qwen2, SeaLLM, VinaLLaMa.
    
    These models typically use instruction-tuning with QLoRA for memory efficiency.
    """
    
    # Override base configurations
    model_type: str = "llm"
    training_method: str = "instruct"  # LLMs primarily use instruction tuning
    
    # ===== LLM-SPECIFIC MODEL CONFIGURATION =====
    model_name: str = "Qwen/Qwen2-7B-Instruct"
    """LLM model name. Supported: Qwen2.5, Qwen2, SeaLLM-v3, VinaLLaMa variants."""
    
    # ===== SEQUENCE LENGTH CONFIGURATION =====
    max_seq_length: int = 2048
    """Maximum sequence length for input + output combined."""
    
    # ===== QUANTIZATION CONFIGURATION (QLoRA) =====
    use_qlora: bool = True
    """Whether to use QLoRA (Quantized LoRA) for memory-efficient training."""
    
    load_in_4bit: bool = True
    """Whether to load model in 4-bit precision."""
    
    load_in_8bit: bool = False
    """Whether to load model in 8-bit precision (alternative to 4-bit)."""
    
    bnb_4bit_quant_type: str = "nf4"
    """Quantization type for 4-bit: 'fp4' or 'nf4'. nf4 is usually better."""
    
    bnb_4bit_compute_dtype: str = "float16"
    """Compute dtype for 4-bit quantization: 'float16' or 'bfloat16'."""
    
    bnb_4bit_use_double_quant: bool = True
    """Whether to use double quantization for additional memory savings."""
    
    # ===== LORA CONFIGURATION =====
    lora_r: int = 16
    """LoRA rank. Higher values = more parameters but potentially better performance."""
    
    lora_alpha: int = 32
    """LoRA alpha parameter. Typically 2x the rank value."""
    
    lora_dropout: float = 0.05
    """LoRA dropout rate for regularization."""
    
    lora_target_modules: List[str] = field(default_factory=lambda: [
        'up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj'
    ])
    """Target modules for LoRA adaptation. Covers most attention and MLP layers."""
    
    lora_bias: str = "none"
    """LoRA bias setting: 'none', 'all', or 'lora_only'."""
    
    lora_task_type: str = "CAUSAL_LM"
    """LoRA task type for the model."""
    
    # ===== TRAINING CONFIGURATION =====
    gradient_accumulation_steps: int = 2
    """Steps to accumulate gradients. Increase if batch size is too small."""
    
    packing: bool = True
    """Whether to use sequence packing for efficiency."""
    
    dataset_text_field: str = "instruction"
    """Field name in dataset containing the instruction text."""
    
    # ===== OPTIMIZER CONFIGURATION =====
    optim: str = "paged_adamw_32bit"
    """Optimizer type. paged_adamw_32bit is memory-efficient for large models."""
    
    # ===== INSTRUCTION TEMPLATE CONFIGURATION =====
    chat_template: str = "chatml"
    """Chat template format: 'chatml', 'vicuna', 'alpaca', etc."""
    
    system_message: str = "Bạn là chuyên gia về lĩnh vực pháp luật tại Việt Nam."
    """System message for instruction formatting."""
    
    # ===== SUPPORTED MODELS =====
    supported_models: List[str] = field(default_factory=lambda: [
        # Qwen2.5 family
        "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct", 
        "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct",
        
        # Qwen2 family  
        "Qwen/Qwen2-0.5B", "Qwen/Qwen2-0.5B-Instruct",
        "Qwen/Qwen2-1.5B", "Qwen/Qwen2-1.5B-Instruct",
        "Qwen/Qwen2-7B", "Qwen/Qwen2-7B-Instruct",
        
        # SeaLLM family
        "SeaLLMs/SeaLLMs-v3-1.5B", "SeaLLMs/SeaLLMs-v3-1.5B-Chat",
        "SeaLLMs/SeaLLMs-v3-7B", "SeaLLMs/SeaLLMs-v3-7B-Chat",
        
        # VinaLLaMa family
        "vilm/vinallama-2.7b", "vilm/vinallama-2.7b-chat",
        "vilm/vinallama-7b", "vilm/vinallama-7b-chat"
    ])
    """List of supported LLM models."""
    
    def __post_init__(self):
        """Post-initialization validation for LLM-specific parameters."""
        super().__post_init__()
        
        # Validate model name
        if self.model_name not in self.supported_models:
            print(f"Warning: {self.model_name} not in officially supported models.")
            print(f"Supported models: {self.supported_models}")
        
        # Validate quantization settings
        if self.load_in_4bit and self.load_in_8bit:
            raise ValueError("Cannot use both 4-bit and 8-bit quantization")
        
        if not self.use_qlora and (self.load_in_4bit or self.load_in_8bit):
            print("Warning: Quantization enabled but use_qlora=False")
        
        # Validate LoRA settings
        if self.lora_alpha < self.lora_r:
            print(f"Warning: lora_alpha ({self.lora_alpha}) is less than lora_r ({self.lora_r})")
        
        # Adjust settings based on model size
        model_size = self.get_model_size()
        if model_size >= 7.0:  # 7B+ models
            if self.per_device_train_batch_size > 1:
                print(f"Warning: Large model ({model_size}B parameters). Consider batch_size=1")
            if self.gradient_accumulation_steps < 2:
                print(f"Warning: Large model. Consider gradient_accumulation_steps >= 2")
        
        # Force instruction method for LLMs
        if self.training_method != "instruct":
            print("Warning: LLMs typically use instruction tuning. Setting training_method='instruct'")
            self.training_method = "instruct"
    
    def get_model_family(self) -> str:
        """Get the model family name."""
        model_lower = self.model_name.lower()
        if "qwen2.5" in model_lower:
            return "qwen2.5"
        elif "qwen2" in model_lower:
            return "qwen2"
        elif "seallm" in model_lower:
            return "seallm"
        elif "vinallama" in model_lower or "vilm" in model_lower:
            return "vinallama"
        else:
            return "unknown"
    
    def get_model_size(self) -> float:
        """Extract model size in billions of parameters."""
        model_lower = self.model_name.lower()
        if "0.5b" in model_lower:
            return 0.5
        elif "1.5b" in model_lower:
            return 1.5
        elif "2.7b" in model_lower:
            return 2.7
        elif "3b" in model_lower:
            return 3.0
        elif "7b" in model_lower:
            return 7.0
        else:
            return 7.0  # Default assumption for unknown sizes
    
    def is_instruct_model(self) -> bool:
        """Check if this is an instruction-tuned model."""
        model_lower = self.model_name.lower()
        return any(keyword in model_lower for keyword in ["instruct", "chat"])
    
    def get_recommended_settings(self) -> dict:
        """Get recommended settings based on model size and type."""
        model_size = self.get_model_size()
        
        settings = {
            "per_device_train_batch_size": 1 if model_size >= 7.0 else 2,
            "gradient_accumulation_steps": 4 if model_size >= 7.0 else 2,
            "learning_rate": 1e-5 if model_size >= 7.0 else 3e-5,
            "num_train_epochs": 3 if model_size >= 7.0 else 5,
        }
        
        return settings