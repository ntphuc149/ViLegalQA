"""
LLM (Large Language Model) configuration for decoder-only models.
Supports Qwen2, SeaLLM, VinaLLaMA with QLoRA and instruction tuning.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from base_config import BaseConfig


@dataclass
class LLMConfig(BaseConfig):
    """
    Configuration for LLM (decoder-only) models like Qwen2, SeaLLM, VinaLLaMA.
    """
    
    # ===== MODEL TYPE OVERRIDE =====
    model_type: str = "llm"  # Fixed for LLM models
    training_method: str = "instruct"  # LLMs typically use instruction tuning
    
    # ===== LLM SPECIFIC PARAMETERS =====
    model_name: str = "Qwen/Qwen2-7B-Instruct"
    tokenizer_name: Optional[str] = None  # Use model_name if None
    trust_remote_code: bool = True
    
    # ===== SEQUENCE LENGTHS =====
    max_seq_length: int = 2048  # Context length for LLMs
    max_new_tokens: int = 256   # Max tokens to generate
    
    # ===== QUANTIZATION (QLoRA) =====
    use_qlora: bool = True
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    bnb_4bit_quant_type: str = "nf4"  # "fp4" or "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"  # "float16", "bfloat16", "float32"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_storage: str = "uint8"
    
    # ===== LORA PARAMETERS =====
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_bias: str = "none"  # "none", "all", "lora_only"
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    lora_modules_to_save: Optional[List[str]] = None
    
    # ===== TRAINING SPECIFIC =====
    learning_rate: float = 1e-5
    warmup_ratio: float = 0.03
    gradient_accumulation_steps: int = 2
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    max_grad_norm: float = 0.3
    weight_decay: float = 0.0
    
    # ===== INSTRUCTION TUNING =====
    packing: bool = True
    dataset_text_field: str = "instruction"
    chat_template: Optional[str] = None
    use_chat_template: bool = True
    
    # ===== SFT TRAINER SPECIFIC =====
    max_seq_length_sft: Optional[int] = None  # Will use max_seq_length if None
    
    # ===== EVALUATION =====
    eval_strategy: str = "steps"
    eval_steps: int = 100
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # ===== GENERATION PARAMETERS =====
    do_sample: bool = True
    temperature: float = 0.1
    top_p: float = 0.75
    top_k: int = 50
    repetition_penalty: float = 1.05
    
    # ===== SYSTEM MESSAGE =====
    system_message: str = "Bạn là chuyên gia về lĩnh vực pháp luật tại Việt Nam."
    
    def __post_init__(self):
        """Post-initialization for LLM-specific validation."""
        super().__post_init__()
        self.validate_llm_config()
        self.setup_llm_defaults()
    
    def validate_llm_config(self) -> None:
        """Validate LLM-specific configuration."""
        # Ensure model type is correct
        if self.model_type != "llm":
            raise ValueError(f"LLMConfig requires model_type='llm', got '{self.model_type}'")
        
        # Validate quantization settings
        if self.load_in_4bit and self.load_in_8bit:
            raise ValueError("Cannot use both 4-bit and 8-bit quantization")
        
        if self.bnb_4bit_quant_type not in ["fp4", "nf4"]:
            raise ValueError(f"Invalid quantization type: {self.bnb_4bit_quant_type}")
        
        if self.bnb_4bit_compute_dtype not in ["float16", "bfloat16", "float32"]:
            raise ValueError(f"Invalid compute dtype: {self.bnb_4bit_compute_dtype}")
        
        # Validate LoRA parameters
        if self.lora_r <= 0:
            raise ValueError("lora_r must be positive")
        
        if self.lora_alpha <= 0:
            raise ValueError("lora_alpha must be positive")
        
        if not (0 <= self.lora_dropout <= 1):
            raise ValueError("lora_dropout must be between 0 and 1")
        
        if self.lora_bias not in ["none", "all", "lora_only"]:
            raise ValueError(f"Invalid lora_bias: {self.lora_bias}")
        
        # Validate supported models
        supported_models = ["qwen", "llama", "mistral", "sea", "vina"]
        if not any(model in self.model_name.lower() for model in supported_models):
            print(f"Warning: {self.model_name} may not be supported. "
                  f"Supported models include Qwen2, LLaMA, Mistral, SeaLLM, VinaLLaMA variants.")
    
    def setup_llm_defaults(self) -> None:
        """Setup LLM-specific defaults."""
        # Set max_seq_length_sft if not provided
        if self.max_seq_length_sft is None:
            self.max_seq_length_sft = self.max_seq_length
        
        # Adjust settings for specific model families
        if "qwen" in self.model_name.lower():
            self.setup_qwen_defaults()
        elif "llama" in self.model_name.lower():
            self.setup_llama_defaults()
        elif "sea" in self.model_name.lower():
            self.setup_seallm_defaults()
    
    def setup_qwen_defaults(self) -> None:
        """Setup defaults for Qwen models."""
        # Qwen-specific LoRA targets
        self.lora_target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
        
        # Adjust compute dtype for Qwen
        if "qwen2.5" in self.model_name.lower():
            self.bnb_4bit_compute_dtype = "bfloat16"
    
    def setup_llama_defaults(self) -> None:
        """Setup defaults for LLaMA models."""
        # LLaMA-specific LoRA targets
        self.lora_target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    
    def setup_seallm_defaults(self) -> None:
        """Setup defaults for SeaLLM models."""
        # SeaLLM is based on LLaMA architecture
        self.setup_llama_defaults()
    
    def get_quantization_config(self) -> Dict[str, Any]:
        """Get quantization configuration for BitsAndBytesConfig."""
        if not (self.use_qlora or self.load_in_4bit or self.load_in_8bit):
            return {}
        
        config = {
            "load_in_4bit": self.load_in_4bit,
            "load_in_8bit": self.load_in_8bit,
        }
        
        if self.load_in_4bit:
            config.update({
                "bnb_4bit_quant_type": self.bnb_4bit_quant_type,
                "bnb_4bit_compute_dtype": self.bnb_4bit_compute_dtype,
                "bnb_4bit_use_double_quant": self.bnb_4bit_use_double_quant,
                "bnb_4bit_quant_storage": self.bnb_4bit_quant_storage,
            })
        
        return config
    
    def get_lora_config(self) -> Dict[str, Any]:
        """Get LoRA configuration for PEFT."""
        config = {
            "r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "target_modules": self.lora_target_modules,
            "lora_dropout": self.lora_dropout,
            "bias": self.lora_bias,
            "task_type": "CAUSAL_LM",
        }
        
        if self.lora_modules_to_save:
            config["modules_to_save"] = self.lora_modules_to_save
        
        return config
    
    def get_generation_config(self) -> Dict[str, Any]:
        """Get generation configuration for inference."""
        return {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "pad_token_id": None,  # Will be set by tokenizer
            "eos_token_id": None,  # Will be set by tokenizer
        }
    
    def get_sft_trainer_config(self) -> Dict[str, Any]:
        """Get configuration for SFTTrainer."""
        return {
            "max_seq_length": self.max_seq_length_sft,
            "packing": self.packing,
            "dataset_text_field": self.dataset_text_field,
        }
    
    def get_model_load_config(self) -> Dict[str, Any]:
        """Get configuration for model loading."""
        return {
            "trust_remote_code": self.trust_remote_code,
            "torch_dtype": self.bnb_4bit_compute_dtype if self.use_qlora else "auto",
            "device_map": "auto" if self.use_qlora else None,
        }
    
    def estimate_memory_usage(self) -> Dict[str, float]:
        """Estimate memory usage for different model sizes."""
        # Model size estimation based on parameter count
        size_estimates = {
            "0.5B": {"full": 2000, "qlora": 800},
            "1.5B": {"full": 6000, "qlora": 1500},
            "3B": {"full": 12000, "qlora": 3000},
            "7B": {"full": 28000, "qlora": 6000},
            "13B": {"full": 52000, "qlora": 10000},
        }
        
        # Extract model size from name
        model_size = "7B"  # Default
        for size in size_estimates.keys():
            if size.lower() in self.model_name.lower():
                model_size = size
                break
        
        estimates = size_estimates.get(model_size, size_estimates["7B"])
        
        if self.use_qlora:
            return {"estimated_memory_mb": estimates["qlora"]}
        else:
            return {"estimated_memory_mb": estimates["full"]}


# Predefined configurations for common LLM models
@dataclass
class Qwen2_7B_InstructConfig(LLMConfig):
    """Configuration for Qwen2-7B-Instruct model."""
    model_name: str = "Qwen/Qwen2-7B-Instruct"
    max_seq_length: int = 2048
    learning_rate: float = 1e-5
    lora_r: int = 16
    lora_alpha: int = 32


@dataclass
class Qwen25_7B_InstructConfig(LLMConfig):
    """Configuration for Qwen2.5-7B-Instruct model."""
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    max_seq_length: int = 2048
    learning_rate: float = 1e-5
    lora_r: int = 16
    lora_alpha: int = 32


@dataclass
class SeaLLM_7B_Config(LLMConfig):
    """Configuration for SeaLLM-v3-7B model."""
    model_name: str = "SeaLLMs/SeaLLMs-v3-7B-Chat"
    max_seq_length: int = 2048
    learning_rate: float = 1e-5
    lora_r: int = 16
    lora_alpha: int = 32


@dataclass
class VinaLLaMA_7B_Config(LLMConfig):
    """Configuration for VinaLLaMA-7B model."""
    model_name: str = "vilm/vinallama-7b-chat"
    max_seq_length: int = 2048
    learning_rate: float = 1e-5
    lora_r: int = 16
    lora_alpha: int = 32


def get_llm_config(model_name: str) -> LLMConfig:
    """
    Get appropriate LLM configuration based on model name.
    
    Args:
        model_name: Model name/identifier
        
    Returns:
        Appropriate LLMConfig instance
    """
    model_configs = {
        "Qwen/Qwen2-7B-Instruct": Qwen2_7B_InstructConfig,
        "Qwen/Qwen2.5-7B-Instruct": Qwen25_7B_InstructConfig,
        "SeaLLMs/SeaLLMs-v3-7B-Chat": SeaLLM_7B_Config,
        "vilm/vinallama-7b-chat": VinaLLaMA_7B_Config,
    }
    
    config_class = model_configs.get(model_name)
    
    if config_class:
        return config_class()
    else:
        # Fallback to base LLM config
        return LLMConfig(model_name=model_name)