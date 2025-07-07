"""
Large Language Model (LLM) implementations for decoder-only models.
Supports Qwen2, SeaLLM, VinaLLaMa and similar models with QLoRA for instruction tuning.
"""

import torch
from typing import Dict, Any, Union, List, Optional
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

from .base_model import BaseAQAModel, ModelOutput
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class LLMModel(BaseAQAModel):
    """
    Base class for Large Language Models (decoder-only architecture).
    Handles common functionality for Qwen2, SeaLLM, VinaLLaMa with QLoRA support.
    """
    
    def __init__(self, config: Any):
        """
        Initialize LLM model.
        
        Args:
            config: LLM configuration object
        """
        super().__init__(config)
        self.bnb_config: Optional[BitsAndBytesConfig] = None
        self.lora_config: Optional[Any] = None  # Will be LoraConfig from peft
        
    def load_model(self) -> None:
        """Load LLM model and tokenizer with quantization support."""
        try:
            logger.info(f"Loading LLM model: {self.config.model_name}")
            
            # Setup quantization config if needed
            if getattr(self.config, 'use_qlora', False):
                self._setup_quantization_config()
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                use_fast=True,
                trust_remote_code=True
            )
            
            # Load model with quantization
            model_kwargs = {
                "torch_dtype": torch.float16,
                "device_map": "auto" if torch.cuda.is_available() else None,
                "trust_remote_code": True
            }
            
            if self.bnb_config is not None:
                model_kwargs["quantization_config"] = self.bnb_config
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                **model_kwargs
            )
            
            # Setup special tokens
            self._setup_special_tokens()
            
            self._is_loaded = True
            logger.info("✓ LLM model loaded successfully")
            
            # Log model info
            info = self.get_model_info()
            logger.info(f"Model parameters: {info['parameters']['total']:,}")
            if self.bnb_config:
                logger.info(f"Trainable parameters: {info['parameters']['trainable']:,}")
            
        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            raise
    
    def _setup_quantization_config(self) -> None:
        """Setup BitsAndBytesConfig for quantization."""
        if not getattr(self.config, 'use_qlora', False):
            return
        
        logger.info("Setting up quantization config")
        
        try:
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=getattr(self.config, 'load_in_4bit', True),
                load_in_8bit=getattr(self.config, 'load_in_8bit', False),
                bnb_4bit_quant_type=getattr(self.config, 'bnb_4bit_quant_type', 'nf4'),
                bnb_4bit_compute_dtype=getattr(torch, getattr(self.config, 'bnb_4bit_compute_dtype', 'float16')),
                bnb_4bit_use_double_quant=getattr(self.config, 'bnb_4bit_use_double_quant', True)
            )
            
            logger.info("✓ Quantization config created")
            
        except Exception as e:
            logger.error(f"Failed to setup quantization: {e}")
            raise
    
    def _setup_special_tokens(self) -> None:
        """Setup special tokens for LLM tokenizer."""
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                # Add pad token for models that don't have one
                self.tokenizer.add_special_tokens({'pad_token': '<pad>'})
        
        # Set padding side to left for decoder-only models
        self.tokenizer.padding_side = "left"
        
        # Resize model embeddings if new tokens were added
        if len(self.tokenizer) > self.model.config.vocab_size:
            self.model.resize_token_embeddings(len(self.tokenizer))
            logger.info(f"Resized model embeddings to {len(self.tokenizer)}")
    
    def prepare_for_training(self) -> None:
        """Prepare LLM model for training with LoRA."""
        if not self._is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        logger.info("Preparing LLM model for training with LoRA")
        
        try:
            from peft import (
                LoraConfig,
                get_peft_model,
                prepare_model_for_kbit_training,
                TaskType
            )
            
            # Prepare model for k-bit training if quantized
            if self.bnb_config is not None:
                self.model = prepare_model_for_kbit_training(self.model)
                logger.info("✓ Model prepared for k-bit training")
            
            # Setup LoRA config
            self.lora_config = LoraConfig(
                r=getattr(self.config, 'lora_r', 16),
                lora_alpha=getattr(self.config, 'lora_alpha', 32),
                lora_dropout=getattr(self.config, 'lora_dropout', 0.05),
                bias=getattr(self.config, 'lora_bias', 'none'),
                task_type=getattr(TaskType, getattr(self.config, 'lora_task_type', 'CAUSAL_LM')),
                target_modules=getattr(self.config, 'lora_target_modules', [
                    'up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj'
                ])
            )
            
            # Apply LoRA to model
            self.model = get_peft_model(self.model, self.lora_config)
            
            # Enable gradient checkpointing if specified
            if getattr(self.config, 'gradient_checkpointing', False):
                self.model.gradient_checkpointing_enable()
                logger.info("✓ Gradient checkpointing enabled")
            
            # Print trainable parameters
            self.model.print_trainable_parameters()
            
            self._is_prepared_for_training = True
            logger.info("✓ LLM model prepared for training with LoRA")
            
        except ImportError:
            logger.error("PEFT not installed. Install with: pip install peft")
            raise
        except Exception as e:
            logger.error(f"Failed to prepare model for training: {e}")
            raise
    
    def generate(
        self,
        inputs: Union[Dict[str, torch.Tensor], str, List[str]],
        **generation_kwargs
    ) -> ModelOutput:
        """
        Generate responses using LLM model.
        
        Args:
            inputs: Input data (tokenized or raw text)
            **generation_kwargs: Additional generation parameters
            
        Returns:
            ModelOutput with generated text
        """
        if not self._is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if not self.validate_inputs(inputs):
            raise ValueError("Invalid inputs")
        
        # Set model to eval mode for generation
        self.model.eval()
        
        with torch.no_grad():
            # Prepare inputs
            if isinstance(inputs, (str, list)):
                tokenized_inputs = self._tokenize_inputs(inputs)
            else:
                tokenized_inputs = inputs
            
            # Move inputs to device
            tokenized_inputs = {k: v.to(self.device) for k, v in tokenized_inputs.items()}
            
            # Get generation config
            generation_config = self.get_generation_config()
            generation_config.update(generation_kwargs)
            
            # Generate
            try:
                input_length = tokenized_inputs["input_ids"].shape[-1]
                
                generated_tokens = self.model.generate(
                    input_ids=tokenized_inputs["input_ids"],
                    attention_mask=tokenized_inputs.get("attention_mask"),
                    **generation_config
                )
                
                # Extract only the newly generated tokens
                new_tokens = generated_tokens[:, input_length:]
                
                # Decode generated text
                generated_text = self.tokenizer.batch_decode(
                    new_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                
                # Return single string if single input
                if isinstance(inputs, str):
                    generated_text = generated_text[0]
                
                return ModelOutput(
                    generated_text=generated_text,
                    metadata={
                        "num_generated_tokens": new_tokens.shape[-1],
                        "input_length": input_length,
                        "generation_config": generation_config
                    }
                )
                
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                raise
    
    def _tokenize_inputs(self, inputs: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        """
        Tokenize input text(s) for LLM.
        
        Args:
            inputs: Input text or list of texts
            
        Returns:
            Tokenized inputs
        """
        max_length = getattr(self.config, 'max_seq_length', 2048)
        
        if isinstance(inputs, str):
            inputs = [inputs]
        
        tokenized = self.tokenizer(
            inputs,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        return tokenized
    
    def compute_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute training loss for LLM.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels
            
        Returns:
            Loss tensor
        """
        self.model.train()
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs.loss
    
    def _get_model_specific_info(self) -> Dict[str, Any]:
        """Get LLM-specific model information."""
        info = {
            "architecture": "decoder-only",
            "supports_generation": True,
            "max_seq_length": getattr(self.config, 'max_seq_length', 2048),
            "uses_quantization": self.bnb_config is not None,
            "uses_lora": self.lora_config is not None
        }
        
        if self.model is not None:
            info.update({
                "vocab_size": self.model.config.vocab_size,
                "hidden_size": getattr(self.model.config, 'hidden_size', 'unknown'),
                "num_layers": getattr(self.model.config, 'num_hidden_layers', 'unknown'),
                "num_attention_heads": getattr(self.model.config, 'num_attention_heads', 'unknown')
            })
        
        if self.lora_config is not None:
            info["lora_config"] = {
                "r": self.lora_config.r,
                "lora_alpha": self.lora_config.lora_alpha,
                "lora_dropout": self.lora_config.lora_dropout,
                "target_modules": self.lora_config.target_modules
            }
        
        return info


class QwenModel(LLMModel):
    """
    Qwen2/Qwen2.5 model implementation with Vietnamese legal instruction tuning.
    """
    
    def __init__(self, config: Any):
        """Initialize Qwen model."""
        super().__init__(config)
        logger.info("Initialized QwenModel")
    
    def load_model(self) -> None:
        """Load Qwen model with specific configurations."""
        logger.info("Loading Qwen model with optimized settings")
        
        # Call parent load_model
        super().load_model()
        
        # Qwen-specific configurations
        if hasattr(self.model.config, 'rope_scaling'):
            logger.info(f"Qwen RoPE scaling: {self.model.config.rope_scaling}")
    
    def _setup_special_tokens(self) -> None:
        """Setup Qwen-specific special tokens."""
        super()._setup_special_tokens()
        
        # Qwen uses specific chat tokens
        if not hasattr(self.tokenizer, 'im_start_id'):
            # Add Qwen chat tokens if not present
            special_tokens = {
                'additional_special_tokens': ['<|im_start|>', '<|im_end|>']
            }
            self.tokenizer.add_special_tokens(special_tokens)
            self.model.resize_token_embeddings(len(self.tokenizer))
            logger.info("Added Qwen-specific chat tokens")


class SeaLLMModel(LLMModel):
    """
    SeaLLM model implementation optimized for Southeast Asian languages including Vietnamese.
    """
    
    def __init__(self, config: Any):
        """Initialize SeaLLM model."""
        super().__init__(config)
        logger.info("Initialized SeaLLMModel")
    
    def load_model(self) -> None:
        """Load SeaLLM model with specific configurations."""
        logger.info("Loading SeaLLM model with Southeast Asian optimizations")
        
        # Call parent load_model
        super().load_model()
        
        # SeaLLM-specific configurations
        logger.info("SeaLLM loaded with multilingual capabilities")
    
    def _setup_special_tokens(self) -> None:
        """Setup SeaLLM-specific special tokens."""
        super()._setup_special_tokens()
        
        # SeaLLM uses ChatML format similar to Qwen
        if '<|im_start|>' not in self.tokenizer.get_vocab():
            special_tokens = {
                'additional_special_tokens': ['<|im_start|>', '<|im_end|>']
            }
            self.tokenizer.add_special_tokens(special_tokens)
            self.model.resize_token_embeddings(len(self.tokenizer))
            logger.info("Added SeaLLM chat tokens")


class VinaLLaMaModel(LLMModel):
    """
    VinaLLaMa model implementation specifically trained for Vietnamese.
    """
    
    def __init__(self, config: Any):
        """Initialize VinaLLaMa model."""
        super().__init__(config)
        logger.info("Initialized VinaLLaMaModel")
    
    def load_model(self) -> None:
        """Load VinaLLaMa model with specific configurations."""
        logger.info("Loading VinaLLaMa model with Vietnamese optimizations")
        
        # Call parent load_model
        super().load_model()
        
        # VinaLLaMa-specific configurations
        logger.info("VinaLLaMa loaded with Vietnamese language optimizations")
    
    def _setup_special_tokens(self) -> None:
        """Setup VinaLLaMa-specific special tokens."""
        super()._setup_special_tokens()
        
        # VinaLLaMa may use different chat format
        # Check if it's a chat model
        if "chat" in self.config.model_name.lower():
            # Use Vicuna-style format for chat models
            logger.info("Using Vicuna-style chat format for VinaLLaMa-chat")


def create_llm_model(config: Any) -> LLMModel:
    """
    Factory function to create appropriate LLM model based on config.
    
    Args:
        config: Model configuration
        
    Returns:
        Appropriate LLM model instance
    """
    model_name_lower = config.model_name.lower()
    
    if "qwen" in model_name_lower:
        return QwenModel(config)
    elif "seallm" in model_name_lower:
        return SeaLLMModel(config)
    elif "vinallama" in model_name_lower or "vilm" in model_name_lower:
        return VinaLLaMaModel(config)
    else:
        # Default to base LLM model
        logger.warning(f"Unknown LLM model: {config.model_name}, using base LLMModel")
        return LLMModel(config)