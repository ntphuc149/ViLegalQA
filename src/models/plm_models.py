"""
PLM (Pre-trained Language Model) implementations for encoder-decoder models.
Supports ViT5, BARTPho, T5, and other encoder-decoder architectures.
"""

import torch
from typing import Dict, Any, Optional, List, Union
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    T5Tokenizer,
    T5ForConditionalGeneration
)

from .base_model import BaseAQAModel
from ..utils.logging_utils import get_logger
from ..utils.model_utils import get_optimal_device

logger = get_logger(__name__)


class PLMModel(BaseAQAModel):
    """
    Implementation for encoder-decoder PLMs (ViT5, BARTPho, T5, etc.).
    """
    
    def __init__(self, config: Any):
        """
        Initialize PLM model.
        
        Args:
            config: PLM configuration object
        """
        super().__init__(config)
        self.generation_config = None
        
    def load_model(self) -> None:
        """Load the encoder-decoder model and tokenizer."""
        try:
            logger.info(f"Loading PLM model: {self.config.model_name}")
            
            # Determine tokenizer name
            tokenizer_name = getattr(self.config, 'tokenizer_name', None) or self.config.model_name
            
            # Load tokenizer
            self.tokenizer = self._load_tokenizer(tokenizer_name)
            
            # Load model
            self.model = self._load_model_architecture()
            
            # Setup device
            self.device = get_optimal_device()
            self.to_device(self.device)
            
            # Setup generation config
            self._setup_generation_config()
            
            self.is_loaded = True
            logger.info("✓ PLM model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load PLM model: {e}")
            raise
    
    def _load_tokenizer(self, tokenizer_name: str) -> Any:
        """Load tokenizer with proper configuration."""
        try:
            # Handle special cases for Vietnamese models
            if "vit5" in tokenizer_name.lower():
                # ViT5 uses T5Tokenizer
                tokenizer = T5Tokenizer.from_pretrained(
                    tokenizer_name,
                    cache_dir=self.config.get_model_cache_dir(),
                    use_auth_token=getattr(self.config, 'use_auth_token', True)
                )
            else:
                # Standard AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_name,
                    cache_dir=self.config.get_model_cache_dir(),
                    use_auth_token=getattr(self.config, 'use_auth_token', True),
                    trust_remote_code=getattr(self.config, 'trust_remote_code', False)
                )
            
            # Ensure pad token is set
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    tokenizer.add_special_tokens({'pad_token': '<pad>'})
            
            logger.info(f"✓ Tokenizer loaded: {len(tokenizer)} vocab size")
            return tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
    
    def _load_model_architecture(self) -> Any:
        """Load the model architecture."""
        try:
            model_kwargs = {
                'cache_dir': self.config.get_model_cache_dir(),
                'use_auth_token': getattr(self.config, 'use_auth_token', True),
                'trust_remote_code': getattr(self.config, 'trust_remote_code', False)
            }
            
            # Handle special cases
            if "vit5" in self.config.model_name.lower():
                # ViT5 uses T5ForConditionalGeneration
                model = T5ForConditionalGeneration.from_pretrained(
                    self.config.model_name,
                    **model_kwargs
                )
            else:
                # Standard AutoModel
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.config.model_name,
                    **model_kwargs
                )
            
            # Apply model config updates if any
            config_updates = getattr(self.config, 'get_model_config_updates', lambda: {})()
            if config_updates:
                for key, value in config_updates.items():
                    if hasattr(model.config, key):
                        setattr(model.config, key, value)
                        logger.info(f"Updated model config: {key} = {value}")
            
            # Resize token embeddings if needed
            if len(self.tokenizer) != model.config.vocab_size:
                logger.info(f"Resizing token embeddings: {model.config.vocab_size} -> {len(self.tokenizer)}")
                model.resize_token_embeddings(len(self.tokenizer))
            
            logger.info(f"✓ Model architecture loaded: {model.config.model_type}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model architecture: {e}")
            raise
    
    def _setup_generation_config(self) -> None:
        """Setup generation configuration."""
        try:
            generation_config = self.config.get_generation_config()
            
            # Set tokenizer-specific tokens
            generation_config.update({
                'pad_token_id': self.tokenizer.pad_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
                'decoder_start_token_id': getattr(self.tokenizer, 'bos_token_id', self.tokenizer.eos_token_id)
            })
            
            # Update model's generation config
            for key, value in generation_config.items():
                if value is not None and hasattr(self.model.generation_config, key):
                    setattr(self.model.generation_config, key, value)
            
            self.generation_config = generation_config
            logger.info("✓ Generation config setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup generation config: {e}")
    
    def prepare_for_training(self) -> None:
        """Prepare PLM model for training."""
        try:
            if not self.is_loaded:
                raise ValueError("Model not loaded. Call load_model() first.")
            
            logger.info("Preparing PLM model for training")
            
            # Enable gradient computation
            self.model.train()
            
            # Enable gradient computation for all parameters
            for param in self.model.parameters():
                param.requires_grad = True
            
            # Apply any model-specific training preparations
            self._apply_training_optimizations()
            
            self.is_prepared_for_training = True
            logger.info("✓ PLM model prepared for training")
            
        except Exception as e:
            logger.error(f"Failed to prepare PLM model for training: {e}")
            raise
    
    def _apply_training_optimizations(self) -> None:
        """Apply PLM-specific training optimizations."""
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            logger.info("✓ Gradient checkpointing enabled")
        
        # Set attention implementation if available
        if hasattr(self.model.config, 'attn_implementation'):
            # Use flash attention if available
            self.model.config.attn_implementation = "flash_attention_2"
            logger.info("✓ Flash attention enabled")
    
    def generate(
        self,
        inputs: Union[str, Dict[str, torch.Tensor]],
        **generation_kwargs
    ) -> Dict[str, Any]:
        """
        Generate text using the PLM model.
        
        Args:
            inputs: Input text or tokenized inputs
            **generation_kwargs: Generation parameters
            
        Returns:
            Dictionary with generated outputs
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if not self.validate_inputs(inputs):
            raise ValueError("Invalid inputs provided")
        
        try:
            self.model.eval()
            
            # Tokenize inputs if string provided
            if isinstance(inputs, str):
                tokenized_inputs = self._tokenize_input(inputs)
            else:
                tokenized_inputs = inputs
            
            # Move inputs to device
            tokenized_inputs = {k: v.to(self.device) for k, v in tokenized_inputs.items()}
            
            # Merge generation config
            gen_config = {**self.generation_config, **generation_kwargs}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=tokenized_inputs["input_ids"],
                    attention_mask=tokenized_inputs.get("attention_mask"),
                    **gen_config
                )
            
            # Decode outputs
            generated_texts = self._decode_outputs(outputs, tokenized_inputs["input_ids"])
            
            return {
                "generated_texts": generated_texts,
                "input_length": tokenized_inputs["input_ids"].shape[1],
                "output_length": outputs.shape[1],
                "generation_config": gen_config
            }
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def _tokenize_input(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize input text."""
        try:
            # Handle instruction format if configured
            if getattr(self.config, 'use_instruction_format', False):
                # Assume input is already formatted
                processed_text = text
            else:
                processed_text = text
            
            # Tokenize
            tokenized = self.tokenizer(
                processed_text,
                max_length=self.config.max_source_length,
                padding=getattr(self.config, 'padding', 'max_length'),
                truncation=True,
                return_tensors="pt"
            )
            
            return tokenized
            
        except Exception as e:
            logger.error(f"Tokenization failed: {e}")
            raise
    
    def _decode_outputs(
        self,
        outputs: torch.Tensor,
        input_ids: torch.Tensor
    ) -> List[str]:
        """Decode model outputs to text."""
        try:
            # Skip input tokens for decoder-only models, keep all for encoder-decoder
            decoded_outputs = []
            
            for i, output_sequence in enumerate(outputs):
                # For encoder-decoder models, we decode the full output
                decoded_text = self.tokenizer.decode(
                    output_sequence,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                
                # Post-process the output
                decoded_text = self._post_process_output(decoded_text)
                decoded_outputs.append(decoded_text)
            
            return decoded_outputs
            
        except Exception as e:
            logger.error(f"Decoding failed: {e}")
            raise
    
    def _post_process_output(self, text: str) -> str:
        """Post-process generated text."""
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Remove any residual special tokens
        special_tokens = ["<pad>", "<s>", "</s>", "<unk>"]
        for token in special_tokens:
            text = text.replace(token, "")
        
        # Clean up Vietnamese text specific issues
        text = text.strip()
        
        return text
    
    def encode_text(
        self,
        text: str,
        max_length: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text for training/evaluation.
        
        Args:
            text: Input text
            max_length: Maximum sequence length
            
        Returns:
            Tokenized inputs
        """
        max_length = max_length or self.config.max_source_length
        
        return self.tokenizer(
            text,
            max_length=max_length,
            padding=getattr(self.config, 'padding', 'max_length'),
            truncation=True,
            return_tensors="pt"
        )
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get configuration for training."""
        return self.config.get_training_args_updates()
    
    def supports_gradient_checkpointing(self) -> bool:
        """Check if model supports gradient checkpointing."""
        return hasattr(self.model, 'gradient_checkpointing_enable')
    
    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for memory efficiency."""
        if self.supports_gradient_checkpointing():
            self.model.gradient_checkpointing_enable()
            logger.info("✓ Gradient checkpointing enabled")
        else:
            logger.warning("Gradient checkpointing not supported by this model")
    
    def get_model_specific_info(self) -> Dict[str, Any]:
        """Get PLM-specific model information."""
        info = super().get_model_info()
        
        if self.is_loaded:
            # Add encoder-decoder specific info
            info.update({
                "architecture": "encoder-decoder",
                "supports_generation": True,
                "max_source_length": self.config.max_source_length,
                "max_target_length": self.config.max_target_length,
                "vocab_size": len(self.tokenizer) if self.tokenizer else None,
                "pad_token_id": self.tokenizer.pad_token_id if self.tokenizer else None,
                "eos_token_id": self.tokenizer.eos_token_id if self.tokenizer else None,
            })
            
            # Add generation config
            if self.generation_config:
                info["generation_config"] = self.generation_config.copy()
        
        return info