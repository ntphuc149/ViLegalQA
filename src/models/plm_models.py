"""
Pre-trained Language Model (PLM) implementations for encoder-decoder models.
Supports ViT5, BARTPho and similar seq2seq models for AQA tasks.
"""

import torch
from typing import Dict, Any, Union, List, Optional
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    T5TokenizerFast,
    BartTokenizerFast
)

from .base_model import BaseAQAModel, ModelOutput
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class PLMModel(BaseAQAModel):
    """
    Base class for Pre-trained Language Models (encoder-decoder architecture).
    Handles common functionality for ViT5, BARTPho, and similar models.
    """
    
    def __init__(self, config: Any):
        """
        Initialize PLM model.
        
        Args:
            config: PLM configuration object
        """
        super().__init__(config)
        self.data_collator: Optional[DataCollatorForSeq2Seq] = None
        
    def load_model(self) -> None:
        """Load PLM model and tokenizer."""
        try:
            logger.info(f"Loading PLM model: {self.config.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                use_fast=True
            )
            
            # Load model
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Setup special tokens if needed
            self._setup_special_tokens()
            
            # Move to device
            self.to_device()
            
            # Create data collator
            self.data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                model=self.model,
                return_tensors="pt"
            )
            
            self._is_loaded = True
            logger.info("✓ PLM model loaded successfully")
            
            # Log model info
            info = self.get_model_info()
            logger.info(f"Model parameters: {info['parameters']['total']:,}")
            
        except Exception as e:
            logger.error(f"Failed to load PLM model: {e}")
            raise
    
    def _setup_special_tokens(self) -> None:
        """Setup special tokens for the tokenizer."""
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '<pad>'})
        
        # Resize model embeddings if new tokens were added
        if len(self.tokenizer) > self.model.config.vocab_size:
            self.model.resize_token_embeddings(len(self.tokenizer))
            logger.info(f"Resized model embeddings to {len(self.tokenizer)}")
    
    def prepare_for_training(self) -> None:
        """Prepare PLM model for training."""
        if not self._is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        logger.info("Preparing PLM model for training")
        
        # Enable gradient checkpointing if specified
        if getattr(self.config, 'gradient_checkpointing', False):
            self.model.gradient_checkpointing_enable()
            logger.info("✓ Gradient checkpointing enabled")
        
        # Set model to training mode
        self.model.train()
        
        # Freeze certain layers if specified
        if hasattr(self.config, 'freeze_encoder') and self.config.freeze_encoder:
            self._freeze_encoder()
        
        self._is_prepared_for_training = True
        logger.info("✓ PLM model prepared for training")
    
    def _freeze_encoder(self) -> None:
        """Freeze encoder layers."""
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        
        logger.info("✓ Encoder layers frozen")
    
    def generate(
        self,
        inputs: Union[Dict[str, torch.Tensor], str, List[str]],
        **generation_kwargs
    ) -> ModelOutput:
        """
        Generate responses using PLM model.
        
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
                generated_tokens = self.model.generate(
                    input_ids=tokenized_inputs["input_ids"],
                    attention_mask=tokenized_inputs.get("attention_mask"),
                    **generation_config
                )
                
                # Decode generated text
                generated_text = self.tokenizer.batch_decode(
                    generated_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                
                # Return single string if single input
                if isinstance(inputs, str):
                    generated_text = generated_text[0]
                
                return ModelOutput(
                    generated_text=generated_text,
                    metadata={
                        "num_generated_tokens": generated_tokens.shape[-1],
                        "generation_config": generation_config
                    }
                )
                
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                raise
    
    def _tokenize_inputs(self, inputs: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        """
        Tokenize input text(s).
        
        Args:
            inputs: Input text or list of texts
            
        Returns:
            Tokenized inputs
        """
        max_length = getattr(self.config, 'max_source_length', 1024)
        
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
        Compute training loss.
        
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
        """Get PLM-specific model information."""
        info = {
            "architecture": "encoder-decoder",
            "supports_generation": True,
            "max_source_length": getattr(self.config, 'max_source_length', 1024),
            "max_target_length": getattr(self.config, 'max_target_length', 256)
        }
        
        if self.model is not None:
            info.update({
                "vocab_size": self.model.config.vocab_size,
                "d_model": getattr(self.model.config, 'd_model', 'unknown'),
                "num_layers": getattr(self.model.config, 'num_layers', 'unknown'),
                "num_heads": getattr(self.model.config, 'num_heads', 'unknown')
            })
        
        return info


class ViT5Model(PLMModel):
    """
    ViT5 model implementation for Vietnamese text-to-text generation.
    """
    
    def __init__(self, config: Any):
        """Initialize ViT5 model."""
        super().__init__(config)
        logger.info("Initialized ViT5Model")
    
    def load_model(self) -> None:
        """Load ViT5 model with specific configurations."""
        logger.info("Loading ViT5 model with Vietnamese-specific settings")
        
        # Call parent load_model
        super().load_model()
        
        # ViT5-specific configurations
        if hasattr(self.model.config, 'feed_forward_proj'):
            logger.info(f"ViT5 feed forward projection: {self.model.config.feed_forward_proj}")
    
    def _setup_special_tokens(self) -> None:
        """Setup ViT5-specific special tokens."""
        super()._setup_special_tokens()
        
        # ViT5 may need specific Vietnamese tokens
        special_tokens = []
        
        if not hasattr(self.tokenizer, 'task_prefix'):
            # Add task-specific prefix if needed
            special_tokens.append('<legal_qa>')
        
        if special_tokens:
            self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
            self.model.resize_token_embeddings(len(self.tokenizer))
            logger.info(f"Added ViT5-specific tokens: {special_tokens}")


class BARTPhoBart(PLMModel):
    """
    BARTPho model implementation for Vietnamese text generation.
    """
    
    def __init__(self, config: Any):
        """Initialize BARTPho model."""
        super().__init__(config)
        logger.info("Initialized BARTPhoBart")
    
    def load_model(self) -> None:
        """Load BARTPho model with specific configurations."""
        logger.info("Loading BARTPho model with Vietnamese-specific settings")
        
        # Call parent load_model
        super().load_model()
        
        # BARTPho-specific configurations
        if hasattr(self.model.config, 'activation_function'):
            logger.info(f"BARTPho activation function: {self.model.config.activation_function}")
    
    def _setup_special_tokens(self) -> None:
        """Setup BARTPho-specific special tokens."""
        super()._setup_special_tokens()
        
        # BARTPho tokenizer specifics
        if isinstance(self.tokenizer, BartTokenizerFast):
            # Ensure proper Vietnamese tokenization
            logger.info("Using BARTPho tokenizer with Vietnamese optimizations")


def create_plm_model(config: Any) -> PLMModel:
    """
    Factory function to create appropriate PLM model based on config.
    
    Args:
        config: Model configuration
        
    Returns:
        Appropriate PLM model instance
    """
    model_name_lower = config.model_name.lower()
    
    if "vit5" in model_name_lower:
        return ViT5Model(config)
    elif "bartpho" in model_name_lower:
        return BARTPhoBart(config)
    else:
        # Default to base PLM model
        logger.warning(f"Unknown PLM model: {config.model_name}, using base PLMModel")
        return PLMModel(config)