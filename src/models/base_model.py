"""
Base model interface for all AQA models.
Provides common functionality and abstract methods for model implementations.
"""

import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..utils.logging_utils import get_logger
from ..utils.model_utils import count_parameters

logger = get_logger(__name__)


@dataclass
class ModelOutput:
    """
    Standard output format for all AQA models.
    """
    generated_text: Optional[Union[str, List[str]]] = None
    logits: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    attention_weights: Optional[torch.Tensor] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseAQAModel(ABC):
    """
    Abstract base class for all AQA models.
    
    Provides common interface and functionality for:
    - Model loading and saving
    - Training preparation
    - Generation/inference
    - Model information and statistics
    """
    
    def __init__(self, config: Any):
        """
        Initialize the base model.
        
        Args:
            config: Configuration object containing model parameters
        """
        self.config = config
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._is_loaded = False
        self._is_prepared_for_training = False
        
        logger.info(f"Initialized {self.__class__.__name__} with config")
    
    @abstractmethod
    def load_model(self) -> None:
        """
        Load the model and tokenizer.
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def prepare_for_training(self) -> None:
        """
        Prepare model for training (LoRA, quantization, etc.).
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def generate(
        self,
        inputs: Union[Dict[str, torch.Tensor], str, List[str]],
        **generation_kwargs
    ) -> ModelOutput:
        """
        Generate responses for given inputs.
        Must be implemented by subclasses.
        
        Args:
            inputs: Input data (can be tokenized or raw text)
            **generation_kwargs: Additional generation parameters
            
        Returns:
            ModelOutput containing generated text and metadata
        """
        pass
    
    def save_model(self, output_dir: str) -> None:
        """
        Save the model and tokenizer.
        
        Args:
            output_dir: Directory to save model
        """
        if not self._is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(output_path)
            logger.info(f"Model saved to {output_path}")
        else:
            # For models with LoRA or custom wrappers
            torch.save(self.model.state_dict(), output_path / "pytorch_model.bin")
            logger.info(f"Model state dict saved to {output_path}")
        
        # Save tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_path)
            logger.info(f"Tokenizer saved to {output_path}")
        
        # Save config
        self.config.save_config(str(output_path / "config.json"))
        logger.info(f"Config saved to {output_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
        
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        # Load based on model type and training method
        if self.config.model_type == "llm" and self.config.training_method == "instruct":
            self._load_llm_checkpoint(checkpoint_path)
        else:
            self._load_plm_checkpoint(checkpoint_path)
        
        self._is_loaded = True
        logger.info("✓ Checkpoint loaded successfully")
    
    def _load_llm_checkpoint(self, checkpoint_path: Path) -> None:
        """Load LLM checkpoint with LoRA adapters."""
        try:
            from peft import AutoPeftModelForCausalLM
            from transformers import AutoTokenizer
            
            # Load model with PEFT
            self.model = AutoPeftModelForCausalLM.from_pretrained(
                checkpoint_path,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
            
        except ImportError:
            logger.error("PEFT not installed. Install with: pip install peft")
            raise
        except Exception as e:
            logger.error(f"Failed to load LLM checkpoint: {e}")
            raise
    
    def _load_plm_checkpoint(self, checkpoint_path: Path) -> None:
        """Load PLM checkpoint."""
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            
            # Load model
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.float16 if self.config.fp16 else torch.float32
            )
            
            # Load tokenizer  
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
            
        except Exception as e:
            logger.error(f"Failed to load PLM checkpoint: {e}")
            raise
    
    def to_device(self, device: Optional[Union[str, torch.device]] = None) -> None:
        """
        Move model to specified device.
        
        Args:
            device: Target device (defaults to self.device)
        """
        if device is not None:
            self.device = torch.device(device)
        
        if self.model is not None:
            self.model = self.model.to(self.device)
            logger.info(f"Model moved to {self.device}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Returns:
            Dictionary containing model metadata
        """
        if not self._is_loaded:
            return {"status": "not_loaded"}
        
        info = {
            "model_name": self.config.model_name,
            "model_type": self.config.model_type,
            "training_method": self.config.training_method,
            "device": str(self.device),
            "is_prepared_for_training": self._is_prepared_for_training,
            "parameters": {}
        }
        
        # Parameter counts
        if self.model is not None:
            info["parameters"] = {
                "total": count_parameters(self.model, trainable_only=False),
                "trainable": count_parameters(self.model, trainable_only=True)
            }
            
            # Memory usage estimation
            total_params = info["parameters"]["total"]
            memory_mb = (total_params * 4) / (1024 * 1024)  # Assuming fp32
            info["estimated_memory_mb"] = memory_mb
        
        # Model-specific information
        if hasattr(self, '_get_model_specific_info'):
            info.update(self._get_model_specific_info())
        
        return info
    
    def validate_inputs(self, inputs: Any) -> bool:
        """
        Validate inputs before processing.
        
        Args:
            inputs: Input data to validate
            
        Returns:
            True if inputs are valid
        """
        if inputs is None:
            logger.error("Inputs cannot be None")
            return False
        
        if isinstance(inputs, str) and len(inputs.strip()) == 0:
            logger.error("Input text cannot be empty")
            return False
        
        if isinstance(inputs, list) and len(inputs) == 0:
            logger.error("Input list cannot be empty")
            return False
        
        if isinstance(inputs, dict):
            required_keys = ["input_ids"] if self.config.model_type == "llm" else ["input_ids", "attention_mask"]
            missing_keys = [key for key in required_keys if key not in inputs]
            if missing_keys:
                logger.error(f"Missing required keys in inputs: {missing_keys}")
                return False
        
        return True
    
    def get_generation_config(self) -> Dict[str, Any]:
        """
        Get generation configuration based on model config.
        
        Returns:
            Generation parameters dictionary
        """
        generation_config = {
            "max_new_tokens": getattr(self.config, 'max_new_tokens', 256),
            "do_sample": getattr(self.config, 'do_sample', True),
            "temperature": getattr(self.config, 'temperature', 0.1),
            "top_p": getattr(self.config, 'top_p', 0.75),
            "top_k": getattr(self.config, 'top_k', 50),
            "num_beams": getattr(self.config, 'num_beams', 4),
            "early_stopping": getattr(self.config, 'early_stopping', True),
            "pad_token_id": self.tokenizer.pad_token_id if self.tokenizer else None,
            "eos_token_id": self.tokenizer.eos_token_id if self.tokenizer else None
        }
        
        # Model-specific adjustments
        if self.config.model_type == "plm":
            # PLMs typically use beam search
            generation_config["do_sample"] = False
            generation_config["num_beams"] = getattr(self.config, 'generation_num_beams', 4)
        else:
            # LLMs typically use sampling
            generation_config["do_sample"] = True
            generation_config["num_beams"] = 1
        
        return generation_config
    
    def cleanup(self) -> None:
        """
        Clean up model resources.
        """
        if self.model is not None:
            del self.model
            self.model = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Model resources cleaned up")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded
    
    @property
    def is_prepared_for_training(self) -> bool:
        """Check if model is prepared for training."""
        return self._is_prepared_for_training
    
    @property
    def model_family(self) -> str:
        """Get model family name."""
        model_name_lower = self.config.model_name.lower()
        
        if "vit5" in model_name_lower:
            return "vit5"
        elif "bartpho" in model_name_lower:
            return "bartpho"
        elif "qwen" in model_name_lower:
            return "qwen"
        elif "seallm" in model_name_lower:
            return "seallm"
        elif "vinallama" in model_name_lower or "vilm" in model_name_lower:
            return "vinallama"
        else:
            return "unknown"