"""
Base model interface for ViBidLQA-AQA system.
Defines common interface for all model types (PLM and LLM).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
import torch
from pathlib import Path

from ..utils.logging_utils import get_logger
from ..utils.model_utils import get_model_size, cleanup_memory

logger = get_logger(__name__)


class BaseAQAModel(ABC):
    """
    Abstract base class for all AQA models.
    Provides common interface for PLMs and LLMs.
    """
    
    def __init__(self, config: Any):
        """
        Initialize base model.
        
        Args:
            config: Model configuration object
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = None
        self.is_loaded = False
        self.is_prepared_for_training = False
        
        logger.info(f"Initializing {self.__class__.__name__} with model: {config.model_name}")
    
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
        inputs: Union[str, Dict[str, torch.Tensor]],
        **generation_kwargs
    ) -> Dict[str, Any]:
        """
        Generate responses from the model.
        Must be implemented by subclasses.
        
        Args:
            inputs: Input text or tokenized inputs
            **generation_kwargs: Generation parameters
            
        Returns:
            Dictionary with generated outputs
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information including parameter counts.
        
        Returns:
            Dictionary with model information
        """
        if not self.is_loaded:
            return {"error": "Model not loaded"}
        
        try:
            total_params, trainable_params = get_model_size(self.model)
            
            info = {
                "model_name": self.config.model_name,
                "model_type": self.config.model_type,
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "trainable_ratio": trainable_params / total_params if total_params > 0 else 0,
                "device": str(self.device) if self.device else None,
                "is_loaded": self.is_loaded,
                "is_prepared_for_training": self.is_prepared_for_training,
            }
            
            # Add model-specific info
            if hasattr(self.model, 'config'):
                model_config = self.model.config
                info.update({
                    "vocab_size": getattr(model_config, 'vocab_size', None),
                    "hidden_size": getattr(model_config, 'hidden_size', None),
                    "num_attention_heads": getattr(model_config, 'num_attention_heads', None),
                    "num_hidden_layers": getattr(model_config, 'num_hidden_layers', None),
                })
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {"error": str(e)}
    
    def save_model(self, output_dir: Union[str, Path]) -> None:
        """
        Save the model and tokenizer.
        
        Args:
            output_dir: Directory to save the model
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Cannot save.")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save model
            if hasattr(self.model, 'save_pretrained'):
                self.model.save_pretrained(output_dir)
                logger.info(f"Model saved to: {output_dir}")
            else:
                # Fallback to torch.save
                torch.save(self.model.state_dict(), output_dir / "pytorch_model.bin")
                logger.info(f"Model state dict saved to: {output_dir}")
            
            # Save tokenizer
            if self.tokenizer and hasattr(self.tokenizer, 'save_pretrained'):
                self.tokenizer.save_pretrained(output_dir)
                logger.info(f"Tokenizer saved to: {output_dir}")
            
            # Save config
            if hasattr(self.config, 'to_dict'):
                import json
                config_path = output_dir / "training_config.json"
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(self.config.to_dict(), f, indent=2, ensure_ascii=False)
                logger.info(f"Training config saved to: {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            if checkpoint_path.is_dir():
                # Load from directory (HuggingFace format)
                self._load_from_directory(checkpoint_path)
            else:
                # Load from file (PyTorch checkpoint)
                self._load_from_file(checkpoint_path)
            
            logger.info(f"Model loaded from checkpoint: {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def _load_from_directory(self, checkpoint_dir: Path) -> None:
        """Load model from directory."""
        if hasattr(self.model, 'from_pretrained'):
            # For transformers models
            self.model = self.model.__class__.from_pretrained(
                checkpoint_dir,
                torch_dtype=getattr(self.config, 'torch_dtype', 'auto'),
                device_map=getattr(self.config, 'device_map', None),
                trust_remote_code=getattr(self.config, 'trust_remote_code', False)
            )
        else:
            raise NotImplementedError("Directory loading not supported for this model type")
    
    def _load_from_file(self, checkpoint_file: Path) -> None:
        """Load model from file."""
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            # Assume checkpoint is the state dict
            self.model.load_state_dict(checkpoint)
    
    def to_device(self, device: Optional[Union[str, torch.device]] = None) -> None:
        """
        Move model to device.
        
        Args:
            device: Target device
        """
        if device is None:
            device = self.config.device if hasattr(self.config, 'device') else 'auto'
        
        if device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            device = torch.device(device)
        
        try:
            if self.model is not None:
                self.model = self.model.to(device)
                self.device = device
                logger.info(f"Model moved to device: {device}")
        except Exception as e:
            logger.error(f"Failed to move model to device {device}: {e}")
            raise
    
    def enable_eval_mode(self) -> None:
        """Set model to evaluation mode."""
        if self.model is not None:
            self.model.eval()
            logger.debug("Model set to evaluation mode")
    
    def enable_train_mode(self) -> None:
        """Set model to training mode."""
        if self.model is not None:
            self.model.train()
            logger.debug("Model set to training mode")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage.
        
        Returns:
            Dictionary with memory usage information
        """
        memory_info = {}
        
        try:
            if torch.cuda.is_available() and self.device and 'cuda' in str(self.device):
                device_idx = self.device.index if self.device.index is not None else 0
                
                memory_info.update({
                    "allocated_mb": torch.cuda.memory_allocated(device_idx) / 1024**2,
                    "cached_mb": torch.cuda.memory_reserved(device_idx) / 1024**2,
                    "max_allocated_mb": torch.cuda.max_memory_allocated(device_idx) / 1024**2,
                    "device": str(self.device)
                })
            else:
                memory_info["device"] = "cpu"
                memory_info["note"] = "CPU memory tracking not available"
            
        except Exception as e:
            memory_info["error"] = str(e)
        
        return memory_info
    
    def cleanup(self) -> None:
        """Clean up model and free memory."""
        try:
            if self.model is not None:
                del self.model
                self.model = None
            
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            
            self.is_loaded = False
            self.is_prepared_for_training = False
            
            cleanup_memory()
            logger.info("Model cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during model cleanup: {e}")
    
    def validate_inputs(self, inputs: Any) -> bool:
        """
        Validate inputs before processing.
        
        Args:
            inputs: Inputs to validate
            
        Returns:
            True if inputs are valid
        """
        if inputs is None:
            logger.error("Inputs cannot be None")
            return False
        
        if isinstance(inputs, str) and len(inputs.strip()) == 0:
            logger.error("Input string cannot be empty")
            return False
        
        if isinstance(inputs, dict):
            required_keys = ["input_ids"]
            missing_keys = [key for key in required_keys if key not in inputs]
            if missing_keys:
                logger.error(f"Missing required keys in inputs: {missing_keys}")
                return False
        
        return True
    
    def estimate_inference_time(self, input_length: int) -> float:
        """
        Estimate inference time based on input length.
        
        Args:
            input_length: Input sequence length
            
        Returns:
            Estimated inference time in seconds
        """
        # Very rough estimation based on model size and input length
        if not self.is_loaded:
            return 0.0
        
        try:
            total_params, _ = get_model_size(self.model)
            
            # Base time estimation (very approximate)
            base_time = 0.001  # 1ms base
            param_factor = total_params / 1e9  # Scale by billions of parameters
            length_factor = input_length / 512  # Scale by input length
            
            estimated_time = base_time * param_factor * length_factor
            
            # Adjust for device
            if self.device and 'cuda' in str(self.device):
                estimated_time *= 0.3  # GPU is faster
            
            return max(0.01, estimated_time)  # Minimum 10ms
            
        except Exception:
            return 0.1  # Default fallback
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return (
            f"{self.__class__.__name__}("
            f"model_name='{self.config.model_name}', "
            f"is_loaded={self.is_loaded}, "
            f"device={self.device})"
        )
    
    def __del__(self):
        """Destructor to clean up resources."""
        try:
            self.cleanup()
        except:
            pass  # Ignore errors during destruction