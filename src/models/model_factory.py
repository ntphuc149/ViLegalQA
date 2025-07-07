"""
Model factory for creating appropriate model instances based on configuration.
Provides unified interface for model creation and management.
"""

from typing import Union, Optional, Dict, Any
from .base_model import BaseAQAModel
from .plm_models import create_plm_model, PLMModel
from .llm_models import create_llm_model, LLMModel
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class ModelFactory:
    """
    Factory class for creating and managing AQA models.
    """
    
    @staticmethod
    def create_model(config: Any) -> BaseAQAModel:
        """
        Create appropriate model based on configuration.
        
        Args:
            config: Model configuration object
            
        Returns:
            Appropriate model instance
        """
        logger.info(f"Creating {config.model_type} model: {config.model_name}")
        
        if config.model_type == "plm":
            model = create_plm_model(config)
        elif config.model_type == "llm":
            model = create_llm_model(config)
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")
        
        logger.info(f"✓ Created {model.__class__.__name__}")
        return model
    
    @staticmethod
    def load_model_from_checkpoint(
        config: Any,
        checkpoint_path: str,
        load_weights: bool = True
    ) -> BaseAQAModel:
        """
        Load model from checkpoint.
        
        Args:
            config: Model configuration
            checkpoint_path: Path to checkpoint
            load_weights: Whether to load model weights
            
        Returns:
            Loaded model instance
        """
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        
        # Create model instance
        model = ModelFactory.create_model(config)
        
        if load_weights:
            # Load checkpoint
            model.load_checkpoint(checkpoint_path)
        else:
            # Just load the base model
            model.load_model()
        
        return model
    
    @staticmethod
    def create_model_for_training(config: Any) -> BaseAQAModel:
        """
        Create and prepare model for training.
        
        Args:
            config: Model configuration
            
        Returns:
            Model ready for training
        """
        logger.info("Creating model for training")
        
        # Create model
        model = ModelFactory.create_model(config)
        
        # Load model
        model.load_model()
        
        # Prepare for training
        model.prepare_for_training()
        
        logger.info("✓ Model ready for training")
        return model
    
    @staticmethod
    def create_model_for_inference(
        config: Any,
        checkpoint_path: Optional[str] = None
    ) -> BaseAQAModel:
        """
        Create model for inference.
        
        Args:
            config: Model configuration
            checkpoint_path: Optional checkpoint path
            
        Returns:
            Model ready for inference
        """
        logger.info("Creating model for inference")
        
        if checkpoint_path:
            # Load from checkpoint
            model = ModelFactory.load_model_from_checkpoint(config, checkpoint_path)
        else:
            # Load base model
            model = ModelFactory.create_model(config)
            model.load_model()
        
        # Set to eval mode
        if model.model is not None:
            model.model.eval()
        
        logger.info("✓ Model ready for inference")
        return model
    
    @staticmethod
    def get_supported_models() -> Dict[str, Dict[str, Any]]:
        """
        Get information about supported models.
        
        Returns:
            Dictionary of supported models and their capabilities
        """
        return {
            "plm_models": {
                "vit5": {
                    "family": "vit5",
                    "variants": ["VietAI/vit5-base", "VietAI/vit5-large"],
                    "training_methods": ["finetune", "instruct"],
                    "description": "Vietnamese T5 model for text-to-text generation"
                },
                "bartpho": {
                    "family": "bartpho", 
                    "variants": [
                        "vinai/bartpho-syllable",
                        "vinai/bartpho-word",
                        "vinai/bartpho-syllable-base",
                        "vinai/bartpho-word-base"
                    ],
                    "training_methods": ["finetune", "instruct"],
                    "description": "Vietnamese BART model for text generation"
                }
            },
            "llm_models": {
                "qwen": {
                    "family": "qwen",
                    "variants": [
                        "Qwen/Qwen2-0.5B", "Qwen/Qwen2-0.5B-Instruct",
                        "Qwen/Qwen2-1.5B", "Qwen/Qwen2-1.5B-Instruct",
                        "Qwen/Qwen2-7B", "Qwen/Qwen2-7B-Instruct",
                        "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct",
                        "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct",
                        "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-3B-Instruct",
                        "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct"
                    ],
                    "training_methods": ["instruct"],
                    "description": "Qwen family models with strong multilingual capabilities"
                },
                "seallm": {
                    "family": "seallm",
                    "variants": [
                        "SeaLLMs/SeaLLMs-v3-1.5B", "SeaLLMs/SeaLLMs-v3-1.5B-Chat",
                        "SeaLLMs/SeaLLMs-v3-7B", "SeaLLMs/SeaLLMs-v3-7B-Chat"
                    ],
                    "training_methods": ["instruct"],
                    "description": "Southeast Asian optimized language models"
                },
                "vinallama": {
                    "family": "vinallama",
                    "variants": [
                        "vilm/vinallama-2.7b", "vilm/vinallama-2.7b-chat",
                        "vilm/vinallama-7b", "vilm/vinallama-7b-chat"
                    ],
                    "training_methods": ["instruct"],
                    "description": "Vietnamese-specialized LLaMA models"
                }
            }
        }
    
    @staticmethod
    def validate_model_config(config: Any) -> bool:
        """
        Validate model configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if configuration is valid
        """
        try:
            # Check required fields
            required_fields = ["model_name", "model_type", "training_method"]
            for field in required_fields:
                if not hasattr(config, field):
                    logger.error(f"Missing required field: {field}")
                    return False
            
            # Validate model type
            if config.model_type not in ["plm", "llm"]:
                logger.error(f"Invalid model_type: {config.model_type}")
                return False
            
            # Validate training method
            if config.training_method not in ["finetune", "instruct"]:
                logger.error(f"Invalid training_method: {config.training_method}")
                return False
            
            # Check compatibility
            if config.model_type == "llm" and config.training_method == "finetune":
                logger.error("LLM models only support instruction tuning")
                return False
            
            # Validate model name
            supported_models = ModelFactory.get_supported_models()
            all_variants = []
            
            for model_family in supported_models["plm_models"].values():
                all_variants.extend(model_family["variants"])
            
            for model_family in supported_models["llm_models"].values():
                all_variants.extend(model_family["variants"])
            
            if config.model_name not in all_variants:
                logger.warning(f"Model {config.model_name} not in officially supported list")
            
            logger.info("✓ Model configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    @staticmethod
    def get_model_recommendations(task_type: str = "legal_qa") -> Dict[str, Any]:
        """
        Get model recommendations for specific tasks.
        
        Args:
            task_type: Type of task (e.g., "legal_qa")
            
        Returns:
            Recommended models and configurations
        """
        if task_type == "legal_qa":
            return {
                "recommended": {
                    "plm_finetune": {
                        "model": "VietAI/vit5-base",
                        "reason": "Good balance of performance and efficiency for Vietnamese"
                    },
                    "plm_instruct": {
                        "model": "VietAI/vit5-base", 
                        "reason": "Supports instruction following with Vietnamese optimization"
                    },
                    "llm_instruct": {
                        "model": "Qwen/Qwen2.5-7B-Instruct",
                        "reason": "State-of-the-art performance with strong Vietnamese capabilities"
                    }
                },
                "budget_friendly": {
                    "plm": "VietAI/vit5-base",
                    "llm": "Qwen/Qwen2.5-1.5B-Instruct"
                },
                "high_performance": {
                    "plm": "VietAI/vit5-large",
                    "llm": "Qwen/Qwen2.5-7B-Instruct"
                }
            }
        
        return {"error": f"No recommendations available for task: {task_type}"}


# Convenience functions
def create_model(config: Any) -> BaseAQAModel:
    """Convenience function for creating models."""
    return ModelFactory.create_model(config)


def create_model_for_training(config: Any) -> BaseAQAModel:
    """Convenience function for creating training-ready models."""
    return ModelFactory.create_model_for_training(config)


def create_model_for_inference(
    config: Any,
    checkpoint_path: Optional[str] = None
) -> BaseAQAModel:
    """Convenience function for creating inference-ready models."""
    return ModelFactory.create_model_for_inference(config, checkpoint_path)