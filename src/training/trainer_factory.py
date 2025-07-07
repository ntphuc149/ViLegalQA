"""
Trainer factory for creating appropriate trainer instances based on configuration.
Provides unified interface for trainer creation and management.
"""

from typing import Any, Optional
from datasets import DatasetDict

from .base_trainer import BaseAQATrainer
from .finetune_trainer import FinetuneTrainer
from .instruct_trainer import InstructTrainer
from ..models.base_model import BaseAQAModel
from ..data.data_processor import AQADataProcessor
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class TrainerFactory:
    """
    Factory class for creating appropriate trainers based on configuration.
    """
    
    @staticmethod
    def create_trainer(
        config: Any,
        model: BaseAQAModel,
        dataset: DatasetDict,
        data_processor: Optional[AQADataProcessor] = None
    ) -> BaseAQATrainer:
        """
        Create appropriate trainer based on configuration.
        
        Args:
            config: Training configuration
            model: Model to train
            dataset: Training dataset
            data_processor: Data processor for preprocessing
            
        Returns:
            Appropriate trainer instance
        """
        logger.info(f"Creating trainer for {config.training_method} method")
        
        if config.training_method == "finetune":
            if config.model_type != "plm":
                raise ValueError("Fine-tuning only supports PLM models")
            trainer = FinetuneTrainer(config, model, dataset, data_processor)
            
        elif config.training_method == "instruct":
            trainer = InstructTrainer(config, model, dataset, data_processor)
            
        else:
            raise ValueError(f"Unsupported training method: {config.training_method}")
        
        logger.info(f"✓ Created {trainer.__class__.__name__}")
        return trainer
    
    @staticmethod
    def create_trainer_for_training(
        config: Any,
        model: BaseAQAModel,
        dataset: DatasetDict,
        data_processor: Optional[AQADataProcessor] = None,
        auto_setup: bool = True
    ) -> BaseAQATrainer:
        """
        Create trainer and optionally setup for immediate training.
        
        Args:
            config: Training configuration
            model: Model to train
            dataset: Training dataset
            data_processor: Data processor for preprocessing
            auto_setup: Whether to automatically setup trainer
            
        Returns:
            Trainer ready for training
        """
        logger.info("Creating trainer for immediate training")
        
        # Create trainer
        trainer = TrainerFactory.create_trainer(config, model, dataset, data_processor)
        
        # Setup trainer if requested
        if auto_setup:
            trainer.setup_trainer()
            logger.info("✓ Trainer setup completed")
        
        return trainer
    
    @staticmethod
    def validate_trainer_requirements(config: Any) -> bool:
        """
        Validate that all requirements for trainer creation are met.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if requirements are met
        """
        try:
            # Check basic configuration
            required_fields = ["training_method", "model_type"]
            for field in required_fields:
                if not hasattr(config, field):
                    logger.error(f"Missing required field: {field}")
                    return False
            
            # Validate training method
            if config.training_method not in ["finetune", "instruct"]:
                logger.error(f"Invalid training method: {config.training_method}")
                return False
            
            # Validate model type
            if config.model_type not in ["plm", "llm"]:
                logger.error(f"Invalid model type: {config.model_type}")
                return False
            
            # Check compatibility
            if config.training_method == "finetune" and config.model_type != "plm":
                logger.error("Fine-tuning only supports PLM models")
                return False
            
            # Check required dependencies
            if config.training_method == "instruct" and config.model_type == "llm":
                try:
                    import peft
                    logger.info("✓ PEFT available for LLM instruction tuning")
                except ImportError:
                    logger.error("PEFT required for LLM instruction tuning but not installed")
                    return False
                
                try:
                    import trl
                    logger.info("✓ TRL available for SFTTrainer")
                except ImportError:
                    logger.warning("TRL not available, will use standard Trainer")
            
            logger.info("✓ Trainer requirements validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Trainer requirements validation failed: {e}")
            return False
    
    @staticmethod
    def get_recommended_settings(config: Any) -> dict:
        """
        Get recommended training settings based on model and method.
        
        Args:
            config: Configuration object
            
        Returns:
            Dictionary with recommended settings
        """
        recommendations = {}
        
        if config.training_method == "finetune":
            recommendations.update({
                "gradient_accumulation_steps": 16,
                "per_device_train_batch_size": 2,
                "learning_rate": 3e-5,
                "num_train_epochs": 5,
                "warmup_ratio": 0.05,
                "eval_steps": 500,
                "save_steps": 500
            })
            
        elif config.training_method == "instruct":
            if config.model_type == "plm":
                recommendations.update({
                    "gradient_accumulation_steps": 2,
                    "per_device_train_batch_size": 2,
                    "learning_rate": 3e-5,
                    "num_train_epochs": 3,
                    "warmup_ratio": 0.05
                })
            else:  # LLM
                # Get model size for recommendations
                model_size = getattr(config, 'model_name', '').lower()
                
                if "7b" in model_size:
                    recommendations.update({
                        "gradient_accumulation_steps": 4,
                        "per_device_train_batch_size": 1,
                        "learning_rate": 1e-5,
                        "num_train_epochs": 3,
                        "lora_r": 16,
                        "lora_alpha": 32
                    })
                elif any(size in model_size for size in ["0.5b", "1.5b", "3b"]):
                    recommendations.update({
                        "gradient_accumulation_steps": 2,
                        "per_device_train_batch_size": 2,
                        "learning_rate": 3e-5,
                        "num_train_epochs": 5,
                        "lora_r": 16,
                        "lora_alpha": 32
                    })
                else:
                    # Default for unknown sizes
                    recommendations.update({
                        "gradient_accumulation_steps": 2,
                        "per_device_train_batch_size": 1,
                        "learning_rate": 1e-5,
                        "num_train_epochs": 3
                    })
        
        return recommendations
    
    @staticmethod
    def estimate_training_time(
        config: Any,
        dataset_size: int,
        device_info: Optional[dict] = None
    ) -> dict:
        """
        Estimate training time based on configuration and dataset size.
        
        Args:
            config: Training configuration
            dataset_size: Number of training samples
            device_info: Optional device information
            
        Returns:
            Dictionary with time estimates
        """
        # Base estimates (very rough)
        samples_per_second = {
            "plm_finetune": 2.0,
            "plm_instruct": 1.5,
            "llm_instruct_small": 0.5,  # <3B models
            "llm_instruct_large": 0.2   # 7B+ models
        }
        
        # Determine training type
        if config.training_method == "finetune":
            training_type = "plm_finetune"
        elif config.model_type == "plm":
            training_type = "plm_instruct"
        else:
            model_name = getattr(config, 'model_name', '').lower()
            if "7b" in model_name:
                training_type = "llm_instruct_large"
            else:
                training_type = "llm_instruct_small"
        
        # Calculate estimates
        batch_size = getattr(config, 'per_device_train_batch_size', 1)
        grad_accum = getattr(config, 'gradient_accumulation_steps', 1)
        epochs = getattr(config, 'num_train_epochs', 3)
        
        effective_batch_size = batch_size * grad_accum
        steps_per_epoch = dataset_size // effective_batch_size
        total_steps = steps_per_epoch * epochs
        
        base_rate = samples_per_second[training_type]
        estimated_seconds = total_steps / base_rate
        
        # Adjust for device if provided
        if device_info:
            gpu_name = device_info.get('name', '').lower()
            if 'a100' in gpu_name:
                estimated_seconds *= 0.6  # A100 is faster
            elif 'v100' in gpu_name:
                estimated_seconds *= 0.8  # V100 is moderately faster
            elif 't4' in gpu_name:
                estimated_seconds *= 1.5  # T4 is slower
            elif 'p100' in gpu_name:
                estimated_seconds *= 2.0  # P100 is much slower
        
        # Convert to human-readable format
        hours = estimated_seconds / 3600
        minutes = (estimated_seconds % 3600) / 60
        
        return {
            "estimated_seconds": estimated_seconds,
            "estimated_hours": hours,
            "estimated_minutes": minutes,
            "human_readable": f"{int(hours)}h {int(minutes)}m",
            "total_steps": total_steps,
            "steps_per_epoch": steps_per_epoch,
            "training_type": training_type,
            "note": "Estimates are approximate and may vary significantly based on hardware and actual model complexity"
        }
    
    @staticmethod
    def get_memory_requirements(config: Any) -> dict:
        """
        Estimate memory requirements for training.
        
        Args:
            config: Training configuration
            
        Returns:
            Dictionary with memory estimates
        """
        model_name = getattr(config, 'model_name', '').lower()
        
        # Base memory requirements (GB)
        model_memory = {
            "vit5-base": 1.2,
            "vit5-large": 3.0,
            "bartpho": 0.8,
            "qwen2-0.5b": 2.0,
            "qwen2-1.5b": 4.0,
            "qwen2-7b": 14.0,
            "seallm-1.5b": 4.0,
            "seallm-7b": 14.0,
            "vinallama-2.7b": 6.0,
            "vinallama-7b": 14.0
        }
        
        # Estimate model memory
        base_memory = 8.0  # Default
        for model_key, memory in model_memory.items():
            if model_key.replace('-', '').replace('_', '') in model_name.replace('-', '').replace('_', ''):
                base_memory = memory
                break
        
        # Training overhead
        if config.training_method == "finetune":
            # Full model training
            training_overhead = base_memory * 3  # Gradients + optimizer states
        else:
            # Instruction tuning
            if config.model_type == "llm":
                # LoRA training - much less memory
                if getattr(config, 'use_qlora', False):
                    training_overhead = base_memory * 0.5  # 4-bit + LoRA
                else:
                    training_overhead = base_memory * 1.5  # LoRA only
            else:
                # PLM instruction tuning
                training_overhead = base_memory * 2
        
        # Batch size impact
        batch_size = getattr(config, 'per_device_train_batch_size', 1)
        batch_memory = batch_size * 0.5  # Rough estimate
        
        total_memory = base_memory + training_overhead + batch_memory
        
        return {
            "model_memory_gb": base_memory,
            "training_overhead_gb": training_overhead,
            "batch_memory_gb": batch_memory,
            "total_estimated_gb": total_memory,
            "recommended_gpu_memory_gb": total_memory * 1.2,  # Add safety margin
            "supports_quantization": config.model_type == "llm" and getattr(config, 'use_qlora', False),
            "memory_optimization_tips": [
                "Use gradient checkpointing to reduce memory",
                "Reduce batch size if OOM errors occur",
                "Use LoRA for LLM training to reduce memory",
                "Enable 4-bit quantization for LLMs if supported"
            ]
        }
    
    @staticmethod
    def get_training_tips(config: Any) -> dict:
        """
        Get training tips and best practices.
        
        Args:
            config: Training configuration
            
        Returns:
            Dictionary with training tips
        """
        tips = {
            "general": [
                "Monitor training loss for convergence",
                "Use early stopping to prevent overfitting",
                "Save checkpoints regularly",
                "Validate on held-out data",
                "Log training metrics for analysis"
            ]
        }
        
        if config.training_method == "finetune":
            tips["finetune"] = [
                "Start with lower learning rates (1e-5 to 3e-5)",
                "Use warmup for stable training",
                "Monitor validation metrics closely",
                "Consider freezing encoder layers initially",
                "Use beam search for generation evaluation"
            ]
        
        if config.training_method == "instruct":
            tips["instruct"] = [
                "Use instruction templates consistently",
                "Balance system prompts and user queries",
                "Monitor generation quality manually",
                "Use temperature < 0.5 for consistent outputs",
                "Validate instruction following capabilities"
            ]
            
            if config.model_type == "llm":
                tips["llm"] = [
                    "Use LoRA for memory efficiency",
                    "Start with rank 16-32 for LoRA",
                    "Enable 4-bit quantization if needed",
                    "Use gradient accumulation for effective batch size",
                    "Monitor LoRA adapter weights",
                    "Test different chat templates",
                    "Use packing for sequence efficiency"
                ]
        
        return tips


# Convenience functions
def create_trainer(
    config: Any,
    model: BaseAQAModel,
    dataset: DatasetDict,
    data_processor: Optional[AQADataProcessor] = None
) -> BaseAQATrainer:
    """Convenience function for creating trainers."""
    return TrainerFactory.create_trainer(config, model, dataset, data_processor)


def create_trainer_for_training(
    config: Any,
    model: BaseAQAModel,
    dataset: DatasetDict,
    data_processor: Optional[AQADataProcessor] = None,
    auto_setup: bool = True
) -> BaseAQATrainer:
    """Convenience function for creating training-ready trainers."""
    return TrainerFactory.create_trainer_for_training(config, model, dataset, data_processor, auto_setup)