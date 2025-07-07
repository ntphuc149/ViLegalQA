"""
Base trainer class for all AQA training methods.
Provides common functionality and abstract methods for specific trainer implementations.
"""

import os
import time
import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union
from pathlib import Path
from datasets import DatasetDict

from ..models.base_model import BaseAQAModel
from ..data.data_processor import AQADataProcessor
from ..utils.logging_utils import get_logger
from ..utils.file_utils import ensure_dir, save_json

logger = get_logger(__name__)


@dataclass
class TrainingOutput:
    """
    Standard output format for training results.
    """
    success: bool = False
    final_loss: Optional[float] = None
    best_metric: Optional[float] = None
    training_time: Optional[float] = None
    total_steps: Optional[int] = None
    model_path: Optional[str] = None
    logs: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class BaseAQATrainer(ABC):
    """
    Abstract base class for all AQA trainers.
    
    Provides common interface and functionality for:
    - Training setup and configuration
    - Training loop management
    - Model saving and checkpointing
    - Training monitoring and logging
    """
    
    def __init__(
        self,
        config: Any,
        model: BaseAQAModel,
        dataset: DatasetDict,
        data_processor: Optional[AQADataProcessor] = None
    ):
        """
        Initialize the base trainer.
        
        Args:
            config: Training configuration object
            model: Model to train
            dataset: Training dataset
            data_processor: Data processor for preprocessing
        """
        self.config = config
        self.model = model
        self.dataset = dataset
        self.data_processor = data_processor
        
        # Training state
        self.trainer = None
        self.training_args = None
        self.is_setup = False
        self.start_time = None
        self.training_logs = []
        
        # Output directory setup
        self.output_dir = Path(config.output_dir)
        ensure_dir(self.output_dir)
        
        logger.info(f"Initialized {self.__class__.__name__}")
    
    @abstractmethod
    def setup_trainer(self) -> None:
        """
        Setup the trainer instance with appropriate configuration.
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def get_training_arguments(self) -> Any:
        """
        Get training arguments for the specific trainer.
        Must be implemented by subclasses.
        
        Returns:
            Training arguments object
        """
        pass
    
    def train(self) -> TrainingOutput:
        """
        Main training loop.
        
        Returns:
            TrainingOutput with training results
        """
        logger.info("Starting training process")
        
        try:
            # Setup trainer if not already done
            if not self.is_setup:
                self.setup_trainer()
            
            # Validate setup
            if not self._validate_training_setup():
                return TrainingOutput(
                    success=False,
                    error_message="Training setup validation failed"
                )
            
            # Start training
            self.start_time = time.time()
            logger.info("🚀 Beginning training...")
            
            # Run training
            training_result = self._run_training()
            
            # Calculate training time
            training_time = time.time() - self.start_time
            
            # Process results
            output = self._process_training_results(training_result, training_time)
            
            # Save model and logs
            if output.success:
                self._save_training_artifacts(output)
            
            logger.info(f"✓ Training completed in {training_time:.2f} seconds")
            return output
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return TrainingOutput(
                success=False,
                error_message=str(e),
                training_time=time.time() - self.start_time if self.start_time else None
            )
    
    def _run_training(self) -> Any:
        """
        Execute the actual training process.
        
        Returns:
            Training result from trainer
        """
        if self.trainer is None:
            raise ValueError("Trainer not setup. Call setup_trainer() first.")
        
        # Setup training callbacks
        self._setup_callbacks()
        
        # Start training
        training_result = self.trainer.train()
        
        return training_result
    
    def _validate_training_setup(self) -> bool:
        """
        Validate that training setup is correct.
        
        Returns:
            True if setup is valid
        """
        try:
            # Check model
            if not self.model.is_loaded:
                logger.error("Model not loaded")
                return False
            
            if not self.model.is_prepared_for_training:
                logger.error("Model not prepared for training")
                return False
            
            # Check dataset
            if not self.dataset or len(self.dataset) == 0:
                logger.error("Empty or invalid dataset")
                return False
            
            # Check required splits
            if "train" not in self.dataset:
                logger.error("No training split found in dataset")
                return False
            
            # Check trainer
            if self.trainer is None:
                logger.error("Trainer not initialized")
                return False
            
            logger.info("✓ Training setup validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Training setup validation failed: {e}")
            return False
    
    def _setup_callbacks(self) -> None:
        """Setup training callbacks for monitoring."""
        if hasattr(self.trainer, 'add_callback'):
            # Add custom callbacks if needed
            pass
    
    def _process_training_results(
        self,
        training_result: Any,
        training_time: float
    ) -> TrainingOutput:
        """
        Process training results into standardized output.
        
        Args:
            training_result: Raw training result
            training_time: Total training time
            
        Returns:
            Processed training output
        """
        try:
            # Extract metrics from training result
            logs = training_result.log_history if hasattr(training_result, 'log_history') else []
            
            # Get final metrics
            final_loss = None
            best_metric = None
            total_steps = None
            
            if logs:
                # Find final training loss
                train_logs = [log for log in logs if 'train_loss' in log]
                if train_logs:
                    final_loss = train_logs[-1]['train_loss']
                
                # Find best evaluation metric
                eval_logs = [log for log in logs if self.config.metric_for_best_model in log]
                if eval_logs:
                    metric_values = [log[self.config.metric_for_best_model] for log in eval_logs]
                    best_metric = max(metric_values) if self.config.greater_is_better else min(metric_values)
                
                # Get total steps
                if train_logs:
                    total_steps = train_logs[-1].get('step', None)
            
            return TrainingOutput(
                success=True,
                final_loss=final_loss,
                best_metric=best_metric,
                training_time=training_time,
                total_steps=total_steps,
                model_path=str(self.output_dir),
                logs=logs
            )
            
        except Exception as e:
            logger.error(f"Failed to process training results: {e}")
            return TrainingOutput(
                success=False,
                training_time=training_time,
                error_message=f"Result processing failed: {e}"
            )
    
    def _save_training_artifacts(self, output: TrainingOutput) -> None:
        """
        Save training artifacts including model, config, and logs.
        
        Args:
            output: Training output to save
        """
        try:
            logger.info("Saving training artifacts...")
            
            # Save model
            self.model.save_model(str(self.output_dir))
            
            # Save training logs
            if output.logs:
                logs_path = self.output_dir / "training_logs.json"
                save_json({"logs": output.logs}, str(logs_path))
            
            # Save training summary
            summary = {
                "training_completed": True,
                "final_loss": output.final_loss,
                "best_metric": output.best_metric,
                "training_time": output.training_time,
                "total_steps": output.total_steps,
                "config": self.config.to_dict()
            }
            
            summary_path = self.output_dir / "training_summary.json"
            save_json(summary, str(summary_path))
            
            logger.info(f"✓ Training artifacts saved to {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save training artifacts: {e}")
    
    def resume_training(self, checkpoint_path: str) -> TrainingOutput:
        """
        Resume training from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            
        Returns:
            TrainingOutput with training results
        """
        logger.info(f"Resuming training from: {checkpoint_path}")
        
        try:
            # Load checkpoint
            self.model.load_checkpoint(checkpoint_path)
            
            # Setup trainer
            self.setup_trainer()
            
            # Resume training
            if hasattr(self.trainer, 'train'):
                # Pass resume_from_checkpoint for HuggingFace trainers
                training_result = self.trainer.train(resume_from_checkpoint=checkpoint_path)
            else:
                training_result = self.trainer.train()
            
            # Process results
            training_time = time.time() - self.start_time if self.start_time else 0
            output = self._process_training_results(training_result, training_time)
            
            return output
            
        except Exception as e:
            logger.error(f"Failed to resume training: {e}")
            return TrainingOutput(
                success=False,
                error_message=f"Resume training failed: {e}"
            )
    
    def get_training_state(self) -> Dict[str, Any]:
        """
        Get current training state information.
        
        Returns:
            Dictionary containing training state
        """
        state = {
            "is_setup": self.is_setup,
            "model_loaded": self.model.is_loaded if self.model else False,
            "model_prepared": self.model.is_prepared_for_training if self.model else False,
            "output_dir": str(self.output_dir),
            "training_method": self.config.training_method,
            "model_type": self.config.model_type
        }
        
        if self.trainer and hasattr(self.trainer, 'state'):
            trainer_state = self.trainer.state
            state.update({
                "current_epoch": getattr(trainer_state, 'epoch', None),
                "global_step": getattr(trainer_state, 'global_step', None),
                "total_steps": getattr(trainer_state, 'max_steps', None)
            })
        
        return state
    
    def stop_training(self) -> None:
        """Stop training gracefully."""
        if self.trainer and hasattr(self.trainer, 'control'):
            self.trainer.control.should_training_stop = True
            logger.info("Training stop requested")
        else:
            logger.warning("No active trainer to stop")
    
    def cleanup(self) -> None:
        """Clean up training resources."""
        if self.trainer:
            del self.trainer
            self.trainer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Training resources cleaned up")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
    
    @property
    def device(self) -> torch.device:
        """Get training device."""
        return self.model.device if self.model else torch.device("cpu")
    
    @property
    def is_distributed(self) -> bool:
        """Check if training is distributed."""
        return torch.distributed.is_initialized() if hasattr(torch, 'distributed') else False