"""
Fine-tuning trainer for PLM models using traditional seq2seq training.
Implements sequence-to-sequence training for encoder-decoder models.
"""

import torch
from typing import Any, Optional, Dict
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from datasets import DatasetDict

from .base_trainer import BaseAQATrainer, TrainingOutput
from ..models.base_model import BaseAQAModel
from ..data.data_processor import AQADataProcessor
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class FinetuneTrainer(BaseAQATrainer):
    """
    Trainer for traditional fine-tuning of PLM models.
    
    Uses Seq2SeqTrainer for encoder-decoder models like ViT5 and BARTPho
    with standard sequence-to-sequence training objectives.
    """
    
    def __init__(
        self,
        config: Any,
        model: BaseAQAModel,
        dataset: DatasetDict,
        data_processor: Optional[AQADataProcessor] = None
    ):
        """
        Initialize fine-tuning trainer.
        
        Args:
            config: Fine-tuning configuration
            model: PLM model to train
            dataset: Training dataset
            data_processor: Data processor for preprocessing
        """
        super().__init__(config, model, dataset, data_processor)
        
        # Fine-tuning specific attributes
        self.data_collator: Optional[DataCollatorForSeq2Seq] = None
        self.compute_metrics_fn: Optional[Any] = None
        
        # Validate configuration
        if config.model_type != "plm":
            raise ValueError("FinetuneTrainer only supports PLM models")
        
        if config.training_method != "finetune":
            raise ValueError("FinetuneTrainer only supports fine-tuning method")
        
        logger.info("Initialized FinetuneTrainer for PLM fine-tuning")
    
    def setup_trainer(self) -> None:
        """Setup Seq2SeqTrainer for fine-tuning."""
        logger.info("Setting up Seq2SeqTrainer for fine-tuning")
        
        try:
            # Get training arguments
            self.training_args = self.get_training_arguments()
            
            # Setup data collator
            self._setup_data_collator()
            
            # Setup metrics computation
            self._setup_metrics()
            
            # Create trainer
            self.trainer = Seq2SeqTrainer(
                model=self.model.model,
                args=self.training_args,
                train_dataset=self.dataset["train"],
                eval_dataset=self.dataset.get("validation"),
                tokenizer=self.model.tokenizer,
                data_collator=self.data_collator,
                compute_metrics=self.compute_metrics_fn,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
            )
            
            self.is_setup = True
            logger.info("✓ Seq2SeqTrainer setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup trainer: {e}")
            raise
    
    def get_training_arguments(self) -> Seq2SeqTrainingArguments:
        """
        Get Seq2SeqTrainingArguments from config.
        
        Returns:
            Configured training arguments
        """
        logger.info("Creating Seq2SeqTrainingArguments")
        
        # Get base arguments from config
        args_dict = {
            "output_dir": str(self.output_dir),
            "num_train_epochs": self.config.num_train_epochs,
            "per_device_train_batch_size": self.config.per_device_train_batch_size,
            "per_device_eval_batch_size": self.config.per_device_eval_batch_size,
            "learning_rate": self.config.learning_rate,
            "warmup_ratio": getattr(self.config, 'warmup_ratio', 0.05),
            "weight_decay": getattr(self.config, 'weight_decay', 0.01),
            "adam_epsilon": getattr(self.config, 'adam_epsilon', 1e-8),
            "max_grad_norm": getattr(self.config, 'max_grad_norm', 1.0),
            "logging_steps": getattr(self.config, 'logging_steps', 100),
            "eval_steps": getattr(self.config, 'eval_steps', 500),
            "save_steps": getattr(self.config, 'save_steps', 500),
            "save_total_limit": getattr(self.config, 'save_total_limit', 2),
            "evaluation_strategy": getattr(self.config, 'evaluation_strategy', 'steps'),
            "load_best_model_at_end": getattr(self.config, 'load_best_model_at_end', True),
            "metric_for_best_model": getattr(self.config, 'metric_for_best_model', 'eval_rouge1'),
            "greater_is_better": getattr(self.config, 'greater_is_better', True),
            "fp16": getattr(self.config, 'fp16', False),
            "bf16": getattr(self.config, 'bf16', False),
            "dataloader_num_workers": getattr(self.config, 'dataloader_num_workers', 4),
            "dataloader_pin_memory": getattr(self.config, 'dataloader_pin_memory', True),
            "gradient_checkpointing": getattr(self.config, 'gradient_checkpointing', False),
            "gradient_accumulation_steps": getattr(self.config, 'gradient_accumulation_steps', 16),
            "group_by_length": getattr(self.config, 'group_by_length', True),
            "remove_unused_columns": getattr(self.config, 'remove_unused_columns', True),
            "seed": getattr(self.config, 'seed', 42),
            "report_to": "wandb" if getattr(self.config, 'use_wandb', False) else "none"
        }
        
        # Seq2Seq specific arguments
        seq2seq_args = {
            "predict_with_generate": getattr(self.config, 'predict_with_generate', True),
            "generation_max_length": getattr(self.config, 'max_target_length', 256),
            "generation_num_beams": getattr(self.config, 'generation_num_beams', 4)
        }
        
        args_dict.update(seq2seq_args)
        
        # Add any model-specific settings
        if hasattr(self.config, 'model_specific_settings'):
            args_dict.update(self.config.model_specific_settings)
        
        return Seq2SeqTrainingArguments(**args_dict)
    
    def _setup_data_collator(self) -> None:
        """Setup data collator for seq2seq training."""
        self.data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.model.tokenizer,
            model=self.model.model,
            return_tensors="pt",
            padding=True
        )
        
        logger.info("✓ Data collator setup completed")
    
    def _setup_metrics(self) -> None:
        """Setup metrics computation for evaluation."""
        try:
            import evaluate
            import numpy as np
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            
            # Load metrics
            rouge = evaluate.load('rouge')
            meteor = evaluate.load('meteor')
            
            def compute_metrics(eval_preds):
                """Compute metrics for evaluation."""
                predictions, labels = eval_preds
                
                # Decode predictions and labels
                decoded_preds = self.model.tokenizer.batch_decode(
                    predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                
                # Replace -100 in labels as we can't decode them
                labels = np.where(labels != -100, labels, self.model.tokenizer.pad_token_id)
                decoded_labels = self.model.tokenizer.batch_decode(
                    labels, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                
                # Clean up text
                decoded_preds = [pred.strip() for pred in decoded_preds]
                decoded_labels = [label.strip() for label in decoded_labels]
                
                # Compute ROUGE scores
                rouge_scores = rouge.compute(
                    predictions=decoded_preds,
                    references=decoded_labels
                )
                
                # Compute METEOR score
                meteor_score = meteor.compute(
                    predictions=decoded_preds,
                    references=decoded_labels
                )
                
                # Compute BLEU score
                bleu_scores = []
                smoothing = SmoothingFunction().method1
                
                for pred, label in zip(decoded_preds, decoded_labels):
                    try:
                        pred_tokens = pred.split()
                        label_tokens = label.split()
                        bleu = sentence_bleu([label_tokens], pred_tokens, smoothing_function=smoothing)
                        bleu_scores.append(bleu)
                    except:
                        bleu_scores.append(0.0)
                
                return {
                    "rouge1": rouge_scores['rouge1'],
                    "rouge2": rouge_scores['rouge2'],
                    "rougeL": rouge_scores['rougeL'],
                    "meteor": meteor_score['meteor'],
                    "bleu": np.mean(bleu_scores)
                }
            
            self.compute_metrics_fn = compute_metrics
            logger.info("✓ Metrics computation setup completed")
            
        except ImportError as e:
            logger.warning(f"Some metrics libraries not available: {e}")
            self.compute_metrics_fn = None
        except Exception as e:
            logger.error(f"Failed to setup metrics: {e}")
            self.compute_metrics_fn = None
    
    def _validate_training_setup(self) -> bool:
        """
        Validate fine-tuning specific setup.
        
        Returns:
            True if setup is valid
        """
        # Run base validation
        if not super()._validate_training_setup():
            return False
        
        try:
            # Check PLM-specific requirements
            if not hasattr(self.model.model, 'encoder') or not hasattr(self.model.model, 'decoder'):
                logger.error("Model must be encoder-decoder for fine-tuning")
                return False
            
            # Check dataset format
            train_sample = self.dataset["train"][0]
            required_fields = ["input_ids", "attention_mask", "labels"]
            
            missing_fields = [field for field in required_fields if field not in train_sample]
            if missing_fields:
                logger.error(f"Missing required fields in training data: {missing_fields}")
                return False
            
            # Check data collator
            if self.data_collator is None:
                logger.error("Data collator not setup")
                return False
            
            logger.info("✓ Fine-tuning setup validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Fine-tuning validation failed: {e}")
            return False
    
    def get_model_outputs_for_eval(self, eval_dataset=None):
        """
        Get model outputs for manual evaluation.
        
        Args:
            eval_dataset: Dataset to evaluate on (uses validation split if None)
            
        Returns:
            Dictionary with predictions and references
        """
        if not self.is_setup:
            raise ValueError("Trainer not setup. Call setup_trainer() first.")
        
        dataset = eval_dataset or self.dataset.get("validation") or self.dataset.get("test")
        if dataset is None:
            raise ValueError("No evaluation dataset available")
        
        logger.info("Generating predictions for evaluation")
        
        # Set model to eval mode
        self.model.model.eval()
        
        predictions = []
        references = []
        
        with torch.no_grad():
            for i, sample in enumerate(dataset):
                if i % 100 == 0:
                    logger.info(f"Processing sample {i}/{len(dataset)}")
                
                # Prepare input
                input_ids = torch.tensor([sample["input_ids"]]).to(self.device)
                attention_mask = torch.tensor([sample["attention_mask"]]).to(self.device)
                
                # Generate
                generated = self.model.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=getattr(self.config, 'max_target_length', 256),
                    num_beams=getattr(self.config, 'generation_num_beams', 4),
                    early_stopping=True
                )
                
                # Decode
                pred_text = self.model.tokenizer.decode(
                    generated[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                
                # Get reference
                labels = sample["labels"]
                labels = [label if label != -100 else self.model.tokenizer.pad_token_id for label in labels]
                ref_text = self.model.tokenizer.decode(
                    labels, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                
                predictions.append(pred_text.strip())
                references.append(ref_text.strip())
        
        return {
            "predictions": predictions,
            "references": references
        }
    
    def save_predictions(self, output_path: str, eval_dataset=None) -> None:
        """
        Save model predictions to file.
        
        Args:
            output_path: Path to save predictions
            eval_dataset: Dataset to evaluate on
        """
        import pandas as pd
        
        outputs = self.get_model_outputs_for_eval(eval_dataset)
        
        df = pd.DataFrame({
            "predictions": outputs["predictions"],
            "references": outputs["references"]
        })
        
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"Predictions saved to {output_path}")
    
    def get_training_progress(self) -> Dict[str, Any]:
        """
        Get current training progress information.
        
        Returns:
            Dictionary with training progress
        """
        progress = self.get_training_state()
        
        if self.trainer and hasattr(self.trainer, 'state'):
            state = self.trainer.state
            
            progress.update({
                "training_progress": {
                    "current_epoch": getattr(state, 'epoch', 0),
                    "global_step": getattr(state, 'global_step', 0),
                    "max_steps": getattr(state, 'max_steps', 0),
                    "progress_percent": (getattr(state, 'global_step', 0) / max(getattr(state, 'max_steps', 1), 1)) * 100
                }
            })
            
            # Add recent logs if available
            if hasattr(state, 'log_history') and state.log_history:
                recent_logs = state.log_history[-5:]  # Last 5 log entries
                progress["recent_logs"] = recent_logs
        
        return progress