"""
Instruction fine-tuning trainer for both PLM and LLM models.
Supports instruction-based training with different approaches for encoder-decoder and decoder-only models.
"""

import torch
from typing import Any, Optional, Dict, Union
from transformers import (
    Trainer,
    TrainingArguments,
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from datasets import DatasetDict

from .base_trainer import BaseAQATrainer, TrainingOutput
from ..models.base_model import BaseAQAModel
from ..data.data_processor import AQADataProcessor
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class InstructTrainer(BaseAQATrainer):
    """
    Trainer for instruction fine-tuning of both PLM and LLM models.
    
    Handles:
    - PLM instruction tuning using Seq2SeqTrainer
    - LLM instruction tuning using SFTTrainer/Trainer with LoRA
    """
    
    def __init__(
        self,
        config: Any,
        model: BaseAQAModel,
        dataset: DatasetDict,
        data_processor: Optional[AQADataProcessor] = None
    ):
        """
        Initialize instruction trainer.
        
        Args:
            config: Instruction training configuration
            model: Model to train (PLM or LLM)
            dataset: Training dataset
            data_processor: Data processor for preprocessing
        """
        super().__init__(config, model, dataset, data_processor)
        
        # Instruction training specific attributes
        self.data_collator: Optional[Union[DataCollatorForSeq2Seq, DataCollatorForLanguageModeling]] = None
        self.compute_metrics_fn: Optional[Any] = None
        self.sft_trainer_available = False
        
        # Check if SFTTrainer is available for LLM training
        try:
            from trl import SFTTrainer
            self.sft_trainer_available = True
            logger.info("SFTTrainer available for LLM instruction tuning")
        except ImportError:
            logger.warning("TRL not available. Using standard Trainer for LLM instruction tuning")
        
        # Validate configuration
        if config.training_method != "instruct":
            raise ValueError("InstructTrainer only supports instruction tuning method")
        
        logger.info(f"Initialized InstructTrainer for {config.model_type} instruction tuning")
    
    def setup_trainer(self) -> None:
        """Setup trainer based on model type (PLM vs LLM)."""
        logger.info(f"Setting up instruction trainer for {self.config.model_type}")
        
        try:
            if self.config.model_type == "plm":
                self._setup_plm_instruction_trainer()
            elif self.config.model_type == "llm":
                self._setup_llm_instruction_trainer()
            else:
                raise ValueError(f"Unsupported model type: {self.config.model_type}")
            
            self.is_setup = True
            logger.info("✓ Instruction trainer setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup instruction trainer: {e}")
            raise
    
    def _setup_plm_instruction_trainer(self) -> None:
        """Setup instruction trainer for PLM models using Seq2SeqTrainer."""
        logger.info("Setting up PLM instruction trainer")
        
        # Get training arguments
        self.training_args = self._get_plm_training_arguments()
        
        # Setup data collator for seq2seq
        self.data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.model.tokenizer,
            model=self.model.model,
            return_tensors="pt",
            padding=True
        )
        
        # Setup metrics
        self._setup_metrics()
        
        # Create Seq2SeqTrainer
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
        
        logger.info("✓ PLM instruction trainer setup completed")
    
    def _setup_llm_instruction_trainer(self) -> None:
        """Setup instruction trainer for LLM models."""
        logger.info("Setting up LLM instruction trainer")
        
        if self.sft_trainer_available:
            self._setup_sft_trainer()
        else:
            self._setup_standard_trainer_for_llm()
    
    def _setup_sft_trainer(self) -> None:
        """Setup SFTTrainer for LLM instruction tuning."""
        try:
            from trl import SFTTrainer
            
            # Get training arguments
            self.training_args = self._get_llm_training_arguments()
            
            # Create SFTTrainer
            trainer_kwargs = {
                "model": self.model.model,
                "args": self.training_args,
                "train_dataset": self.dataset["train"],
                "eval_dataset": self.dataset.get("validation"),
                "tokenizer": self.model.tokenizer,
                "max_seq_length": getattr(self.config, 'max_seq_length', 2048),
                "dataset_text_field": getattr(self.config, 'dataset_text_field', 'instruction'),
                "packing": getattr(self.config, 'packing', True)
            }
            
            # Add PEFT config if available
            if hasattr(self.model, 'lora_config') and self.model.lora_config:
                trainer_kwargs["peft_config"] = self.model.lora_config
            
            self.trainer = SFTTrainer(**trainer_kwargs)
            
            logger.info("✓ SFTTrainer setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup SFTTrainer: {e}")
            raise
    
    def _setup_standard_trainer_for_llm(self) -> None:
        """Setup standard Trainer for LLM instruction tuning when SFTTrainer not available."""
        logger.info("Setting up standard Trainer for LLM")
        
        # Get training arguments
        self.training_args = self._get_llm_training_arguments()
        
        # Setup data collator for causal LM
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.model.tokenizer,
            mlm=False,  # Causal LM, not masked LM
            return_tensors="pt"
        )
        
        # Create standard Trainer
        self.trainer = Trainer(
            model=self.model.model,
            args=self.training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset.get("validation"),
            tokenizer=self.model.tokenizer,
            data_collator=self.data_collator
        )
        
        logger.info("✓ Standard Trainer setup completed")
    
    def get_training_arguments(self) -> Union[Seq2SeqTrainingArguments, TrainingArguments]:
        """
        Get training arguments based on model type.
        
        Returns:
            Appropriate training arguments
        """
        if self.config.model_type == "plm":
            return self._get_plm_training_arguments()
        else:
            return self._get_llm_training_arguments()
    
    def _get_plm_training_arguments(self) -> Seq2SeqTrainingArguments:
        """Get Seq2SeqTrainingArguments for PLM instruction tuning."""
        logger.info("Creating Seq2SeqTrainingArguments for PLM instruction tuning")
        
        args_dict = {
            "output_dir": str(self.output_dir),
            "num_train_epochs": self.config.num_train_epochs,
            "per_device_train_batch_size": self.config.per_device_train_batch_size,
            "per_device_eval_batch_size": self.config.per_device_eval_batch_size,
            "learning_rate": self.config.learning_rate,
            "warmup_ratio": getattr(self.config, 'warmup_ratio', 0.05),
            "weight_decay": getattr(self.config, 'weight_decay', 0.01),
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
            "gradient_accumulation_steps": getattr(self.config, 'gradient_accumulation_steps', 2),
            "seed": getattr(self.config, 'seed', 42),
            "report_to": "wandb" if getattr(self.config, 'use_wandb', False) else "none",
            
            # Seq2Seq specific
            "predict_with_generate": True,
            "generation_max_length": getattr(self.config, 'max_target_length', 256),
            "generation_num_beams": 4
        }
        
        return Seq2SeqTrainingArguments(**args_dict)
    
    def _get_llm_training_arguments(self) -> TrainingArguments:
        """Get TrainingArguments for LLM instruction tuning."""
        logger.info("Creating TrainingArguments for LLM instruction tuning")
        
        args_dict = {
            "output_dir": str(self.output_dir),
            "num_train_epochs": self.config.num_train_epochs,
            "per_device_train_batch_size": self.config.per_device_train_batch_size,
            "per_device_eval_batch_size": self.config.per_device_eval_batch_size,
            "learning_rate": self.config.learning_rate,
            "warmup_ratio": getattr(self.config, 'warmup_ratio', 0.05),
            "weight_decay": getattr(self.config, 'weight_decay', 0.01),
            "logging_steps": getattr(self.config, 'logging_steps', 100),
            "eval_steps": getattr(self.config, 'eval_steps', 500),
            "save_steps": getattr(self.config, 'save_steps', 500),
            "save_total_limit": getattr(self.config, 'save_total_limit', 2),
            "evaluation_strategy": getattr(self.config, 'evaluation_strategy', 'steps'),
            "load_best_model_at_end": getattr(self.config, 'load_best_model_at_end', True),
            "gradient_accumulation_steps": getattr(self.config, 'gradient_accumulation_steps', 2),
            "fp16": getattr(self.config, 'fp16', False),
            "bf16": getattr(self.config, 'bf16', False),
            "seed": getattr(self.config, 'seed', 42),
            "report_to": "wandb" if getattr(self.config, 'use_wandb', False) else "none"
        }
        
        # LLM specific settings
        if hasattr(self.config, 'optim'):
            args_dict["optim"] = self.config.optim
        
        # Add group_by_length for efficiency
        args_dict["group_by_length"] = True
        
        return TrainingArguments(**args_dict)
    
    def _setup_metrics(self) -> None:
        """Setup metrics computation for instruction tuning evaluation."""
        if self.config.model_type != "plm":
            # Skip metrics for LLMs as they use different evaluation approach
            return
        
        try:
            import evaluate
            import numpy as np
            
            # Load metrics
            rouge = evaluate.load('rouge')
            
            def compute_metrics(eval_preds):
                """Compute metrics for PLM instruction tuning."""
                predictions, labels = eval_preds
                
                # Decode predictions and labels
                decoded_preds = self.model.tokenizer.batch_decode(
                    predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                
                # Replace -100 in labels
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
                
                return {
                    "rouge1": rouge_scores['rouge1'],
                    "rouge2": rouge_scores['rouge2'],
                    "rougeL": rouge_scores['rougeL']
                }
            
            self.compute_metrics_fn = compute_metrics
            logger.info("✓ Metrics computation setup completed")
            
        except ImportError as e:
            logger.warning(f"Metrics libraries not available: {e}")
            self.compute_metrics_fn = None
        except Exception as e:
            logger.error(f"Failed to setup metrics: {e}")
            self.compute_metrics_fn = None
    
    def _validate_training_setup(self) -> bool:
        """
        Validate instruction training specific setup.
        
        Returns:
            True if setup is valid
        """
        # Run base validation
        if not super()._validate_training_setup():
            return False
        
        try:
            # Check model type specific requirements
            if self.config.model_type == "plm":
                return self._validate_plm_setup()
            elif self.config.model_type == "llm":
                return self._validate_llm_setup()
            else:
                logger.error(f"Unsupported model type: {self.config.model_type}")
                return False
            
        except Exception as e:
            logger.error(f"Instruction training validation failed: {e}")
            return False
    
    def _validate_plm_setup(self) -> bool:
        """Validate PLM instruction training setup."""
        # Check encoder-decoder architecture
        if not (hasattr(self.model.model, 'encoder') and hasattr(self.model.model, 'decoder')):
            logger.error("PLM model must be encoder-decoder for instruction tuning")
            return False
        
        # Check dataset format
        train_sample = self.dataset["train"][0]
        required_fields = ["input_ids", "attention_mask", "labels"]
        
        missing_fields = [field for field in required_fields if field not in train_sample]
        if missing_fields:
            logger.error(f"Missing required fields in PLM training data: {missing_fields}")
            return False
        
        logger.info("✓ PLM instruction training setup validation passed")
        return True
    
    def _validate_llm_setup(self) -> bool:
        """Validate LLM instruction training setup."""
        # Check decoder-only architecture
        if hasattr(self.model.model, 'encoder'):
            logger.error("LLM model should be decoder-only for instruction tuning")
            return False
        
        # Check LoRA preparation
        if not hasattr(self.model.model, 'peft_config') and not hasattr(self.model, 'lora_config'):
            logger.warning("LLM model may not be prepared with LoRA adapters")
        
        # Check dataset format
        train_sample = self.dataset["train"][0]
        expected_field = getattr(self.config, 'dataset_text_field', 'instruction')
        
        if expected_field not in train_sample:
            logger.error(f"Missing expected field '{expected_field}' in LLM training data")
            return False
        
        logger.info("✓ LLM instruction training setup validation passed")
        return True
    
    def generate_sample_outputs(self, num_samples: int = 5) -> Dict[str, Any]:
        """
        Generate sample outputs for qualitative evaluation.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Dictionary with sample inputs and outputs
        """
        if not self.is_setup:
            raise ValueError("Trainer not setup. Call setup_trainer() first.")
        
        # Get validation or test dataset
        eval_dataset = self.dataset.get("validation") or self.dataset.get("test")
        if eval_dataset is None:
            raise ValueError("No evaluation dataset available")
        
        # Select random samples
        import random
        indices = random.sample(range(len(eval_dataset)), min(num_samples, len(eval_dataset)))
        
        samples = []
        self.model.model.eval()
        
        with torch.no_grad():
            for idx in indices:
                sample = eval_dataset[idx]
                
                if self.config.model_type == "plm":
                    # PLM generation
                    input_ids = torch.tensor([sample["input_ids"]]).to(self.device)
                    attention_mask = torch.tensor([sample["attention_mask"]]).to(self.device)
                    
                    generated = self.model.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=getattr(self.config, 'max_target_length', 256),
                        num_beams=4,
                        early_stopping=True
                    )
                    
                    generated_text = self.model.tokenizer.decode(
                        generated[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
                    )
                    
                    # Get reference
                    labels = sample["labels"]
                    labels = [l if l != -100 else self.model.tokenizer.pad_token_id for l in labels]
                    reference_text = self.model.tokenizer.decode(
                        labels, skip_special_tokens=True, clean_up_tokenization_spaces=True
                    )
                    
                    # Get input text
                    input_text = self.model.tokenizer.decode(
                        sample["input_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=True
                    )
                    
                else:
                    # LLM generation
                    instruction_field = getattr(self.config, 'dataset_text_field', 'instruction')
                    instruction = sample[instruction_field]
                    
                    # Tokenize instruction
                    inputs = self.model.tokenizer(
                        instruction,
                        return_tensors="pt",
                        max_length=getattr(self.config, 'max_seq_length', 2048),
                        truncation=True
                    ).to(self.device)
                    
                    # Generate
                    generated = self.model.model.generate(
                        **inputs,
                        max_new_tokens=getattr(self.config, 'max_new_tokens', 256),
                        do_sample=True,
                        temperature=0.1,
                        pad_token_id=self.model.tokenizer.pad_token_id
                    )
                    
                    # Extract only new tokens
                    input_length = inputs["input_ids"].shape[-1]
                    new_tokens = generated[:, input_length:]
                    
                    generated_text = self.model.tokenizer.decode(
                        new_tokens[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
                    )
                    
                    input_text = instruction
                    reference_text = sample.get("output", "N/A")
                
                samples.append({
                    "input": input_text.strip(),
                    "generated": generated_text.strip(),
                    "reference": reference_text.strip()
                })
        
        return {"samples": samples}
    
    def get_training_progress(self) -> Dict[str, Any]:
        """
        Get current training progress for instruction tuning.
        
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
                },
                "model_type": self.config.model_type,
                "trainer_type": "SFTTrainer" if self.sft_trainer_available and self.config.model_type == "llm" else "Standard"
            })
            
            # Add LoRA info for LLMs
            if self.config.model_type == "llm" and hasattr(self.model, 'lora_config'):
                progress["lora_config"] = {
                    "r": self.model.lora_config.r,
                    "lora_alpha": self.model.lora_config.lora_alpha,
                    "target_modules": self.model.lora_config.target_modules
                }
        
        return progress