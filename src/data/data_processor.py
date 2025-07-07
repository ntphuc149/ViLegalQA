"""
Data processor for AQA tasks with support for both traditional fine-tuning
and instruction-based training for PLMs and LLMs.
"""

import re
from typing import Dict, List, Any, Optional, Callable
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizer

from .instruction_templates import InstructionTemplateManager
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class AQADataProcessor:
    """
    Data processor for Abstractive Question Answering tasks.
    
    Handles:
    - Data preprocessing and cleaning
    - Instruction formatting for different training methods
    - Tokenization and encoding
    - Data validation and quality checks
    """
    
    def __init__(
        self,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        instruction_manager: Optional[InstructionTemplateManager] = None
    ):
        """
        Initialize the data processor.
        
        Args:
            tokenizer: Tokenizer for encoding text
            instruction_manager: Manager for instruction templates
        """
        self.tokenizer = tokenizer
        self.instruction_manager = instruction_manager or InstructionTemplateManager()
        
        logger.info("Initialized AQADataProcessor")
    
    def process_dataset(
        self,
        dataset: DatasetDict,
        config: Any,  # Config object
        stage: str = "train"
    ) -> DatasetDict:
        """
        Process dataset for training or inference.
        
        Args:
            dataset: Raw dataset to process
            config: Configuration object with processing parameters
            stage: Processing stage ('train', 'inference')
            
        Returns:
            Processed dataset ready for training/inference
        """
        logger.info(f"Processing dataset for {config.training_method} {config.model_type} model")
        
        processed_dataset = {}
        
        for split_name, split_data in dataset.items():
            logger.info(f"Processing {split_name} split ({len(split_data)} samples)")
            
            if config.training_method == "finetune":
                processed_split = self._process_for_finetune(split_data, config, stage)
            else:  # instruction
                processed_split = self._process_for_instruction(split_data, config, stage)
            
            processed_dataset[split_name] = processed_split
            logger.info(f"✓ Processed {split_name}: {len(processed_split)} samples")
        
        return DatasetDict(processed_dataset)
    
    def _process_for_finetune(
        self,
        dataset: Dataset,
        config: Any,
        stage: str
    ) -> Dataset:
        """Process dataset for traditional fine-tuning (PLMs only)."""
        
        def process_example(example):
            # Clean text
            context = self._clean_text(example["context"])
            question = self._clean_text(example["question"])
            answer = self._clean_text(example["abstractive_answer"]) if stage == "train" else ""
            
            # Format using instruction template
            formatted = self.instruction_manager.format_for_plm(
                context=context,
                question=question,
                answer=answer,
                template_name=getattr(config, 'instruction_template_name', 'vietnamese_legal')
            )
            
            result = {
                "input_text": formatted["input_text"],
                "target_text": formatted["target_text"]
            }
            
            # Add tokenization if tokenizer is available
            if self.tokenizer is not None:
                # Tokenize input
                input_encoding = self.tokenizer(
                    formatted["input_text"],
                    max_length=getattr(config, 'max_source_length', 1024),
                    truncation=True,
                    padding="max_length" if getattr(config, 'padding', 'max_length') == 'max_length' else False
                )
                
                # Tokenize target
                if formatted["target_text"]:
                    target_encoding = self.tokenizer(
                        formatted["target_text"],
                        max_length=getattr(config, 'max_target_length', 256),
                        truncation=True,
                        padding="max_length" if getattr(config, 'padding', 'max_length') == 'max_length' else False
                    )
                    result["labels"] = target_encoding["input_ids"]
                
                result.update(input_encoding)
            
            return result
        
        return dataset.map(
            process_example,
            remove_columns=dataset.column_names,
            desc=f"Processing for fine-tuning"
        )
    
    def _process_for_instruction(
        self,
        dataset: Dataset,
        config: Any,
        stage: str
    ) -> Dataset:
        """Process dataset for instruction-based training."""
        
        def process_example(example):
            # Clean text
            context = self._clean_text(example["context"])
            question = self._clean_text(example["question"])
            answer = self._clean_text(example["abstractive_answer"]) if stage == "train" else ""
            
            # Format instruction based on model type
            if config.model_type == "plm":
                # PLM with instruction format
                formatted = self.instruction_manager.format_for_plm(
                    context=context,
                    question=question,
                    answer=answer,
                    template_name=getattr(config, 'instruction_template_name', 'vietnamese_legal')
                )
                
                result = {
                    "input_text": formatted["input_text"],
                    "target_text": formatted["target_text"]
                }
                
                # Tokenize for PLM
                if self.tokenizer is not None:
                    input_encoding = self.tokenizer(
                        formatted["input_text"],
                        max_length=getattr(config, 'max_source_length', 1024),
                        truncation=True,
                        padding="max_length" if getattr(config, 'padding', 'max_length') == 'max_length' else False
                    )
                    
                    if formatted["target_text"]:
                        target_encoding = self.tokenizer(
                            formatted["target_text"],
                            max_length=getattr(config, 'max_target_length', 256),
                            truncation=True,
                            padding="max_length" if getattr(config, 'padding', 'max_length') == 'max_length' else False
                        )
                        result["labels"] = target_encoding["input_ids"]
                    
                    result.update(input_encoding)
            
            else:  # LLM
                # Format for LLM instruction tuning
                instruction_text = self.instruction_manager.format_for_llm(
                    context=context,
                    question=question,
                    answer=answer,
                    chat_template_name=getattr(config, 'chat_template', 'chatml'),
                    instruction_template_name=getattr(config, 'instruction_template_name', 'vietnamese_legal'),
                    include_assistant_start=stage == "train"
                )
                
                result = {
                    "instruction": instruction_text,
                    "input": "",  # For compatibility with SFT trainers
                    "output": answer if stage == "train" else ""
                }
                
                # For SFT trainers that expect specific field names
                if hasattr(config, 'dataset_text_field'):
                    field_name = config.dataset_text_field
                    result[field_name] = instruction_text
            
            # Add original fields for reference
            result.update({
                "original_context": example["context"],
                "original_question": example["question"],
                "original_answer": example["abstractive_answer"]
            })
            
            return result
        
        return dataset.map(
            process_example,
            remove_columns=dataset.column_names,
            desc=f"Processing for instruction tuning"
        )
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special tokens that might interfere with training
        special_tokens = ['<unk>', '<pad>', '</s>', '<s>']
        for token in special_tokens:
            text = text.replace(token, '')
        
        # Remove or normalize problematic characters
        text = re.sub(r'[^\w\s,.;:!?()\-\u00C0-\u1EF9]', '', text)  # Keep Vietnamese characters
        
        return text
    
    def validate_processed_data(
        self,
        dataset: DatasetDict,
        config: Any
    ) -> bool:
        """
        Validate processed dataset for training.
        
        Args:
            dataset: Processed dataset
            config: Configuration object
            
        Returns:
            True if validation passes, False otherwise
        """
        logger.info("Validating processed dataset...")
        
        try:
            for split_name, split_data in dataset.items():
                if len(split_data) == 0:
                    logger.error(f"Empty {split_name} split")
                    return False
                
                # Check required fields based on training method and model type
                sample = split_data[0]
                
                if config.training_method == "finetune":
                    required_fields = ["input_text"]
                    if split_name == "train":
                        required_fields.append("target_text")
                else:  # instruction
                    if config.model_type == "plm":
                        required_fields = ["input_text"]
                        if split_name == "train":
                            required_fields.append("target_text")
                    else:  # LLM
                        required_fields = ["instruction"]
                
                missing_fields = [field for field in required_fields if field not in sample]
                if missing_fields:
                    logger.error(f"Missing fields in {split_name}: {missing_fields}")
                    return False
                
                # Check for empty required fields
                for field in required_fields:
                    if not sample[field] or (isinstance(sample[field], str) and len(sample[field].strip()) == 0):
                        logger.error(f"Empty {field} in {split_name}")
                        return False
                
                # Check sequence lengths if tokenizer is available
                if self.tokenizer is not None:
                    self._validate_sequence_lengths(split_data, config, split_name)
                
                logger.info(f"✓ {split_name} validation passed")
            
            logger.info("✓ Dataset validation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Dataset validation failed: {e}")
            return False
    
    def _validate_sequence_lengths(
        self,
        dataset: Dataset,
        config: Any,
        split_name: str
    ) -> None:
        """Validate sequence lengths against model limits."""
        
        # Sample a few examples for length checking
        sample_size = min(100, len(dataset))
        samples = dataset.select(range(sample_size))
        
        long_sequences = 0
        max_length = getattr(config, 'max_seq_length', 2048)
        
        for sample in samples:
            if config.training_method == "finetune" or config.model_type == "plm":
                text_to_check = sample.get("input_text", "")
            else:  # LLM instruction
                text_to_check = sample.get("instruction", "")
            
            if text_to_check:
                tokens = self.tokenizer.encode(text_to_check, add_special_tokens=True)
                if len(tokens) > max_length:
                    long_sequences += 1
        
        if long_sequences > 0:
            percentage = (long_sequences / sample_size) * 100
            logger.warning(
                f"{split_name}: {long_sequences}/{sample_size} ({percentage:.1f}%) samples exceed "
                f"max_length ({max_length}). Consider increasing max_length or improving data preprocessing."
            )
    
    def create_inference_input(
        self,
        context: str,
        question: str,
        config: Any
    ) -> Dict[str, Any]:
        """
        Create input for single inference.
        
        Args:
            context: Legal document context
            question: Question to answer
            config: Configuration object
            
        Returns:
            Formatted input for inference
        """
        # Clean inputs
        context = self._clean_text(context)
        question = self._clean_text(question)
        
        # Format based on model type and training method
        if config.training_method == "finetune" or config.model_type == "plm":
            formatted = self.instruction_manager.format_for_plm(
                context=context,
                question=question,
                answer="",
                template_name=getattr(config, 'instruction_template_name', 'vietnamese_legal')
            )
            
            input_text = formatted["input_text"]
            
            if self.tokenizer is not None:
                encoding = self.tokenizer(
                    input_text,
                    max_length=getattr(config, 'max_source_length', 1024),
                    truncation=True,
                    padding=False,
                    return_tensors="pt"
                )
                return {
                    "input_text": input_text,
                    **encoding
                }
            else:
                return {"input_text": input_text}
        
        else:  # LLM instruction
            instruction_text = self.instruction_manager.format_for_llm(
                context=context,
                question=question,
                answer="",
                chat_template_name=getattr(config, 'chat_template', 'chatml'),
                instruction_template_name=getattr(config, 'instruction_template_name', 'vietnamese_legal'),
                include_assistant_start=True
            )
            
            if self.tokenizer is not None:
                encoding = self.tokenizer(
                    instruction_text,
                    max_length=getattr(config, 'max_seq_length', 2048),
                    truncation=True,
                    padding=False,
                    return_tensors="pt"
                )
                return {
                    "instruction": instruction_text,
                    **encoding
                }
            else:
                return {"instruction": instruction_text}
    
    def postprocess_generated_text(
        self,
        generated_text: str,
        config: Any,
        original_input: str = ""
    ) -> str:
        """
        Postprocess generated text to extract clean answer.
        
        Args:
            generated_text: Raw generated text
            config: Configuration object
            original_input: Original input text (to remove if present)
            
        Returns:
            Clean answer text
        """
        # Remove original input if it appears in the output
        if original_input and original_input in generated_text:
            generated_text = generated_text.replace(original_input, "").strip()
        
        # For LLM instruction models, extract answer from chat format
        if config.model_type == "llm" and config.training_method == "instruct":
            chat_template_name = getattr(config, 'chat_template', 'chatml')
            generated_text = self.instruction_manager.extract_answer_from_response(
                generated_text, chat_template_name
            )
        
        # Final cleaning
        generated_text = self._clean_generated_text(generated_text)
        
        return generated_text
    
    def _clean_generated_text(self, text: str) -> str:
        """Clean generated text output."""
        if not text:
            return ""
        
        # Remove common artifacts from generation
        artifacts = [
            "<|im_start|>", "<|im_end|>", "<s>", "</s>", "<unk>", "<pad>",
            "### Assistant:", "### Response:", "[INST]", "[/INST]"
        ]
        
        for artifact in artifacts:
            text = text.replace(artifact, "")
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove any remaining special formatting
        text = re.sub(r'^[^\w\u00C0-\u1EF9]+|[^\w\u00C0-\u1EF9\s,.;:!?()]+$', '', text)
        
        return text.strip()
    
    def get_data_statistics(self, dataset: DatasetDict) -> Dict[str, Any]:
        """
        Get statistics about the processed dataset.
        
        Args:
            dataset: Processed dataset
            
        Returns:
            Dictionary containing dataset statistics
        """
        stats = {
            "total_samples": 0,
            "splits": {},
            "avg_lengths": {}
        }
        
        for split_name, split_data in dataset.items():
            num_samples = len(split_data)
            stats["total_samples"] += num_samples
            stats["splits"][split_name] = num_samples
            
            # Calculate average lengths
            if num_samples > 0:
                sample_size = min(1000, num_samples)
                samples = split_data.select(range(sample_size))
                
                lengths = []
                for sample in samples:
                    if "input_text" in sample:
                        lengths.append(len(sample["input_text"]))
                    elif "instruction" in sample:
                        lengths.append(len(sample["instruction"]))
                
                if lengths:
                    stats["avg_lengths"][split_name] = {
                        "avg_char_length": sum(lengths) / len(lengths),
                        "max_char_length": max(lengths),
                        "min_char_length": min(lengths)
                    }
        
        return stats