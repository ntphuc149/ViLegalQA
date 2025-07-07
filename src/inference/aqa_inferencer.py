"""
AQA Inferencer for generating answers from trained models.
Supports both PLM and LLM models with batch and single inference.
"""

import time
import pandas as pd
from typing import Dict, List, Any, Union, Optional
from datasets import Dataset
from tqdm import tqdm

from ..models.base_model import BaseAQAModel
from ..data.data_processor import AQADataProcessor
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class AQAInferencer:
    """
    Inferencer for Abstractive Question Answering models.
    
    Handles inference for both PLM and LLM models with support for:
    - Batch inference on datasets
    - Single sample inference
    - Answer post-processing
    - Results saving and formatting
    """
    
    def __init__(
        self,
        config: Any,
        model: BaseAQAModel,
        data_processor: Optional[AQADataProcessor] = None
    ):
        """
        Initialize AQA inferencer.
        
        Args:
            config: Inference configuration
            model: Trained model for inference
            data_processor: Data processor for input formatting
        """
        self.config = config
        self.model = model
        self.data_processor = data_processor or AQADataProcessor()
        
        # Validate model state
        if not self.model.is_loaded:
            raise ValueError("Model not loaded. Load model before inference.")
        
        # Set model to eval mode
        if self.model.model is not None:
            self.model.model.eval()
        
        logger.info("AQA Inferencer initialized")
    
    def run_inference(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = None,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Run inference on a dataset.
        
        Args:
            dataset: Dataset for inference
            batch_size: Batch size for inference (defaults to eval batch size)
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary with predictions and metadata
        """
        if batch_size is None:
            batch_size = getattr(self.config, 'per_device_eval_batch_size', 1)
        
        logger.info(f"Running inference on {len(dataset)} samples")
        start_time = time.time()
        
        predictions = []
        references = []
        contexts = []
        questions = []
        
        # Process in batches
        for i in tqdm(range(0, len(dataset), batch_size), disable=not show_progress, desc="Inference"):
            batch_end = min(i + batch_size, len(dataset))
            batch = dataset.select(range(i, batch_end))
            
            batch_predictions = self._process_batch(batch)
            predictions.extend(batch_predictions["predictions"])
            references.extend(batch_predictions["references"])
            contexts.extend(batch_predictions["contexts"])
            questions.extend(batch_predictions["questions"])
        
        inference_time = time.time() - start_time
        
        logger.info(f"✓ Inference completed in {inference_time:.2f} seconds")
        
        return {
            "predictions": predictions,
            "references": references,
            "contexts": contexts,
            "questions": questions,
            "inference_time": inference_time,
            "num_samples": len(predictions),
            "samples_per_second": len(predictions) / inference_time
        }
    
    def _process_batch(self, batch: Dataset) -> Dict[str, List[str]]:
        """
        Process a batch of samples.
        
        Args:
            batch: Batch of samples to process
            
        Returns:
            Dictionary with batch results
        """
        batch_predictions = []
        batch_references = []
        batch_contexts = []
        batch_questions = []
        
        for sample in batch:
            # Extract original data
            context = sample.get("original_context", sample.get("context", ""))
            question = sample.get("original_question", sample.get("question", ""))
            reference = sample.get("original_answer", sample.get("abstractive_answer", ""))
            
            # Generate prediction
            prediction = self.predict_single(context, question)
            
            batch_predictions.append(prediction)
            batch_references.append(reference)
            batch_contexts.append(context)
            batch_questions.append(question)
        
        return {
            "predictions": batch_predictions,
            "references": batch_references,
            "contexts": batch_contexts,
            "questions": batch_questions
        }
    
    def predict_single(
        self,
        context: str,
        question: str,
        **generation_kwargs
    ) -> str:
        """
        Generate prediction for a single sample.
        
        Args:
            context: Legal document context
            question: Question to answer
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Generated answer
        """
        try:
            # Create inference input
            inference_input = self.data_processor.create_inference_input(
                context, question, self.config
            )
            
            # Generate response
            output = self.model.generate(inference_input, **generation_kwargs)
            
            # Extract and clean generated text
            generated_text = output.generated_text
            if isinstance(generated_text, list):
                generated_text = generated_text[0]
            
            # Post-process
            cleaned_text = self.data_processor.postprocess_generated_text(
                generated_text, self.config, 
                original_input=inference_input.get("input_text", inference_input.get("instruction", ""))
            )
            
            return cleaned_text
            
        except Exception as e:
            logger.error(f"Prediction failed for sample: {e}")
            return ""
    
    def predict_batch(
        self,
        contexts: List[str],
        questions: List[str],
        **generation_kwargs
    ) -> List[str]:
        """
        Generate predictions for a batch of samples.
        
        Args:
            contexts: List of legal document contexts
            questions: List of questions
            **generation_kwargs: Additional generation parameters
            
        Returns:
            List of generated answers
        """
        if len(contexts) != len(questions):
            raise ValueError("Number of contexts and questions must match")
        
        predictions = []
        for context, question in zip(contexts, questions):
            prediction = self.predict_single(context, question, **generation_kwargs)
            predictions.append(prediction)
        
        return predictions
    
    def save_results(
        self,
        results: Dict[str, Any],
        output_path: str,
        include_inputs: bool = True
    ) -> None:
        """
        Save inference results to CSV file.
        
        Args:
            results: Results from run_inference
            output_path: Path to save results
            include_inputs: Whether to include input context and questions
        """
        data = {
            "predictions": results["predictions"],
            "references": results["references"]
        }
        
        if include_inputs:
            data.update({
                "contexts": results["contexts"],
                "questions": results["questions"]
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        logger.info(f"Results saved to {output_path}")
        
        # Save metadata
        metadata_path = output_path.replace('.csv', '_metadata.json')
        import json
        metadata = {
            "num_samples": results["num_samples"],
            "inference_time": results["inference_time"],
            "samples_per_second": results["samples_per_second"],
            "model_name": self.config.model_name,
            "model_type": self.config.model_type,
            "config": self.config.to_dict()
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Metadata saved to {metadata_path}")
    
    def interactive_inference(self) -> None:
        """
        Run interactive inference session.
        """
        logger.info("Starting interactive inference session")
        print("=== ViBidLQA-AQA Interactive Inference ===")
        print("Type 'quit' to exit")
        print()
        
        while True:
            try:
                # Get input
                print("Enter legal context:")
                context = input("> ").strip()
                
                if context.lower() == 'quit':
                    break
                
                print("Enter question:")
                question = input("> ").strip()
                
                if question.lower() == 'quit':
                    break
                
                if not context or not question:
                    print("Both context and question are required.")
                    continue
                
                # Generate prediction
                print("\nGenerating answer...")
                start_time = time.time()
                prediction = self.predict_single(context, question)
                inference_time = time.time() - start_time
                
                # Display result
                print(f"\nAnswer (generated in {inference_time:.2f}s):")
                print("=" * 50)
                print(prediction)
                print("=" * 50)
                print()
                
            except KeyboardInterrupt:
                print("\nExiting interactive session...")
                break
            except Exception as e:
                print(f"Error during inference: {e}")
                print()
    
    def benchmark_inference(
        self,
        dataset: Dataset,
        num_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Benchmark inference performance.
        
        Args:
            dataset: Dataset for benchmarking
            num_samples: Number of samples to benchmark (None for all)
            
        Returns:
            Benchmark results
        """
        if num_samples is not None:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        
        logger.info(f"Benchmarking inference on {len(dataset)} samples")
        
        # Warm-up run
        if len(dataset) > 0:
            sample = dataset[0]
            context = sample.get("original_context", sample.get("context", ""))
            question = sample.get("original_question", sample.get("question", ""))
            self.predict_single(context, question)
        
        # Benchmark run
        start_time = time.time()
        results = self.run_inference(dataset, show_progress=True)
        total_time = time.time() - start_time
        
        # Calculate metrics
        num_samples = len(results["predictions"])
        avg_time_per_sample = total_time / num_samples
        samples_per_second = num_samples / total_time
        
        # Calculate token statistics if possible
        total_input_tokens = 0
        total_output_tokens = 0
        
        if self.model.tokenizer:
            for i in range(min(100, num_samples)):  # Sample for efficiency
                context = results["contexts"][i]
                question = results["questions"][i]
                prediction = results["predictions"][i]
                
                input_text = f"{context} {question}"
                input_tokens = len(self.model.tokenizer.encode(input_text))
                output_tokens = len(self.model.tokenizer.encode(prediction))
                
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
            
            # Extrapolate to full dataset
            sample_ratio = num_samples / min(100, num_samples)
            total_input_tokens = int(total_input_tokens * sample_ratio)
            total_output_tokens = int(total_output_tokens * sample_ratio)
        
        benchmark_results = {
            "num_samples": num_samples,
            "total_time": total_time,
            "avg_time_per_sample": avg_time_per_sample,
            "samples_per_second": samples_per_second,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "tokens_per_second": (total_input_tokens + total_output_tokens) / total_time if total_time > 0 else 0,
            "model_info": self.model.get_model_info()
        }
        
        logger.info("Benchmark Results:")
        logger.info(f"  Samples: {num_samples}")
        logger.info(f"  Total Time: {total_time:.2f}s")
        logger.info(f"  Samples/sec: {samples_per_second:.2f}")
        logger.info(f"  Avg Time/sample: {avg_time_per_sample:.3f}s")
        
        return benchmark_results
    
    def get_inference_stats(self) -> Dict[str, Any]:
        """
        Get inference configuration and model statistics.
        
        Returns:
            Dictionary with inference information
        """
        stats = {
            "model_info": self.model.get_model_info(),
            "config": {
                "model_name": self.config.model_name,
                "model_type": self.config.model_type,
                "training_method": self.config.training_method,
                "max_new_tokens": getattr(self.config, 'max_new_tokens', 256),
                "temperature": getattr(self.config, 'temperature', 0.1),
                "top_p": getattr(self.config, 'top_p', 0.75),
                "do_sample": getattr(self.config, 'do_sample', True)
            }
        }
        
        return stats