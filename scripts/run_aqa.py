#!/usr/bin/env python3
"""
Main entry point for ViBidLQA-AQA training and inference.
Supports stage-wise execution and end-to-end training pipeline.

Usage:
    python scripts/run_aqa.py --model_name VietAI/vit5-base --do_end2end
    python scripts/run_aqa.py --config configs/vit5_finetune.yaml --do_finetune
    python scripts/run_aqa.py --checkpoint_path ./outputs/model --do_infer
    python scripts/run_aqa.py --results_file ./outputs/results.csv --do_eval
"""

import os
import sys
import time
import argparse
import traceback
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.configs.config_factory import ConfigFactory, add_config_args
from src.data.dataset_loader import ViBidLQALoader
from src.data.data_processor import AQADataProcessor
from src.models.model_factory import ModelFactory
from src.training.trainer_factory import TrainerFactory
from src.inference.aqa_inferencer import AQAInferencer
from src.evaluation.evaluator import AQAEvaluator
from src.utils.logging_utils import setup_logging, get_logger
from src.utils.file_utils import ensure_dir, save_json

logger = get_logger(__name__)


class AQAPipeline:
    """
    Main pipeline orchestrator for ViBidLQA-AQA system.
    Handles end-to-end execution from data loading to evaluation.
    """
    
    def __init__(self, config: Any):
        """
        Initialize AQA pipeline.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.dataset = None
        self.model = None
        self.trainer = None
        self.inferencer = None
        self.evaluator = None
        
        # Setup output directory
        ensure_dir(config.output_dir)
        
        # Initialize components
        self.data_loader = ViBidLQALoader(
            dataset_name=config.dataset_name,
            use_auth_token=getattr(config, 'use_auth_token', True)
        )
        self.data_processor = AQADataProcessor()
        
        logger.info("AQA Pipeline initialized")
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete AQA pipeline based on configuration.
        
        Returns:
            Dictionary with execution results
        """
        start_time = time.time()
        results = {"success": False, "stages_completed": [], "error": None}
        
        try:
            logger.info("🚀 Starting AQA Pipeline execution")
            
            # Execute stages based on configuration
            if self.config.do_end2end:
                results = self._run_end2end()
            else:
                if self.config.do_finetune:
                    results.update(self._run_training_stage())
                    results["stages_completed"].append("training")
                
                if self.config.do_infer:
                    results.update(self._run_inference_stage())
                    results["stages_completed"].append("inference")
                
                if self.config.do_eval:
                    results.update(self._run_evaluation_stage())
                    results["stages_completed"].append("evaluation")
            
            # Calculate total execution time
            total_time = time.time() - start_time
            results["total_execution_time"] = total_time
            results["success"] = True
            
            logger.info(f"✓ Pipeline completed successfully in {total_time:.2f} seconds")
            
        except Exception as e:
            error_msg = f"Pipeline execution failed: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            results["error"] = error_msg
            results["success"] = False
        
        finally:
            # Cleanup resources
            self._cleanup()
        
        return results
    
    def _run_end2end(self) -> Dict[str, Any]:
        """Run complete end-to-end pipeline."""
        logger.info("🔄 Running end-to-end pipeline")
        
        results = {"stages_completed": []}
        
        # Stage 1: Training
        logger.info("📚 Stage 1: Training")
        training_results = self._run_training_stage()
        results.update({"training": training_results})
        results["stages_completed"].append("training")
        
        if not training_results.get("success", False):
            raise Exception("Training stage failed")
        
        # Stage 2: Inference
        logger.info("🔮 Stage 2: Inference")
        inference_results = self._run_inference_stage()
        results.update({"inference": inference_results})
        results["stages_completed"].append("inference")
        
        if not inference_results.get("success", False):
            raise Exception("Inference stage failed")
        
        # Stage 3: Evaluation
        logger.info("📊 Stage 3: Evaluation")
        evaluation_results = self._run_evaluation_stage()
        results.update({"evaluation": evaluation_results})
        results["stages_completed"].append("evaluation")
        
        logger.info("✓ End-to-end pipeline completed")
        return results
    
    def _run_training_stage(self) -> Dict[str, Any]:
        """Run training stage."""
        try:
            logger.info("Starting training stage")
            
            # Load and process dataset
            if self.dataset is None:
                self._load_and_process_dataset()
            
            # Create and prepare model
            if self.model is None:
                self.model = ModelFactory.create_model_for_training(self.config)
            
            # Create and setup trainer
            self.trainer = TrainerFactory.create_trainer_for_training(
                self.config, self.model, self.dataset, self.data_processor
            )
            
            # Run training
            training_result = self.trainer.train()
            
            if training_result.success:
                logger.info("✓ Training completed successfully")
                return {
                    "success": True,
                    "model_path": training_result.model_path,
                    "final_loss": training_result.final_loss,
                    "best_metric": training_result.best_metric,
                    "training_time": training_result.training_time
                }
            else:
                raise Exception(training_result.error_message or "Training failed")
                
        except Exception as e:
            logger.error(f"Training stage failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _run_inference_stage(self) -> Dict[str, Any]:
        """Run inference stage."""
        try:
            logger.info("Starting inference stage")
            
            # Load dataset for inference
            if self.dataset is None:
                self._load_and_process_dataset(stage="inference")
            
            # Create model for inference
            checkpoint_path = getattr(self.config, 'checkpoint_path', None)
            if checkpoint_path is None and hasattr(self.config, 'output_dir'):
                checkpoint_path = self.config.output_dir
            
            if self.model is None:
                self.model = ModelFactory.create_model_for_inference(
                    self.config, checkpoint_path
                )
            
            # Create inferencer
            self.inferencer = AQAInferencer(
                self.config, self.model, self.data_processor
            )
            
            # Run inference on test set
            test_dataset = self.dataset.get("test")
            if test_dataset is None:
                raise ValueError("No test dataset available for inference")
            
            inference_result = self.inferencer.run_inference(test_dataset)
            
            # Save results
            results_file = os.path.join(self.config.output_dir, "results.csv")
            self.inferencer.save_results(inference_result, results_file)
            
            logger.info(f"✓ Inference completed, results saved to {results_file}")
            return {
                "success": True,
                "results_file": results_file,
                "num_samples": len(inference_result["predictions"]),
                "inference_time": inference_result.get("inference_time", 0)
            }
            
        except Exception as e:
            logger.error(f"Inference stage failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _run_evaluation_stage(self) -> Dict[str, Any]:
        """Run evaluation stage."""
        try:
            logger.info("Starting evaluation stage")
            
            # Get results file
            results_file = getattr(self.config, 'results_file', None)
            if results_file is None:
                results_file = os.path.join(self.config.output_dir, "results.csv")
            
            if not os.path.exists(results_file):
                raise FileNotFoundError(f"Results file not found: {results_file}")
            
            # Create evaluator
            self.evaluator = AQAEvaluator(self.config)
            
            # Run evaluation
            evaluation_result = self.evaluator.evaluate_from_file(results_file)
            
            # Save evaluation results
            eval_output_file = os.path.join(self.config.output_dir, "evaluation_results.json")
            save_json(evaluation_result, eval_output_file)
            
            logger.info(f"✓ Evaluation completed, results saved to {eval_output_file}")
            
            # Log key metrics
            metrics = evaluation_result.get("metrics", {})
            logger.info("📊 Evaluation Results:")
            for metric_name, value in metrics.items():
                logger.info(f"  {metric_name}: {value:.4f}")
            
            return {
                "success": True,
                "evaluation_file": eval_output_file,
                "metrics": metrics
            }
            
        except Exception as e:
            logger.error(f"Evaluation stage failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _load_and_process_dataset(self, stage: str = "train") -> None:
        """Load and process dataset."""
        logger.info(f"Loading dataset for {stage}")
        
        # Load dataset
        self.dataset = self.data_loader.load_dataset(
            split_mode=self.config.data_split_mode,
            train_ratio=getattr(self.config, 'train_ratio', 0.8),
            val_ratio=getattr(self.config, 'val_ratio', 0.1),
            test_ratio=getattr(self.config, 'test_ratio', 0.1),
            max_samples=getattr(self.config, 'max_samples', None),
            seed=getattr(self.config, 'seed', 42)
        )
        
        # Validate dataset
        if not self.data_loader.validate_dataset():
            raise ValueError("Dataset validation failed")
        
        # Process dataset
        if stage == "train":
            self.dataset = self.data_processor.process_dataset(
                self.dataset, self.config, stage="train"
            )
            
            # Validate processed dataset
            if not self.data_processor.validate_processed_data(self.dataset, self.config):
                raise ValueError("Processed dataset validation failed")
        
        logger.info("✓ Dataset loaded and processed successfully")
    
    def _cleanup(self) -> None:
        """Cleanup resources."""
        if self.trainer:
            self.trainer.cleanup()
        
        if self.model:
            self.model.cleanup()
        
        logger.info("Resources cleaned up")
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get pipeline information."""
        info = {
            "config": self.config.to_dict(),
            "components": {
                "dataset_loaded": self.dataset is not None,
                "model_loaded": self.model is not None and self.model.is_loaded,
                "trainer_setup": self.trainer is not None and self.trainer.is_setup,
                "inferencer_ready": self.inferencer is not None,
                "evaluator_ready": self.evaluator is not None
            }
        }
        
        if self.dataset:
            info["dataset_info"] = self.data_loader.get_dataset_info()
        
        if self.model:
            info["model_info"] = self.model.get_model_info()
        
        return info


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="ViBidLQA-AQA: Vietnamese Legal Question Answering System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # End-to-end training with ViT5
  python scripts/run_aqa.py --model_name VietAI/vit5-base --training_method finetune --do_end2end
  
  # Instruction tuning with Qwen2
  python scripts/run_aqa.py --model_name Qwen/Qwen2-7B-Instruct --training_method instruct --do_finetune
  
  # Stage-wise execution
  python scripts/run_aqa.py --config configs/my_config.yaml --do_finetune
  python scripts/run_aqa.py --checkpoint_path ./outputs/model --do_infer
  python scripts/run_aqa.py --results_file ./outputs/results.csv --do_eval
  
  # Custom dataset splitting
  python scripts/run_aqa.py --model_name VietAI/vit5-base --data_split_mode auto --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 --do_end2end
        """
    )
    
    # Add all configuration arguments
    parser = add_config_args(parser)
    
    # Add pipeline-specific arguments
    parser.add_argument("--log_level", type=str, default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--log_file", type=str, help="Log file path")
    
    # Validation and info arguments
    parser.add_argument("--validate_only", action="store_true",
                       help="Only validate configuration without running pipeline")
    parser.add_argument("--show_info", action="store_true",
                       help="Show pipeline information and exit")
    parser.add_argument("--show_recommendations", action="store_true",
                       help="Show training recommendations and exit")
    
    return parser


def validate_arguments(args: argparse.Namespace) -> bool:
    """
    Validate command line arguments.
    
    Args:
        args: Parsed arguments
        
    Returns:
        True if arguments are valid
    """
    try:
        # Check that at least one stage is specified
        stage_flags = [args.do_finetune, args.do_infer, args.do_eval, args.do_end2end]
        info_flags = [args.validate_only, args.show_info, args.show_recommendations]
        
        if not any(stage_flags + info_flags):
            logger.error("At least one execution flag must be specified")
            return False
        
        # Check stage-specific requirements
        if args.do_infer and not args.checkpoint_path:
            logger.error("--checkpoint_path is required for inference")
            return False
        
        if args.do_eval and not args.results_file:
            logger.error("--results_file is required for evaluation")
            return False
        
        # Check mutually exclusive flags
        if args.do_end2end and any([args.do_finetune, args.do_infer, args.do_eval]):
            logger.error("--do_end2end cannot be used with other stage flags")
            return False
        
        # Validate paths
        if args.checkpoint_path and not os.path.exists(args.checkpoint_path):
            logger.error(f"Checkpoint path does not exist: {args.checkpoint_path}")
            return False
        
        if args.results_file and not os.path.exists(args.results_file):
            logger.error(f"Results file does not exist: {args.results_file}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Argument validation failed: {e}")
        return False


def main():
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(
        log_level=args.log_level,
        log_file=args.log_file
    )
    
    logger.info("=== ViBidLQA-AQA Pipeline ===")
    logger.info(f"Command: {' '.join(sys.argv)}")
    
    try:
        # Validate arguments
        if not validate_arguments(args):
            sys.exit(1)
        
        # Create configuration
        config = ConfigFactory.from_args(args)
        logger.info(f"Configuration created: {config.model_type} {config.training_method}")
        
        # Handle info requests
        if args.show_info:
            pipeline = AQAPipeline(config)
            info = pipeline.get_pipeline_info()
            print("\n=== Pipeline Information ===")
            print(f"Model: {config.model_name}")
            print(f"Type: {config.model_type}")
            print(f"Training Method: {config.training_method}")
            print(f"Output Directory: {config.output_dir}")
            sys.exit(0)
        
        if args.show_recommendations:
            recommendations = TrainerFactory.get_recommended_settings(config)
            memory_info = TrainerFactory.get_memory_requirements(config)
            
            print("\n=== Training Recommendations ===")
            for key, value in recommendations.items():
                print(f"{key}: {value}")
            
            print(f"\n=== Memory Requirements ===")
            print(f"Estimated GPU Memory: {memory_info['recommended_gpu_memory_gb']:.1f} GB")
            print(f"Supports Quantization: {memory_info['supports_quantization']}")
            sys.exit(0)
        
        if args.validate_only:
            # Validate configuration and requirements
            if ModelFactory.validate_model_config(config):
                if TrainerFactory.validate_trainer_requirements(config):
                    logger.info("✓ All validations passed")
                    sys.exit(0)
            logger.error("✗ Validation failed")
            sys.exit(1)
        
        # Create and run pipeline
        pipeline = AQAPipeline(config)
        results = pipeline.run()
        
        # Print results summary
        print("\n=== Execution Summary ===")
        print(f"Success: {results['success']}")
        print(f"Stages Completed: {', '.join(results.get('stages_completed', []))}")
        
        if results['success']:
            print(f"Total Time: {results.get('total_execution_time', 0):.2f} seconds")
            
            # Print stage-specific results
            if 'training' in results:
                training = results['training']
                if training.get('success'):
                    print(f"Training Time: {training.get('training_time', 0):.2f} seconds")
                    if training.get('best_metric'):
                        print(f"Best Metric: {training['best_metric']:.4f}")
            
            if 'evaluation' in results:
                evaluation = results['evaluation']
                if evaluation.get('success') and evaluation.get('metrics'):
                    print("Key Metrics:")
                    for metric, value in evaluation['metrics'].items():
                        if isinstance(value, (int, float)):
                            print(f"  {metric}: {value:.4f}")
        else:
            print(f"Error: {results.get('error', 'Unknown error')}")
            sys.exit(1)
        
        logger.info("Pipeline execution completed")
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()