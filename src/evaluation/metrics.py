"""
Evaluation metrics for Abstractive Question Answering.
Implements ROUGE, BLEU, METEOR, BERTScore and other metrics.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class AQAMetrics:
    """
    Comprehensive metrics computation for AQA evaluation.
    """
    
    def __init__(self, language: str = "vi"):
        """
        Initialize metrics calculator.
        
        Args:
            language: Language code for language-specific metrics
        """
        self.language = language
        self.rouge = None
        self.meteor = None
        self.bertscore = None
        
        # Initialize metrics
        self._initialize_metrics()
    
    def _initialize_metrics(self) -> None:
        """Initialize metric calculators."""
        try:
            import evaluate
            
            # Load metrics
            self.rouge = evaluate.load('rouge')
            self.meteor = evaluate.load('meteor')
            self.bertscore = evaluate.load("bertscore")
            
            logger.info("✓ Evaluation metrics initialized")
            
        except ImportError as e:
            logger.warning(f"Some metrics not available: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize metrics: {e}")
    
    def compute_rouge(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute ROUGE scores.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            
        Returns:
            Dictionary with ROUGE scores
        """
        if self.rouge is None:
            logger.warning("ROUGE not available")
            return {}
        
        try:
            scores = self.rouge.compute(
                predictions=predictions,
                references=references
            )
            
            # Convert to percentages and extract F1 scores
            return {
                "rouge1": scores['rouge1'] * 100,
                "rouge2": scores['rouge2'] * 100,
                "rougeL": scores['rougeL'] * 100,
                "rougeLsum": scores['rougeLsum'] * 100
            }
            
        except Exception as e:
            logger.error(f"ROUGE computation failed: {e}")
            return {}
    
    def compute_bleu(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute BLEU scores.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            
        Returns:
            Dictionary with BLEU scores
        """
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            import spacy
            
            # Load Vietnamese tokenizer if available
            try:
                nlp = spacy.load('vi_core_news_lg')
            except OSError:
                # Fallback to basic tokenization
                nlp = None
                logger.warning("Vietnamese spaCy model not available, using basic tokenization")
            
            bleu_scores = {1: [], 2: [], 3: [], 4: []}
            weights = [
                (1, 0, 0, 0),
                (0.5, 0.5, 0, 0),
                (0.33, 0.33, 0.33, 0),
                (0.25, 0.25, 0.25, 0.25)
            ]
            
            smoothing = SmoothingFunction().method1
            
            for pred, ref in zip(predictions, references):
                # Tokenize
                if nlp:
                    pred_tokens = [token.text for token in nlp(pred)]
                    ref_tokens = [token.text for token in nlp(ref)]
                else:
                    pred_tokens = pred.split()
                    ref_tokens = ref.split()
                
                # Compute BLEU for different n-grams
                for n, weight in enumerate(weights, start=1):
                    try:
                        score = sentence_bleu(
                            [ref_tokens], pred_tokens,
                            weights=weight,
                            smoothing_function=smoothing
                        )
                        bleu_scores[n].append(score)
                    except:
                        bleu_scores[n].append(0.0)
            
            return {
                f"bleu{n}": (sum(scores) / len(scores)) * 100
                for n, scores in bleu_scores.items()
            }
            
        except ImportError:
            logger.warning("NLTK not available for BLEU computation")
            return {}
        except Exception as e:
            logger.error(f"BLEU computation failed: {e}")
            return {}
    
    def compute_meteor(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute METEOR score.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            
        Returns:
            Dictionary with METEOR score
        """
        if self.meteor is None:
            logger.warning("METEOR not available")
            return {}
        
        try:
            score = self.meteor.compute(
                predictions=predictions,
                references=references
            )
            
            return {"meteor": score['meteor'] * 100}
            
        except Exception as e:
            logger.error(f"METEOR computation failed: {e}")
            return {}
    
    def compute_bertscore(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute BERTScore.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            
        Returns:
            Dictionary with BERTScore metrics
        """
        if self.bertscore is None:
            logger.warning("BERTScore not available")
            return {}
        
        try:
            scores = self.bertscore.compute(
                predictions=predictions,
                references=references,
                lang=self.language
            )
            
            return {
                "bertscore_precision": np.mean(scores['precision']) * 100,
                "bertscore_recall": np.mean(scores['recall']) * 100,
                "bertscore_f1": np.mean(scores['f1']) * 100
            }
            
        except Exception as e:
            logger.error(f"BERTScore computation failed: {e}")
            return {}
    
    def compute_length_stats(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute length statistics.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            
        Returns:
            Dictionary with length statistics
        """
        try:
            pred_lengths = [len(pred.split()) for pred in predictions]
            ref_lengths = [len(ref.split()) for ref in references]
            
            return {
                "avg_pred_length": np.mean(pred_lengths),
                "avg_ref_length": np.mean(ref_lengths),
                "length_ratio": np.mean(pred_lengths) / np.mean(ref_lengths) if np.mean(ref_lengths) > 0 else 0,
                "std_pred_length": np.std(pred_lengths),
                "std_ref_length": np.std(ref_lengths)
            }
            
        except Exception as e:
            logger.error(f"Length statistics computation failed: {e}")
            return {}
    
    def compute_all_metrics(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute all available metrics.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            
        Returns:
            Dictionary with all computed metrics
        """
        logger.info(f"Computing metrics for {len(predictions)} samples")
        
        all_metrics = {}
        
        # ROUGE scores
        rouge_scores = self.compute_rouge(predictions, references)
        all_metrics.update(rouge_scores)
        
        # BLEU scores
        bleu_scores = self.compute_bleu(predictions, references)
        all_metrics.update(bleu_scores)
        
        # METEOR score
        meteor_scores = self.compute_meteor(predictions, references)
        all_metrics.update(meteor_scores)
        
        # BERTScore
        bertscore_scores = self.compute_bertscore(predictions, references)
        all_metrics.update(bertscore_scores)
        
        # Length statistics
        length_stats = self.compute_length_stats(predictions, references)
        all_metrics.update(length_stats)
        
        logger.info(f"✓ Computed {len(all_metrics)} metrics")
        return all_metrics