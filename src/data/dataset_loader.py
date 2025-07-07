"""
Dataset loader for ViBidLQA dataset.
Handles loading, splitting, and basic preprocessing.
"""

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from typing import Dict, Tuple, Optional, Any, Union
from pathlib import Path
import random

from ..utils.logging_utils import get_logger, log_dataset_info
from ..utils.file_utils import ensure_dir, save_json

logger = get_logger(__name__)


class ViBidLQALoader:
    """
    Loader for ViBidLQA dataset with flexible splitting options.
    """
    
    def __init__(
        self,
        dataset_name: str = "Truong-Phuc/ViBidLQA",
        use_auth_token: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize dataset loader.
        
        Args:
            dataset_name: HuggingFace dataset name or local path
            use_auth_token: Whether to use HF authentication
            cache_dir: Cache directory for dataset
        """
        self.dataset_name = dataset_name
        self.use_auth_token = use_auth_token
        self.cache_dir = cache_dir
        self.dataset = None
        
    def load_dataset(
        self,
        split_mode: str = "auto",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1, 
        test_ratio: float = 0.1,
        max_samples: Optional[int] = None,
        seed: int = 42
    ) -> DatasetDict:
        """
        Load and split the ViBidLQA dataset.
        
        Args:
            split_mode: "auto" for custom split, "predefined" for existing splits
            train_ratio: Training data ratio (for auto split)
            val_ratio: Validation data ratio (for auto split)
            test_ratio: Test data ratio (for auto split)
            max_samples: Maximum samples to load (for testing)
            seed: Random seed for splitting
            
        Returns:
            DatasetDict with train/validation/test splits
        """
        logger.info(f"Loading dataset: {self.dataset_name}")
        
        try:
            # Load raw dataset
            if self._is_local_path():
                raw_dataset = self._load_local_dataset()
            else:
                raw_dataset = load_dataset(
                    self.dataset_name,
                    use_auth_token=self.use_auth_token,
                    cache_dir=self.cache_dir
                )
            
            # Apply sample limit if specified
            if max_samples:
                raw_dataset = self._limit_samples(raw_dataset, max_samples, seed)
            
            # Split dataset based on mode
            if split_mode == "auto":
                dataset = self._auto_split_dataset(
                    raw_dataset, train_ratio, val_ratio, test_ratio, seed
                )
            elif split_mode == "predefined":
                dataset = self._use_predefined_splits(raw_dataset)
            else:
                raise ValueError(f"Invalid split_mode: {split_mode}")
            
            # Validate and clean dataset
            dataset = self._validate_and_clean_dataset(dataset)
            
            self.dataset = dataset
            log_dataset_info(dataset, logger)
            
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def _is_local_path(self) -> bool:
        """Check if dataset_name is a local path."""
        return Path(self.dataset_name).exists()
    
    def _load_local_dataset(self) -> DatasetDict:
        """Load dataset from local files."""
        dataset_path = Path(self.dataset_name)
        
        if dataset_path.is_file():
            # Single file - assume CSV/JSON
            if dataset_path.suffix == '.csv':
                df = pd.read_csv(dataset_path)
                return DatasetDict({"train": Dataset.from_pandas(df)})
            elif dataset_path.suffix == '.json':
                df = pd.read_json(dataset_path)
                return DatasetDict({"train": Dataset.from_pandas(df)})
            else:
                raise ValueError(f"Unsupported file format: {dataset_path.suffix}")
        
        elif dataset_path.is_dir():
            # Directory with split files
            splits = {}
            for split_file in dataset_path.glob("*.csv"):
                split_name = split_file.stem
                df = pd.read_csv(split_file)
                splits[split_name] = Dataset.from_pandas(df)
            
            if not splits:
                raise ValueError(f"No CSV files found in {dataset_path}")
            
            return DatasetDict(splits)
        
        else:
            raise ValueError(f"Dataset path not found: {dataset_path}")
    
    def _limit_samples(
        self,
        dataset: DatasetDict,
        max_samples: int,
        seed: int
    ) -> DatasetDict:
        """Limit dataset to maximum number of samples."""
        logger.info(f"Limiting dataset to {max_samples} samples")
        
        # Set random seed
        random.seed(seed)
        
        limited_dataset = {}
        for split_name, split_data in dataset.items():
            if len(split_data) > max_samples:
                # Sample indices
                indices = random.sample(range(len(split_data)), max_samples)
                limited_dataset[split_name] = split_data.select(indices)
            else:
                limited_dataset[split_name] = split_data
        
        return DatasetDict(limited_dataset)
    
    def _auto_split_dataset(
        self,
        raw_dataset: DatasetDict,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        seed: int
    ) -> DatasetDict:
        """Automatically split dataset into train/val/test."""
        logger.info(f"Auto-splitting dataset: {train_ratio:.1f}/{val_ratio:.1f}/{test_ratio:.1f}")
        
        # Combine all data
        if isinstance(raw_dataset, DatasetDict):
            # Combine all existing splits
            all_data = []
            for split_data in raw_dataset.values():
                all_data.extend(split_data)
            combined_dataset = Dataset.from_list(all_data)
        else:
            combined_dataset = raw_dataset
        
        # Calculate split sizes
        total_size = len(combined_dataset)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        test_size = total_size - train_size - val_size
        
        logger.info(f"Split sizes - Train: {train_size}, Val: {val_size}, Test: {test_size}")
        
        # Shuffle and split
        shuffled_dataset = combined_dataset.shuffle(seed=seed)
        
        train_dataset = shuffled_dataset.select(range(train_size))
        val_dataset = shuffled_dataset.select(range(train_size, train_size + val_size))
        test_dataset = shuffled_dataset.select(range(train_size + val_size, total_size))
        
        return DatasetDict({
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset
        })
    
    def _use_predefined_splits(self, raw_dataset: DatasetDict) -> DatasetDict:
        """Use predefined dataset splits."""
        logger.info("Using predefined dataset splits")
        
        if not isinstance(raw_dataset, DatasetDict):
            raise ValueError("Predefined splits mode requires DatasetDict input")
        
        # Check for required splits
        if "train" not in raw_dataset:
            raise ValueError("Missing 'train' split in dataset")
        
        # Create validation split if missing
        if "validation" not in raw_dataset and "val" not in raw_dataset:
            logger.warning("No validation split found, creating from train split")
            train_data = raw_dataset["train"]
            
            # Split train into train/val (90/10)
            split_point = int(len(train_data) * 0.9)
            shuffled_data = train_data.shuffle(seed=42)
            
            new_train = shuffled_data.select(range(split_point))
            new_val = shuffled_data.select(range(split_point, len(train_data)))
            
            raw_dataset = DatasetDict({
                "train": new_train,
                "validation": new_val,
                "test": raw_dataset.get("test", new_val)  # Use val as test if no test split
            })
        
        # Standardize split names
        standardized_dataset = {}
        for split_name, split_data in raw_dataset.items():
            if split_name in ["train", "training"]:
                standardized_dataset["train"] = split_data
            elif split_name in ["validation", "val", "valid", "dev"]:
                standardized_dataset["validation"] = split_data
            elif split_name in ["test", "testing"]:
                standardized_dataset["test"] = split_data
        
        return DatasetDict(standardized_dataset)
    
    def _validate_and_clean_dataset(self, dataset: DatasetDict) -> DatasetDict:
        """Validate and clean the dataset."""
        logger.info("Validating and cleaning dataset")
        
        cleaned_dataset = {}
        
        for split_name, split_data in dataset.items():
            logger.info(f"Processing {split_name} split: {len(split_data)} samples")
            
            # Convert to pandas for easier processing
            df = split_data.to_pandas()
            
            # Check required columns
            required_columns = ["context", "question"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns in {split_name}: {missing_columns}")
            
            # Check for answer column (extractive or abstractive)
            answer_columns = ["answer", "extractive_answer", "abstractive_answer"]
            available_answer_cols = [col for col in answer_columns if col in df.columns]
            
            if not available_answer_cols:
                raise ValueError(f"No answer column found in {split_name}. Expected one of: {answer_columns}")
            
            # Standardize answer column names
            if "abstractive_answer" in df.columns:
                df["answer"] = df["abstractive_answer"]
            elif "extractive_answer" in df.columns:
                df["answer"] = df["extractive_answer"]
            # If "answer" already exists, keep it
            
            # Clean data
            df = self._clean_dataframe(df)
            
            # Convert back to dataset
            cleaned_dataset[split_name] = Dataset.from_pandas(df, preserve_index=False)
        
        return DatasetDict(cleaned_dataset)
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean individual dataframe."""
        original_size = len(df)
        
        # Remove rows with missing critical fields
        df = df.dropna(subset=["context", "question", "answer"])
        
        # Remove empty strings
        df = df[df["context"].str.strip().str.len() > 0]
        df = df[df["question"].str.strip().str.len() > 0]
        df = df[df["answer"].str.strip().str.len() > 0]
        
        # Convert to string type
        df["context"] = df["context"].astype(str)
        df["question"] = df["question"].astype(str)
        df["answer"] = df["answer"].astype(str)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=["context", "question"], keep="first")
        
        cleaned_size = len(df)
        if cleaned_size < original_size:
            logger.info(f"Cleaned dataset: {original_size} -> {cleaned_size} samples")
        
        return df.reset_index(drop=True)
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get comprehensive dataset statistics."""
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        stats = {}
        
        for split_name, split_data in self.dataset.items():
            df = split_data.to_pandas()
            
            split_stats = {
                "num_samples": len(df),
                "avg_context_length": df["context"].str.len().mean(),
                "avg_question_length": df["question"].str.len().mean(),
                "avg_answer_length": df["answer"].str.len().mean(),
                "max_context_length": df["context"].str.len().max(),
                "max_question_length": df["question"].str.len().max(),
                "max_answer_length": df["answer"].str.len().max(),
                "min_context_length": df["context"].str.len().min(),
                "min_question_length": df["question"].str.len().min(),
                "min_answer_length": df["answer"].str.len().min(),
            }
            
            stats[split_name] = split_stats
        
        return stats
    
    def save_dataset_splits(
        self,
        output_dir: Union[str, Path],
        format: str = "csv"
    ) -> None:
        """
        Save dataset splits to files.
        
        Args:
            output_dir: Directory to save splits
            format: Output format ("csv", "json", "parquet")
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        output_dir = Path(output_dir)
        ensure_dir(output_dir)
        
        for split_name, split_data in self.dataset.items():
            df = split_data.to_pandas()
            
            if format == "csv":
                output_path = output_dir / f"{split_name}.csv"
                df.to_csv(output_path, index=False)
            elif format == "json":
                output_path = output_dir / f"{split_name}.json"
                df.to_json(output_path, orient="records", indent=2)
            elif format == "parquet":
                output_path = output_dir / f"{split_name}.parquet"
                df.to_parquet(output_path, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Saved {split_name} split to: {output_path}")
        
        # Save statistics
        stats = self.get_dataset_statistics()
        save_json(stats, output_dir / "dataset_statistics.json")
        logger.info(f"Dataset statistics saved to: {output_dir / 'dataset_statistics.json'}")


def create_sample_dataset(
    num_samples: int = 100,
    output_path: Optional[str] = None
) -> DatasetDict:
    """
    Create a sample dataset for testing purposes.
    
    Args:
        num_samples: Number of samples to create
        output_path: Optional path to save the sample dataset
        
    Returns:
        Sample DatasetDict
    """
    import random
    
    # Sample Vietnamese legal contexts and questions
    sample_contexts = [
        "Luật Đấu thầu số 22/2023/QH15 quy định về việc tổ chức đấu thầu trong các dự án đầu tư công.",
        "Nghị định số 24/2024/NĐ-CP hướng dẫn chi tiết một số điều của Luật Đấu thầu.",
        "Thông tư số 05/2024/TT-BKHĐT quy định về mẫu hồ sơ mời thầu và hồ sơ dự thầu.",
    ]
    
    sample_questions = [
        "Văn bản nào quy định về việc tổ chức đấu thầu?",
        "Nghị định nào hướng dẫn chi tiết Luật Đấu thầu?",
        "Thông tư nào quy định về mẫu hồ sơ đấu thầu?",
    ]
    
    sample_answers = [
        "Luật Đấu thầu số 22/2023/QH15",
        "Nghị định số 24/2024/NĐ-CP",
        "Thông tư số 05/2024/TT-BKHĐT",
    ]
    
    # Generate sample data
    data = []
    for i in range(num_samples):
        context_idx = i % len(sample_contexts)
        question_idx = i % len(sample_questions)
        answer_idx = i % len(sample_answers)
        
        data.append({
            "context": sample_contexts[context_idx],
            "question": sample_questions[question_idx],
            "answer": sample_answers[answer_idx],
            "abstractive_answer": f"Theo quy định, {sample_answers[answer_idx]} là văn bản có thẩm quyền."
        })
    
    # Create dataset
    df = pd.DataFrame(data)
    dataset = Dataset.from_pandas(df, preserve_index=False)
    
    # Split into train/val/test
    train_size = int(num_samples * 0.8)
    val_size = int(num_samples * 0.1)
    
    dataset_dict = DatasetDict({
        "train": dataset.select(range(train_size)),
        "validation": dataset.select(range(train_size, train_size + val_size)),
        "test": dataset.select(range(train_size + val_size, num_samples))
    })
    
    # Save if path provided
    if output_path:
        output_path = Path(output_path)
        ensure_dir(output_path)
        
        for split_name, split_data in dataset_dict.items():
            df = split_data.to_pandas()
            df.to_csv(output_path / f"{split_name}.csv", index=False)
        
        logger.info(f"Sample dataset saved to: {output_path}")
    
    logger.info(f"Created sample dataset with {num_samples} samples")
    return dataset_dict