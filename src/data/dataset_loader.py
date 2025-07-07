"""
Dataset loader for ViBidLQA with flexible splitting and preprocessing.
Supports both predefined splits and automatic splitting with custom ratios.
"""

import logging
from typing import Dict, Tuple, Optional, Union
from pathlib import Path
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.model_selection import train_test_split

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class ViBidLQALoader:
    """
    Loader for ViBidLQA dataset with flexible splitting capabilities.
    
    Supports:
    - Loading from HuggingFace Hub or local path
    - Predefined splits or automatic splitting
    - Data validation and preprocessing
    - Memory-efficient loading for large datasets
    """
    
    def __init__(
        self,
        dataset_name: str = "Truong-Phuc/ViBidLQA",
        use_auth_token: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the dataset loader.
        
        Args:
            dataset_name: Dataset name on HuggingFace Hub or local path
            use_auth_token: Whether to use authentication token
            cache_dir: Directory to cache downloaded datasets
        """
        self.dataset_name = dataset_name
        self.use_auth_token = use_auth_token
        self.cache_dir = cache_dir
        self._raw_dataset = None
        self._processed_dataset = None
        
        logger.info(f"Initialized ViBidLQALoader for dataset: {dataset_name}")
    
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
        Load and split the dataset.
        
        Args:
            split_mode: 'auto' for custom splitting or 'predefined' for existing splits
            train_ratio: Ratio for training set (only used in auto mode)
            val_ratio: Ratio for validation set (only used in auto mode)
            test_ratio: Ratio for test set (only used in auto mode)
            max_samples: Maximum number of samples to load (for debugging)
            seed: Random seed for reproducible splitting
            
        Returns:
            DatasetDict containing train/validation/test splits
        """
        logger.info(f"Loading dataset with split_mode='{split_mode}'")
        
        # Validate split ratios
        if split_mode == "auto":
            total_ratio = train_ratio + val_ratio + test_ratio
            if abs(total_ratio - 1.0) > 1e-6:
                raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
        
        # Load raw dataset
        self._load_raw_dataset()
        
        # Apply sampling if specified
        if max_samples is not None:
            self._apply_sampling(max_samples, seed)
        
        # Split dataset
        if split_mode == "auto":
            dataset_dict = self._auto_split_dataset(train_ratio, val_ratio, test_ratio, seed)
        elif split_mode == "predefined":
            dataset_dict = self._load_predefined_splits()
        else:
            raise ValueError(f"Invalid split_mode: {split_mode}")
        
        self._processed_dataset = dataset_dict
        self._log_dataset_info(dataset_dict)
        
        return dataset_dict
    
    def _load_raw_dataset(self) -> None:
        """Load the raw dataset from HuggingFace Hub or local path."""
        try:
            logger.info(f"Loading dataset from: {self.dataset_name}")
            
            if Path(self.dataset_name).exists():
                # Load from local path
                self._raw_dataset = self._load_local_dataset(self.dataset_name)
            else:
                # Load from HuggingFace Hub
                self._raw_dataset = load_dataset(
                    self.dataset_name,
                    use_auth_token=self.use_auth_token,
                    cache_dir=self.cache_dir
                )
            
            logger.info("Dataset loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def _load_local_dataset(self, path: str) -> DatasetDict:
        """Load dataset from local CSV files."""
        path = Path(path)
        dataset_dict = {}
        
        # Look for standard split files
        for split in ["train", "validation", "test"]:
            csv_file = path / f"{split}.csv"
            if csv_file.exists():
                df = pd.read_csv(csv_file)
                dataset_dict[split] = Dataset.from_pandas(df, preserve_index=False)
                logger.info(f"Loaded {split} split: {len(df)} samples")
        
        # If no split files found, look for single file
        if not dataset_dict:
            csv_files = list(path.glob("*.csv"))
            if csv_files:
                df = pd.read_csv(csv_files[0])
                dataset_dict["train"] = Dataset.from_pandas(df, preserve_index=False)
                logger.info(f"Loaded single file: {len(df)} samples")
        
        return DatasetDict(dataset_dict)
    
    def _apply_sampling(self, max_samples: int, seed: int) -> None:
        """Apply sampling to limit dataset size."""
        logger.info(f"Applying sampling: max_samples={max_samples}")
        
        if isinstance(self._raw_dataset, DatasetDict):
            # Sample from each split
            for split_name, split_data in self._raw_dataset.items():
                if len(split_data) > max_samples:
                    sampled = split_data.shuffle(seed=seed).select(range(max_samples))
                    self._raw_dataset[split_name] = sampled
                    logger.info(f"Sampled {split_name}: {len(sampled)} samples")
        else:
            # Sample from single dataset
            if len(self._raw_dataset) > max_samples:
                self._raw_dataset = self._raw_dataset.shuffle(seed=seed).select(range(max_samples))
                logger.info(f"Sampled dataset: {len(self._raw_dataset)} samples")
    
    def _auto_split_dataset(
        self,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        seed: int
    ) -> DatasetDict:
        """Automatically split dataset into train/val/test."""
        logger.info(f"Auto-splitting dataset: {train_ratio}/{val_ratio}/{test_ratio}")
        
        # Get the dataset to split
        if isinstance(self._raw_dataset, DatasetDict):
            # If already split, concatenate all splits
            all_data = []
            for split_data in self._raw_dataset.values():
                all_data.extend(split_data.to_dict())
            
            # Convert back to dataset
            combined_dict = {key: [item[key] for item in all_data] for key in all_data[0].keys()}
            dataset = Dataset.from_dict(combined_dict)
        else:
            dataset = self._raw_dataset
        
        # Convert to pandas for easier splitting
        df = dataset.to_pandas()
        
        # First split: separate test set
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_ratio,
            random_state=seed,
            stratify=None  # Could add stratification based on answer length
        )
        
        # Second split: separate train and validation
        adjusted_val_ratio = val_ratio / (train_ratio + val_ratio)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=adjusted_val_ratio,
            random_state=seed
        )
        
        # Convert back to datasets
        dataset_dict = DatasetDict({
            "train": Dataset.from_pandas(train_df, preserve_index=False),
            "validation": Dataset.from_pandas(val_df, preserve_index=False),
            "test": Dataset.from_pandas(test_df, preserve_index=False)
        })
        
        return dataset_dict
    
    def _load_predefined_splits(self) -> DatasetDict:
        """Load predefined dataset splits."""
        logger.info("Using predefined dataset splits")
        
        if not isinstance(self._raw_dataset, DatasetDict):
            raise ValueError("Predefined splits not available for this dataset")
        
        # Ensure we have the required splits
        required_splits = ["train", "test"]
        available_splits = list(self._raw_dataset.keys())
        
        # Map common split names
        split_mapping = {
            "validation": ["validation", "val", "dev"],
            "test": ["test", "eval"]
        }
        
        dataset_dict = {}
        
        # Add train split
        if "train" in available_splits:
            dataset_dict["train"] = self._raw_dataset["train"]
        else:
            raise ValueError("No train split found in dataset")
        
        # Add validation split
        val_split = None
        for split_name in split_mapping["validation"]:
            if split_name in available_splits:
                val_split = split_name
                break
        
        if val_split:
            dataset_dict["validation"] = self._raw_dataset[val_split]
        else:
            # Create validation split from train
            logger.warning("No validation split found, creating from train split")
            train_val = self._raw_dataset["train"].train_test_split(test_size=0.1, seed=42)
            dataset_dict["train"] = train_val["train"]
            dataset_dict["validation"] = train_val["test"]
        
        # Add test split
        if "test" in available_splits:
            dataset_dict["test"] = self._raw_dataset["test"]
        else:
            raise ValueError("No test split found in dataset")
        
        return DatasetDict(dataset_dict)
    
    def _log_dataset_info(self, dataset_dict: DatasetDict) -> None:
        """Log information about the loaded dataset."""
        logger.info("=== Dataset Information ===")
        
        total_samples = 0
        for split_name, split_data in dataset_dict.items():
            num_samples = len(split_data)
            total_samples += num_samples
            logger.info(f"{split_name.capitalize()}: {num_samples:,} samples")
        
        logger.info(f"Total: {total_samples:,} samples")
        
        # Log data schema
        if dataset_dict:
            sample_split = next(iter(dataset_dict.values()))
            logger.info(f"Features: {list(sample_split.features.keys())}")
            
            # Log sample data statistics
            if len(sample_split) > 0:
                sample = sample_split[0]
                logger.info("=== Sample Statistics ===")
                
                if "context" in sample:
                    ctx_len = len(sample["context"])
                    logger.info(f"Sample context length: {ctx_len} characters")
                
                if "question" in sample:
                    q_len = len(sample["question"])
                    logger.info(f"Sample question length: {q_len} characters")
                
                if "abstractive_answer" in sample:
                    ans_len = len(sample["abstractive_answer"])
                    logger.info(f"Sample answer length: {ans_len} characters")
    
    def get_dataset_info(self) -> Dict:
        """Get detailed information about the loaded dataset."""
        if self._processed_dataset is None:
            raise ValueError("Dataset not loaded yet. Call load_dataset() first.")
        
        info = {
            "dataset_name": self.dataset_name,
            "splits": {},
            "total_samples": 0,
            "features": None
        }
        
        for split_name, split_data in self._processed_dataset.items():
            num_samples = len(split_data)
            info["splits"][split_name] = {
                "num_samples": num_samples,
                "features": list(split_data.features.keys())
            }
            info["total_samples"] += num_samples
        
        # Get features from first split
        if self._processed_dataset:
            first_split = next(iter(self._processed_dataset.values()))
            info["features"] = list(first_split.features.keys())
        
        return info
    
    def validate_dataset(self) -> bool:
        """
        Validate the loaded dataset for required fields and data quality.
        
        Returns:
            True if dataset is valid, False otherwise
        """
        if self._processed_dataset is None:
            logger.error("No dataset loaded for validation")
            return False
        
        required_fields = ["context", "question", "abstractive_answer"]
        
        try:
            for split_name, split_data in self._processed_dataset.items():
                logger.info(f"Validating {split_name} split...")
                
                # Check required fields
                features = list(split_data.features.keys())
                missing_fields = [field for field in required_fields if field not in features]
                
                if missing_fields:
                    logger.error(f"Missing required fields in {split_name}: {missing_fields}")
                    return False
                
                # Check for empty samples
                if len(split_data) == 0:
                    logger.error(f"Empty {split_name} split")
                    return False
                
                # Sample validation
                sample = split_data[0]
                for field in required_fields:
                    if not sample[field] or len(sample[field].strip()) == 0:
                        logger.error(f"Empty {field} in {split_name} sample")
                        return False
                
                logger.info(f"✓ {split_name} split validation passed")
            
            logger.info("✓ Dataset validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Dataset validation failed: {e}")
            return False
    
    def save_dataset(self, output_dir: str) -> None:
        """Save the processed dataset to disk."""
        if self._processed_dataset is None:
            raise ValueError("No dataset to save. Load dataset first.")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for split_name, split_data in self._processed_dataset.items():
            # Save as CSV
            csv_path = output_path / f"{split_name}.csv"
            split_data.to_csv(csv_path)
            logger.info(f"Saved {split_name} split to {csv_path}")
        
        # Save dataset info
        info_path = output_path / "dataset_info.json"
        import json
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(self.get_dataset_info(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Dataset saved to {output_dir}")