import os
import pandas as pd
import kaggle
from pathlib import Path
import logging
from typing import Tuple, Dict, Any
import zipfile

logger = logging.getLogger(__name__)

class DatasetLoader:
    """Load and manage AI detection datasets from Kaggle"""
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset configurations
        self.datasets = {
            "ai_detection_main": {
                "kaggle_id": "sunilthite/llm-detect-ai-generated-text-dataset",
                "files": ["train.csv", "test.csv"]
            },
            "ai_detection_augmented": {
                "kaggle_id": "alejopaullier/augmented-data-for-llm-detect-ai-generated-text", 
                "files": ["train.csv"]
            }
        }
    
    def setup_kaggle_api(self) -> bool:
        """Setup Kaggle API authentication"""
        try:
            # Check if kaggle.json exists
            kaggle_dir = Path.home() / ".kaggle"
            kaggle_json = kaggle_dir / "kaggle.json"
            
            if not kaggle_json.exists():
                # Try project directory
                project_kaggle = Path(".kaggle/kaggle.json")
                if project_kaggle.exists():
                    kaggle_dir.mkdir(exist_ok=True)
                    import shutil
                    shutil.copy(project_kaggle, kaggle_json)
                    kaggle_json.chmod(0o600)
                else:
                    logger.error("Kaggle API token not found. Please download kaggle.json from kaggle.com")
                    return False
            
            # Test API connection
            kaggle.api.authenticate()
            logger.info("Kaggle API authenticated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup Kaggle API: {e}")
            return False
    
    def download_dataset(self, dataset_name: str) -> bool:
        """Download a specific dataset from Kaggle"""
        try:
            if dataset_name not in self.datasets:
                logger.error(f"Unknown dataset: {dataset_name}")
                return False
            
            dataset_config = self.datasets[dataset_name]
            kaggle_id = dataset_config["kaggle_id"]
            
            # Create dataset directory
            dataset_dir = self.raw_dir / dataset_name
            dataset_dir.mkdir(exist_ok=True)
            
            logger.info(f"Downloading dataset: {kaggle_id}")
            
            # Download dataset
            kaggle.api.dataset_download_files(
                kaggle_id, 
                path=str(dataset_dir), 
                unzip=True
            )
            
            logger.info(f"Dataset {dataset_name} downloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download dataset {dataset_name}: {e}")
            return False
    
    def download_all_datasets(self) -> Dict[str, bool]:
        """Download all configured datasets"""
        results = {}
        
        if not self.setup_kaggle_api():
            return {name: False for name in self.datasets.keys()}
        
        for dataset_name in self.datasets.keys():
            results[dataset_name] = self.download_dataset(dataset_name)
        
        return results
    
    def load_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Load a downloaded dataset into pandas DataFrame"""
        try:
            dataset_dir = self.raw_dir / dataset_name
            
            if not dataset_dir.exists():
                logger.error(f"Dataset {dataset_name} not found. Please download first.")
                return pd.DataFrame()
            
            # Find CSV files in the dataset directory
            csv_files = list(dataset_dir.glob("*.csv"))
            
            if not csv_files:
                logger.error(f"No CSV files found in {dataset_dir}")
                return pd.DataFrame()
            
            # Load the main training file (usually the largest)
            main_file = max(csv_files, key=lambda f: f.stat().st_size)
            logger.info(f"Loading dataset from: {main_file}")
            
            df = pd.read_csv(main_file)
            logger.info(f"Loaded dataset with shape: {df.shape}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            return pd.DataFrame()
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get information about a dataset"""
        try:
            df = self.load_dataset(dataset_name)
            
            if df.empty:
                return {}
            
            info = {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "sample_data": df.head().to_dict('records')
            }
            
            # Try to identify label column
            potential_label_cols = ['label', 'generated', 'ai_generated', 'target', 'class']
            label_col = None
            for col in potential_label_cols:
                if col in df.columns:
                    label_col = col
                    break
            
            if label_col:
                info["label_column"] = label_col
                info["label_distribution"] = df[label_col].value_counts().to_dict()
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get dataset info: {e}")
            return {}
    
    def list_available_datasets(self) -> Dict[str, Dict]:
        """List all available datasets and their status"""
        status = {}
        
        for dataset_name in self.datasets.keys():
            dataset_dir = self.raw_dir / dataset_name
            downloaded = dataset_dir.exists() and any(dataset_dir.glob("*.csv"))
            
            status[dataset_name] = {
                "downloaded": downloaded,
                "kaggle_id": self.datasets[dataset_name]["kaggle_id"],
                "local_path": str(dataset_dir) if downloaded else None
            }
        
        return status

# Convenience function
def setup_datasets() -> DatasetLoader:
    """Setup and return dataset loader instance"""
    loader = DatasetLoader()
    
    logger.info("Setting up datasets...")
    download_results = loader.download_all_datasets()
    
    for dataset_name, success in download_results.items():
        if success:
            logger.info(f"✅ {dataset_name} downloaded successfully")
        else:
            logger.warning(f"❌ {dataset_name} download failed")
    
    return loader

if __name__ == "__main__":
    # Test the dataset loader
    loader = setup_datasets()
    
    # Print dataset information
    for dataset_name in loader.datasets.keys():
        print(f"\n=== {dataset_name.upper()} ===")
        info = loader.get_dataset_info(dataset_name)
        if info:
            print(f"Shape: {info['shape']}")
            print(f"Columns: {info['columns']}")
            if 'label_distribution' in info:
                print(f"Label distribution: {info['label_distribution']}")
        else:
            print("Dataset not available")