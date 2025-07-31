import pandas as pd
import numpy as np
import re
import nltk
import spacy
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from typing import List, Tuple, Dict, Any, Optional
import logging
from pathlib import Path
from .dataset_loader import DatasetLoader

logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Text preprocessing pipeline for AI detection"""
    
    def __init__(self, model_name: str = "bert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = None
        self.nlp = None
        self._setup_nltk()
        self._setup_spacy()
        self._setup_tokenizer()
    
    def _setup_nltk(self):
        """Download required NLTK data"""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            logger.info("NLTK data downloaded successfully")
        except Exception as e:
            logger.warning(f"NLTK setup warning: {e}")
    
    def _setup_spacy(self):
        """Setup spaCy model"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded successfully")
        except Exception as e:
            logger.warning(f"spaCy setup warning: {e}")
    
    def _setup_tokenizer(self):
        """Setup transformer tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            logger.info(f"Tokenizer loaded: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
    
    def clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        if not isinstance(text, str):
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\'\"]', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def extract_features(self, text: str) -> Dict[str, float]:
        """Extract linguistic features from text"""
        if not text:
            return {}
        
        features = {}
        
        # Basic statistics
        features['char_count'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(nltk.sent_tokenize(text))
        features['avg_word_length'] = np.mean([len(word) for word in text.split()])
        features['avg_sentence_length'] = features['word_count'] / max(features['sentence_count'], 1)
        
        # Punctuation statistics
        features['punct_count'] = len(re.findall(r'[^\w\s]', text))
        features['punct_ratio'] = features['punct_count'] / max(features['char_count'], 1)
        
        # Case statistics
        features['uppercase_ratio'] = len(re.findall(r'[A-Z]', text)) / max(features['char_count'], 1)
        features['digit_ratio'] = len(re.findall(r'\d', text)) / max(features['char_count'], 1)
        
        # Advanced features using spaCy (if available)
        if self.nlp:
            try:
                doc = self.nlp(text[:1000000])  # Limit to 1M chars for spaCy
                
                # POS tag distribution
                pos_tags = [token.pos_ for token in doc]
                features['noun_ratio'] = pos_tags.count('NOUN') / max(len(pos_tags), 1)
                features['verb_ratio'] = pos_tags.count('VERB') / max(len(pos_tags), 1)
                features['adj_ratio'] = pos_tags.count('ADJ') / max(len(pos_tags), 1)
                
                # Named entities
                features['entity_count'] = len(doc.ents)
                features['entity_ratio'] = features['entity_count'] / max(features['word_count'], 1)
                
            except Exception as e:
                logger.warning(f"spaCy feature extraction failed: {e}")
        
        return features
    
    def tokenize_for_bert(self, texts: List[str], max_length: int = 512) -> Dict[str, Any]:
        """Tokenize texts for BERT model"""
        if not self.tokenizer:
            raise ValueError("Tokenizer not initialized")
        
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
    
    def preprocess_dataset(self, df: pd.DataFrame, text_column: str, 
                          label_column: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Complete preprocessing pipeline for a dataset"""
        logger.info(f"Preprocessing dataset with {len(df)} samples")
        
        # Make a copy
        processed_df = df.copy()
        
        # Clean text
        processed_df['cleaned_text'] = processed_df[text_column].apply(self.clean_text)
        
        # Remove empty texts
        processed_df = processed_df[processed_df['cleaned_text'].str.len() > 0]
        
        # Extract features
        logger.info("Extracting linguistic features...")
        features_list = []
        for text in processed_df['cleaned_text']:
            features_list.append(self.extract_features(text))
        
        # Convert features to DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Combine with original data
        processed_df = pd.concat([processed_df.reset_index(drop=True), 
                                features_df.reset_index(drop=True)], axis=1)
        
        # Standardize labels
        processed_df['label_binary'] = self._standardize_labels(processed_df[label_column])
        
        # Dataset statistics
        stats = {
            'original_size': len(df),
            'processed_size': len(processed_df),
            'removed_samples': len(df) - len(processed_df),
            'label_distribution': processed_df['label_binary'].value_counts().to_dict(),
            'feature_columns': features_df.columns.tolist(),
            'avg_text_length': processed_df['cleaned_text'].str.len().mean()
        }
        
        logger.info(f"Preprocessing complete. Stats: {stats}")
        return processed_df, stats
    
    def _standardize_labels(self, labels: pd.Series) -> pd.Series:
        """Standardize labels to binary format (0=human, 1=ai)"""
        # Handle different label formats
        labels_lower = labels.astype(str).str.lower()
        
        # Map various label formats to binary
        label_mapping = {
            '0': 0, '1': 1,
            'human': 0, 'ai': 1,
            'human-written': 0, 'ai-generated': 1,
            'real': 0, 'fake': 1,
            'original': 0, 'generated': 1,
            'false': 0, 'true': 1
        }
        
        binary_labels = labels_lower.map(label_mapping)
        
        # Check for unmapped labels
        unmapped = binary_labels.isnull().sum()
        if unmapped > 0:
            logger.warning(f"Found {unmapped} unmapped labels. Setting to 0 (human)")
            binary_labels = binary_labels.fillna(0)
        
        return binary_labels.astype(int)
    
    def create_train_val_split(self, df: pd.DataFrame, 
                              test_size: float = 0.2, 
                              val_size: float = 0.1,
                              random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create train/validation/test splits"""
        
        # First split: train+val vs test
        train_val, test = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state,
            stratify=df['label_binary']
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val,
            test_size=val_ratio,
            random_state=random_state,
            stratify=train_val['label_binary']
        )
        
        logger.info(f"Dataset split - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        
        return train, val, test
    
    def save_processed_dataset(self, df: pd.DataFrame, filename: str):
        """Save processed dataset"""
        filepath = self.processed_dir / filename
        df.to_csv(filepath, index=False)
        logger.info(f"Saved processed dataset to: {filepath}")

class DatasetManager:
    """High-level manager for dataset operations"""
    
    def __init__(self, data_dir: str = "./data"):
        self.loader = DatasetLoader(data_dir)
        self.preprocessor = TextPreprocessor()
    
    def setup_all_datasets(self) -> Dict[str, Any]:
        """Download and preprocess all datasets"""
        # Download datasets
        download_results = self.loader.download_all_datasets()
        
        results = {
            'download_results': download_results,
            'processed_datasets': {}
        }
        
        # Process each successfully downloaded dataset
        for dataset_name, downloaded in download_results.items():
            if downloaded:
                try:
                    # Load raw dataset
                    df = self.loader.load_dataset(dataset_name)
                    
                    if not df.empty:
                        # Detect text and label columns
                        text_col, label_col = self._detect_columns(df)
                        
                        if text_col and label_col:
                            # Preprocess
                            processed_df, stats = self.preprocessor.preprocess_dataset(
                                df, text_col, label_col
                            )
                            
                            # Create splits
                            train, val, test = self.preprocessor.create_train_val_split(processed_df)
                            
                            # Save processed datasets
                            self.preprocessor.save_processed_dataset(train, f"{dataset_name}_train.csv")
                            self.preprocessor.save_processed_dataset(val, f"{dataset_name}_val.csv")
                            self.preprocessor.save_processed_dataset(test, f"{dataset_name}_test.csv")
                            
                            results['processed_datasets'][dataset_name] = {
                                'stats': stats,
                                'text_column': text_col,
                                'label_column': label_col,
                                'splits': {
                                    'train': len(train),
                                    'val': len(val), 
                                    'test': len(test)
                                }
                            }
                        else:
                            logger.error(f"Could not detect text/label columns for {dataset_name}")
                
                except Exception as e:
                    logger.error(f"Failed to process dataset {dataset_name}: {e}")
        
        return results
    
    def _detect_columns(self, df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
        """Auto-detect text and label columns"""
        text_col = None
        label_col = None
        
        # Detect text column (longest average string length)
        text_candidates = []
        for col in df.columns:
            if df[col].dtype == 'object':
                avg_length = df[col].astype(str).str.len().mean()
                if avg_length > 50:  # Assume text columns have longer content
                    text_candidates.append((col, avg_length))
        
        if text_candidates:
            text_col = max(text_candidates, key=lambda x: x[1])[0]
        
        # Detect label column
        label_candidates = ['label', 'generated', 'ai_generated', 'target', 'class', 'is_ai']
        for col in label_candidates:
            if col in df.columns:
                label_col = col
                break
        
        # If no exact match, look for binary columns
        if not label_col:
            for col in df.columns:
                if df[col].dtype in ['int64', 'bool'] and df[col].nunique() == 2:
                    label_col = col
                    break
        
        logger.info(f"Detected columns - Text: {text_col}, Label: {label_col}")
        return text_col, label_col

if __name__ == "__main__":
    # Test the pipeline
    manager = DatasetManager()
    results = manager.setup_all_datasets()
    print("Dataset setup results:", results)