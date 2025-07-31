import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class AIDetectionDataset(Dataset):
    """Dataset class for AI text detection"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BERTAIDetector(nn.Module):
    """BERT-based AI text detection model"""
    
    def __init__(self, model_name: str = "bert-base-uncased", num_classes: int = 2, dropout_rate: float = 0.3):
        super(BERTAIDetector, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load BERT model and tokenizer
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classifier weights"""
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids, attention_mask):
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output
        
        # Apply dropout and classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits
    
    def predict_proba(self, texts: List[str], device: str = 'cpu') -> np.ndarray:
        """Predict probabilities for a list of texts"""
        self.eval()
        
        predictions = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=512,
                    return_tensors='pt'
                )
                
                # Move to device
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)
                
                # Forward pass
                logits = self(input_ids, attention_mask)
                probabilities = torch.softmax(logits, dim=-1)
                
                predictions.append(probabilities.cpu().numpy())
        
        return np.vstack(predictions)
    
    def predict(self, texts: List[str], device: str = 'cpu') -> List[int]:
        """Predict classes for a list of texts"""
        probabilities = self.predict_proba(texts, device)
        return np.argmax(probabilities, axis=1).tolist()

class BERTTrainer:
    """Trainer class for BERT AI detection model"""
    
    def __init__(self, model: BERTAIDetector, device: str = None):
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        logger.info(f"Using device: {self.device}")
    
    def create_data_loader(self, df: pd.DataFrame, batch_size: int = 16, shuffle: bool = True) -> DataLoader:
        """Create DataLoader from DataFrame"""
        texts = df['cleaned_text'].tolist()
        labels = df['label_binary'].tolist()
        
        dataset = AIDetectionDataset(
            texts=texts,
            labels=labels,
            tokenizer=self.model.tokenizer
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0  # Set to 0 for Docker compatibility
        )
    
    def train_epoch(self, data_loader: DataLoader, optimizer, scheduler) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in data_loader:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(input_ids, attention_mask)
            
            # Calculate loss
            loss = nn.CrossEntropyLoss()(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
        
        return total_loss / len(data_loader)
    
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation/test set"""
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                logits = self.model(input_ids, attention_mask)
                
                # Calculate loss
                loss = nn.CrossEntropyLoss()(logits, labels)
                total_loss += loss.item()
                
                # Get predictions
                predictions = torch.argmax(logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        return {
            'loss': total_loss / len(data_loader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
              epochs: int = 3, batch_size: int = 16, learning_rate: float = 2e-5,
              save_path: str = "./data/models/bert_ai_detector.pt") -> Dict[str, List[float]]:
        """Complete training loop"""
        
        # Create data loaders
        train_loader = self.create_data_loader(train_df, batch_size, shuffle=True)
        val_loader = self.create_data_loader(val_df, batch_size, shuffle=False)
        
        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': []
        }
        
        best_f1 = 0
        
        logger.info(f"Starting training for {epochs} epochs...")
        logger.info(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")
        
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader, optimizer, scheduler)
            
            # Evaluate
            val_metrics = self.evaluate(val_loader)
            
            # Log metrics
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
            logger.info(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            logger.info(f"Val F1: {val_metrics['f1']:.4f}")
            
            # Save metrics
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_metrics['loss'])
            history['val_accuracy'].append(val_metrics['accuracy'])
            history['val_f1'].append(val_metrics['f1'])
            
            # Save best model
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                self.save_model(save_path)
                logger.info(f"New best model saved with F1: {best_f1:.4f}")
        
        logger.info("Training completed!")
        return history
    
    def save_model(self, path: str):
        """Save model state"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model.model_name,
            'tokenizer': self.model.tokenizer
        }, path)
        
        logger.info(f"Model saved to: {path}")
    
    def load_model(self, path: str):
        """Load model state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from: {path}")

def create_bert_model(model_name: str = "bert-base-uncased") -> BERTAIDetector:
    """Factory function to create BERT model"""
    return BERTAIDetector(model_name=model_name)

def train_bert_model(train_path: str, val_path: str, 
                    model_name: str = "bert-base-uncased",
                    epochs: int = 3, batch_size: int = 16) -> BERTAIDetector:
    """Train BERT model from CSV files"""
    
    # Load data
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    logger.info(f"Loaded training data: {train_df.shape}")
    logger.info(f"Loaded validation data: {val_df.shape}")
    
    # Create model and trainer
    model = create_bert_model(model_name)
    trainer = BERTTrainer(model)
    
    # Train
    history = trainer.train(
        train_df=train_df,
        val_df=val_df,
        epochs=epochs,
        batch_size=batch_size
    )
    
    return model, history

if __name__ == "__main__":
    # Test the BERT classifier
    train_path = "./data/processed/ai_detection_train.csv"
    val_path = "./data/processed/ai_detection_val.csv"
    
    if Path(train_path).exists() and Path(val_path).exists():
        model, history = train_bert_model(train_path, val_path, epochs=2)
        print("Training completed!")
        print("History:", history)
    else:
        print("Processed datasets not found. Run data pipeline first.")