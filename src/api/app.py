from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import sys
from typing import List, Optional
import logging
import time
import torch
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from models.bert_classifier import BERTAIDetector, BERTTrainer
except ImportError:
    print("BERT models not available, using dummy predictions")
    BERTAIDetector = None
    BERTTrainer = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Text Detection System",
    description="Detect AI-generated vs human-written text using fine-tuned BERT models",
    version="1.0.0"
)

# CORS middleware - Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class TextInput(BaseModel):
    text: str
    model_type: Optional[str] = "bert"

class PredictionResponse(BaseModel):
    prediction: str  # "AI-generated" or "Human-written"
    confidence: float
    model_used: str
    processing_time: float

class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]

# Global variables for models
models = {}
device = None

@app.on_event("startup")
async def load_models():
    """Load pre-trained models on startup"""
    global models, device
    
    logger.info("Starting AI detection system...")
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Try to load BERT model if available
    if BERTAIDetector:
        try:
            model_path = "./data/models/bert_ai_detector.pt"
            
            if Path(model_path).exists():
                logger.info("Loading trained BERT model...")
                model = BERTAIDetector(model_name="bert-base-uncased")
                trainer = BERTTrainer(model, device=device)
                trainer.load_model(model_path)
                models["bert"] = model
                logger.info("✅ Trained BERT model loaded successfully")
            else:
                logger.info("Loading fresh BERT model...")
                model = BERTAIDetector(model_name="bert-base-uncased")
                model.to(device)
                models["bert"] = model
                logger.info("✅ Fresh BERT model loaded")
        except Exception as e:
            logger.error(f"Failed to load BERT model: {e}")
            logger.info("Using dummy predictions")
    else:
        logger.info("BERT models not available, using dummy predictions")

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "AI Text Detection System API",
        "version": "1.0.0",
        "status": "running",
        "models_loaded": list(models.keys()),
        "endpoints": {
            "predict": "/predict",
            "predict_file": "/predict/file",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        models_loaded=list(models.keys())
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_text(input_data: TextInput):
    """Predict if text is AI-generated or human-written"""
    try:
        start_time = time.time()
        
        if "bert" in models:
            # Use actual BERT model
            model = models["bert"]
            prediction_int = model.predict([input_data.text], device=device)[0]
            probabilities = model.predict_proba([input_data.text], device=device)[0]
            
            prediction = "AI-generated" if prediction_int == 1 else "Human-written"
            confidence = float(max(probabilities))
        else:
            # Use dummy predictions
            import random
            prediction = random.choice(["AI-generated", "Human-written"])
            confidence = random.uniform(0.6, 0.95)
        
        processing_time = time.time() - start_time
        
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            model_used=input_data.model_type,
            processing_time=processing_time
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/file")
async def predict_file(file: UploadFile = File(...), model_type: str = "bert"):
    """Predict from uploaded file (placeholder)"""
    try:
        start_time = time.time()
        
        # Read file content
        content = await file.read()
        
        # For now, just extract text from plain text files
        if file.content_type == "text/plain":
            extracted_text = content.decode('utf-8')
        else:
            # Placeholder for other file types
            extracted_text = "File processing coming soon for this file type."
        
        # Use the same prediction logic as text
        if "bert" in models:
            model = models["bert"]
            prediction_int = model.predict([extracted_text], device=device)[0]
            probabilities = model.predict_proba([extracted_text], device=device)[0]
            
            prediction = "AI-generated" if prediction_int == 1 else "Human-written"
            confidence = float(max(probabilities))
        else:
            import random
            prediction = random.choice(["AI-generated", "Human-written"])
            confidence = random.uniform(0.6, 0.95)
        
        processing_time = time.time() - start_time
        
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            model_used=model_type,
            processing_time=processing_time
        )
    
    except Exception as e:
        logger.error(f"File prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
