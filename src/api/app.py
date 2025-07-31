from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import sys
from typing import List, Optional
import logging

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Text Detection System",
    description="Detect AI-generated vs human-written text using fine-tuned BERT models",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://frontend:3000"],
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

# Global variables for models (will be loaded on startup)
models = {}

@app.on_event("startup")
async def load_models():
    """Load pre-trained models on startup"""
    logger.info("Loading AI detection models...")
    # TODO: Implement model loading
    # models["bert"] = load_bert_model()
    # models["roberta"] = load_roberta_model()
    logger.info("Models loaded successfully")

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "AI Text Detection System API",
        "version": "1.0.0",
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
        import time
        start_time = time.time()
        
        # TODO: Implement actual prediction logic
        # For now, return dummy response
        prediction = "Human-written"  # Placeholder
        confidence = 0.85  # Placeholder
        
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

@app.post("/predict/file", response_model=PredictionResponse)
async def predict_file(file: UploadFile = File(...), model_type: str = "bert"):
    """Predict from uploaded file (PDF, TXT, DOCX, or image)"""
    try:
        import time
        start_time = time.time()
        
        # Check file type
        allowed_types = {
            "text/plain": "txt",
            "application/pdf": "pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
            "image/jpeg": "jpg",
            "image/png": "png"
        }
        
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file.content_type}"
            )
        
        # Read file content
        content = await file.read()
        
        # TODO: Implement text extraction based on file type
        # extracted_text = extract_text_from_file(content, file.content_type)
        
        # TODO: Implement actual prediction
        prediction = "AI-generated"  # Placeholder
        confidence = 0.72  # Placeholder
        
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

@app.get("/models")
async def list_available_models():
    """List available models"""
    return {
        "available_models": ["bert", "roberta", "distilbert"],
        "loaded_models": list(models.keys()),
        "default_model": "bert"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)