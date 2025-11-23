"""
FastAPI REST API for Brain Tumor Classification
Production-ready endpoints with proper error handling and monitoring
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import numpy as np
import cv2
from PIL import Image
import io
import logging
from datetime import datetime
import traceback

from inference import ModelInference, validate_image_input, ValidationError


# Initialize FastAPI app
app = FastAPI(
    title="Brain Tumor Classification API",
    description="Medical AI API for brain tumor MRI classification with Grad-CAM explanations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global inference model
model_inference = None


# Response models
class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction: dict = Field(..., description="Prediction details including class and confidence")
    metadata: dict = Field(..., description="Metadata about the prediction")
    clinical_recommendation: str = Field(..., description="Clinical recommendation based on results")
    gradcam_available: bool = Field(default=False, description="Whether Grad-CAM visualization is available")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_version: str
    timestamp: str


class PerformanceResponse(BaseModel):
    """Performance statistics response"""
    total_inferences: int
    total_time_seconds: float
    average_inference_time_ms: float
    throughput_per_second: float


# Dependency injection
def get_model():
    """Dependency to get model instance"""
    global model_inference
    if model_inference is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model_inference


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global model_inference
    try:
        logger.info("Loading model...")
        model_inference = ModelInference(
            config_path="config.json",
            model_weights_path="best_weights_balanced.h5"
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        logger.error(traceback.format_exc())


@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Brain Tumor Classification API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model_inference is not None else "unhealthy",
        model_loaded=model_inference is not None,
        model_version=model_inference.config['model']['version'] if model_inference else "unknown",
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(..., description="MRI image file"),
    return_gradcam: bool = False,
    model: ModelInference = Depends(get_model)
):
    """
    Predict brain tumor class from MRI image
    
    Args:
        file: Uploaded MRI image (JPEG, PNG)
        return_gradcam: Whether to include Grad-CAM visualization
    
    Returns:
        Prediction results with clinical recommendations
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Validate image
        try:
            validate_image_input(image)
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Run inference
        result = model.predict(image, return_gradcam=return_gradcam)
        
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        
        # Add gradcam availability flag
        result['gradcam_available'] = 'gradcam' in result
        
        return PredictionResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(
    files: List[UploadFile] = File(..., description="Multiple MRI image files"),
    model: ModelInference = Depends(get_model)
):
    """
    Predict brain tumor classes for multiple MRI images
    
    Args:
        files: List of uploaded MRI images
    
    Returns:
        List of prediction results
    """
    try:
        if len(files) > 20:  # Limit batch size
            raise HTTPException(status_code=400, detail="Maximum 20 images per batch")
        
        results = []
        
        for idx, file in enumerate(files):
            try:
                # Read and process image
                contents = await file.read()
                nparr = np.frombuffer(contents, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is None:
                    results.append({
                        'filename': file.filename,
                        'error': 'Failed to decode image'
                    })
                    continue
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Validate and predict
                validate_image_input(image)
                result = model.predict(image, return_gradcam=False)
                
                result['filename'] = file.filename
                result['index'] = idx
                results.append(result)
            
            except Exception as e:
                results.append({
                    'filename': file.filename,
                    'index': idx,
                    'error': str(e)
                })
        
        return JSONResponse(content={'results': results, 'total': len(files)})
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/performance", response_model=PerformanceResponse)
async def get_performance_stats(model: ModelInference = Depends(get_model)):
    """
    Get model performance statistics
    
    Returns:
        Performance metrics including inference times and throughput
    """
    stats = model.get_performance_stats()
    return PerformanceResponse(**stats)


@app.get("/model/info")
async def get_model_info(model: ModelInference = Depends(get_model)):
    """
    Get model information and configuration
    
    Returns:
        Model metadata and configuration
    """
    return {
        'model': model.config['model'],
        'inference': model.config['inference'],
        'clinical': model.config['clinical']
    }


@app.get("/classes")
async def get_classes(model: ModelInference = Depends(get_model)):
    """
    Get available tumor classes
    
    Returns:
        List of tumor classes the model can detect
    """
    return {
        'classes': model.class_names,
        'descriptions': {
            'glioma': 'Tumor arising from glial cells',
            'meningioma': 'Tumor arising from meninges',
            'pituitary': 'Tumor in pituitary gland',
            'notumor': 'No tumor detected'
        }
    }


# Error handlers
@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"error": "Validation error", "detail": str(exc)}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
