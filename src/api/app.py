"""
FastAPI application for OTDR fault detection service.
This module implements the REST API endpoints for the OTDR fault detection service.
"""

import os
import numpy as np
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import logging
import json
from datetime import datetime

import sys
sys.path.append('/home/ubuntu/ftth_fault_detection')

from src.inference import OTDRFaultPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('api.log')
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="FTTH Fiber Optic Fault Detection API",
    description="API for detecting, diagnosing, and localizing faults in FTTH fiber optic cables using OTDR traces",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model paths (to be configured via environment variables in production)
AUTOENCODER_MODEL_PATH = os.environ.get(
    "AUTOENCODER_MODEL_PATH", 
    "/home/ubuntu/ftth_fault_detection/models/autoencoder_latest/autoencoder_model.h5"
)
BIGRU_MODEL_PATH = os.environ.get(
    "BIGRU_MODEL_PATH", 
    "/home/ubuntu/ftth_fault_detection/models/bigru_attention_latest/bigru_attention_model.h5"
)

# Initialize predictor
predictor = None

# Pydantic models for request/response validation
class OTDRTrace(BaseModel):
    snr: float
    trace_points: List[float]
    
class PredictionResponse(BaseModel):
    anomaly_detected: bool
    fault_type: Optional[str] = None
    fault_location: Optional[float] = None
    confidence: Optional[float] = None
    reconstruction_error: Optional[float] = None
    raw_predictions: Optional[Dict[str, Any]] = None

@app.on_event("startup")
async def startup_event():
    """Initialize the predictor on startup."""
    global predictor
    try:
        logger.info("Loading models...")
        predictor = OTDRFaultPredictor(
            autoencoder_model_path=AUTOENCODER_MODEL_PATH,
            bigru_model_path=BIGRU_MODEL_PATH
        )
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        # Continue without models, they will be loaded on first request

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "FTTH Fiber Optic Fault Detection API"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    global predictor
    
    if predictor is None:
        try:
            predictor = OTDRFaultPredictor(
                autoencoder_model_path=AUTOENCODER_MODEL_PATH,
                bigru_model_path=BIGRU_MODEL_PATH
            )
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": f"Models not loaded: {str(e)}"}
            )
    
    return {"status": "healthy", "models_loaded": predictor is not None}

@app.post("/predict", response_model=PredictionResponse)
async def predict(trace: OTDRTrace):
    """
    Predict fault from OTDR trace data.
    
    Args:
        trace (OTDRTrace): OTDR trace data with SNR and trace points
        
    Returns:
        PredictionResponse: Prediction results
    """
    global predictor
    
    # Initialize predictor if not already done
    if predictor is None:
        try:
            predictor = OTDRFaultPredictor(
                autoencoder_model_path=AUTOENCODER_MODEL_PATH,
                bigru_model_path=BIGRU_MODEL_PATH
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading models: {str(e)}")
    
    try:
        # Prepare input data
        trace_data = np.array([trace.snr] + trace.trace_points)
        
        # Make prediction
        results = predictor.predict(trace_data)
        
        # Extract relevant information
        anomaly_detected = any(results['anomaly_detection']['is_anomaly'])
        reconstruction_error = results['anomaly_detection']['reconstruction_error'][0]
        
        if anomaly_detected:
            fault_type = results['fault_diagnosis']['class_name'][0]
            fault_location = results['fault_diagnosis']['position'][0]
            
            # Get confidence from class probabilities
            class_index = results['fault_diagnosis']['class_index'][0]
            confidence = results['fault_diagnosis']['class_probabilities'][0][class_index]
        else:
            fault_type = "normal"
            fault_location = None
            confidence = None
        
        # Prepare response
        response = PredictionResponse(
            anomaly_detected=anomaly_detected,
            fault_type=fault_type,
            fault_location=fault_location,
            confidence=confidence,
            reconstruction_error=reconstruction_error,
            raw_predictions=results
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """
    Process batch predictions from CSV file.
    
    Args:
        file (UploadFile): CSV file with OTDR trace data
        background_tasks (BackgroundTasks): Background tasks for async processing
        
    Returns:
        dict: Job information or results
    """
    global predictor
    
    # Initialize predictor if not already done
    if predictor is None:
        try:
            predictor = OTDRFaultPredictor(
                autoencoder_model_path=AUTOENCODER_MODEL_PATH,
                bigru_model_path=BIGRU_MODEL_PATH
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading models: {str(e)}")
    
    try:
        # Read CSV file
        df = pd.read_csv(file.file)
        
        # Check if file has expected columns
        expected_columns = ['SNR'] + [f'P{i}' for i in range(1, 31)]
        missing_columns = [col for col in expected_columns if col not in df.columns]
        
        if missing_columns:
            raise HTTPException(
                status_code=400, 
                detail=f"CSV file missing required columns: {', '.join(missing_columns)}"
            )
        
        # Process each row
        results = []
        for _, row in df.iterrows():
            # Prepare input data
            trace_data = np.array([row['SNR']] + [row[f'P{i}'] for i in range(1, 31)])
            
            # Make prediction
            prediction = predictor.predict(trace_data)
            
            # Extract relevant information
            anomaly_detected = any(prediction['anomaly_detection']['is_anomaly'])
            reconstruction_error = prediction['anomaly_detection']['reconstruction_error'][0]
            
            if anomaly_detected:
                fault_type = prediction['fault_diagnosis']['class_name'][0]
                fault_location = prediction['fault_diagnosis']['position'][0]
                
                # Get confidence from class probabilities
                class_index = prediction['fault_diagnosis']['class_index'][0]
                confidence = prediction['fault_diagnosis']['class_probabilities'][0][class_index]
            else:
                fault_type = "normal"
                fault_location = None
                confidence = None
            
            # Add to results
            results.append({
                'anomaly_detected': anomaly_detected,
                'fault_type': fault_type,
                'fault_location': fault_location,
                'confidence': confidence,
                'reconstruction_error': reconstruction_error
            })
        
        return {"results": results}
    
    except Exception as e:
        logger.error(f"Error processing batch: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing batch: {str(e)}")

@app.post("/models/reload")
async def reload_models():
    """Reload models from disk."""
    global predictor
    
    try:
        predictor = OTDRFaultPredictor(
            autoencoder_model_path=AUTOENCODER_MODEL_PATH,
            bigru_model_path=BIGRU_MODEL_PATH
        )
        return {"status": "success", "message": "Models reloaded successfully"}
    except Exception as e:
        logger.error(f"Error reloading models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error reloading models: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
