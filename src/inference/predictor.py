"""
Predictor module for OTDR fault detection.
This module provides a unified interface for making predictions using the trained models.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

import sys
sys.path.append('./')

from src.preprocessing import OTDRDataProcessor
from src.training import AttentionLayer

class OTDRFaultPredictor:
    """
    Class for making predictions using trained OTDR fault detection models.
    """
    
    def __init__(self, autoencoder_model_path=None, bigru_model_path=None):
        """
        Initialize the predictor with trained models.
        
        Args:
            autoencoder_model_path (str): Path to the trained autoencoder model
            bigru_model_path (str): Path to the trained BiGRU-Attention model
        """
        self.autoencoder_model = None
        self.bigru_model = None
        self.data_processor = OTDRDataProcessor()
        self.anomaly_threshold = None
        self.class_mapping = {
            0: "normal",
            1: "fiber tapping",
            2: "bad splice",
            3: "bending event",
            4: "dirty connector",
            5: "fiber cut",
            6: "PC connector",
            7: "reflector"
        }
        
        # Load models if paths are provided
        if autoencoder_model_path and os.path.exists(autoencoder_model_path):
            self.load_autoencoder(autoencoder_model_path)
            
        if bigru_model_path and os.path.exists(bigru_model_path):
            self.load_bigru(bigru_model_path)
    
    def load_autoencoder(self, model_path, config_path=None):
        """
        Load the autoencoder model.
        
        Args:
            model_path (str): Path to the trained autoencoder model
            config_path (str): Path to the model configuration file
        """
        self.autoencoder_model = load_model(model_path)
        
        # Load threshold from config if available
        if config_path and os.path.exists(config_path):
            config = np.load(config_path, allow_pickle=True).item()
            self.anomaly_threshold = config.get('threshold')
        else:
            # Try to find config in the same directory
            config_path = os.path.join(os.path.dirname(model_path), 'model_config.npy')
            if os.path.exists(config_path):
                config = np.load(config_path, allow_pickle=True).item()
                self.anomaly_threshold = config.get('threshold')
    
    def load_bigru(self, model_path):
        """
        Load the BiGRU-Attention model.
        
        Args:
            model_path (str): Path to the trained BiGRU-Attention model
        """
        # Custom objects needed for loading
        custom_objects = {'AttentionLayer': AttentionLayer}
        
        self.bigru_model = load_model(model_path, custom_objects=custom_objects)
    
    def preprocess_trace(self, trace_data, denoise=True):
        """
        Preprocess OTDR trace data for prediction.
        
        Args:
            trace_data (numpy.ndarray): Raw OTDR trace data
            denoise (bool): Whether to apply denoising
            
        Returns:
            tuple: (preprocessed_data_autoencoder, preprocessed_data_bigru)
        """
        # Ensure trace_data is numpy array
        trace_data = np.array(trace_data)
        
        # Apply denoising if requested
        if denoise:
            trace_data = self.data_processor.denoise_signal(trace_data)
        
        # Prepare data for autoencoder (2D format)
        data_autoencoder = trace_data.copy()
        if data_autoencoder.ndim == 1:
            data_autoencoder = data_autoencoder.reshape(1, -1)
        
        # Prepare data for BiGRU (3D format)
        data_bigru = trace_data.copy()
        if data_bigru.ndim == 1:
            # Single trace, reshape to (1, sequence_length, 1)
            data_bigru = data_bigru.reshape(1, -1, 1)
        elif data_bigru.ndim == 2:
            # Multiple traces, reshape to (samples, sequence_length, 1)
            data_bigru = data_bigru.reshape(data_bigru.shape[0], data_bigru.shape[1], 1)
        
        return data_autoencoder, data_bigru
    
    def detect_anomaly(self, trace_data, denoise=True):
        """
        Detect anomalies in OTDR trace data using the autoencoder model.
        
        Args:
            trace_data (numpy.ndarray): OTDR trace data
            denoise (bool): Whether to apply denoising
            
        Returns:
            dict: Anomaly detection results
        """
        if self.autoencoder_model is None:
            raise ValueError("Autoencoder model not loaded")
            
        if self.anomaly_threshold is None:
            raise ValueError("Anomaly threshold not set")
        
        # Preprocess data
        data_autoencoder, _ = self.preprocess_trace(trace_data, denoise)
        
        # Get reconstructions
        reconstructions = self.autoencoder_model.predict(data_autoencoder)
        
        # Compute reconstruction errors (MSE)
        mse = np.mean(np.square(data_autoencoder - reconstructions), axis=1)
        
        # Classify as anomaly if error > threshold
        is_anomaly = (mse > self.anomaly_threshold).astype(int)
        
        results = {
            'reconstruction_error': mse.tolist(),
            'threshold': self.anomaly_threshold,
            'is_anomaly': is_anomaly.tolist()
        }
        
        return results
    
    def diagnose_fault(self, trace_data, include_snr=True, denoise=True):
        """
        Diagnose fault type and location using the BiGRU-Attention model.
        
        Args:
            trace_data (numpy.ndarray): OTDR trace data
            include_snr (bool): Whether to include SNR as additional input
            denoise (bool): Whether to apply denoising
            
        Returns:
            dict: Fault diagnosis results
        """
        if self.bigru_model is None:
            raise ValueError("BiGRU model not loaded")
        
        # Preprocess data
        _, data_bigru = self.preprocess_trace(trace_data, denoise)
        
        # Prepare inputs based on model architecture
        if include_snr:
            # Extract SNR values (assuming they're stored separately or in the first column)
            snr = data_bigru[:, 0, 0].reshape(-1, 1)  # Reshape to (samples, 1)
            model_inputs = [data_bigru, snr]
        else:
            model_inputs = data_bigru
        
        # Get model predictions
        class_probs, positions = self.bigru_model.predict(model_inputs)
        
        # Convert class probabilities to class indices
        class_indices = np.argmax(class_probs, axis=1)
        
        # Map class indices to class names
        class_names = [self.class_mapping.get(idx, "unknown") for idx in class_indices]
        
        results = {
            'class_index': class_indices.tolist(),
            'class_name': class_names,
            'class_probabilities': class_probs.tolist(),
            'position': positions.flatten().tolist()
        }
        
        return results
    
    def predict(self, trace_data, denoise=True):
        """
        Make complete predictions using both models.
        
        Args:
            trace_data (numpy.ndarray): OTDR trace data
            denoise (bool): Whether to apply denoising
            
        Returns:
            dict: Complete prediction results
        """
        # Detect anomalies
        anomaly_results = self.detect_anomaly(trace_data, denoise)

        if isinstance(trace_data, np.ndarray) and trace_data.ndim == 1:
         trace_data = trace_data.reshape(1, -1)
         
        # If anomaly detected, diagnose fault
        if any(anomaly_results['is_anomaly']):
            fault_results = self.diagnose_fault(trace_data, denoise=denoise)
        else:
            fault_results = {
                'class_index': [0],  # Normal
                'class_name': ["normal"],
                'class_probabilities': [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                'position': [0.0]
            }
        
        # Combine results
        results = {
            'anomaly_detection': anomaly_results,
            'fault_diagnosis': fault_results
        }
        
        return results
