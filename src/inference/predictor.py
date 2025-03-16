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
        if autoencoder_model_path:
            self.load_autoencoder(autoencoder_model_path)
        
        if bigru_model_path:
            self.load_bigru(bigru_model_path)
    
    def load_autoencoder(self, model_path):
        """
        Load the autoencoder model and its anomaly threshold.
        
        Args:
            model_path (str): Path to the trained autoencoder model
        """
        try:
            self.autoencoder_model = load_model(model_path)
            config_path = os.path.join(os.path.dirname(model_path), 'model_config.npy')

            if os.path.exists(config_path):
                config = np.load(config_path, allow_pickle=True).item()
                self.anomaly_threshold = config.get('threshold', None)

        except Exception as e:
            raise ValueError(f"Failed to load autoencoder model: {e}")
    
    def load_bigru(self, model_path):
        """
        Load the BiGRU-Attention model.
        
        Args:
            model_path (str): Path to the trained BiGRU-Attention model
        """
        try:
            self.bigru_model = load_model(model_path, custom_objects={'AttentionLayer': AttentionLayer})
        except Exception as e:
            raise ValueError(f"Failed to load BiGRU model: {e}")
    
    def preprocess_trace(self, trace_data, denoise=True):
        """
        Preprocess OTDR trace data for prediction.
    
        Args:
            trace_data (numpy.ndarray): Raw OTDR trace data
            denoise (bool): Whether to apply denoising
        
        Returns:
            tuple: (preprocessed_data_autoencoder, preprocessed_data_bigru, snr)
        """
        trace_data = np.asarray(trace_data)

        if denoise:
            trace_data = self.data_processor.denoise_signal(trace_data)
        
        if trace_data.ndim == 1:
            snr = trace_data[0]
            trace_points = trace_data[1:].reshape(1, -1, 1)
        else:
            snr = trace_data[:, 0]
            trace_points = trace_data[:, 1:].reshape(trace_data.shape[0], -1, 1)
        
        return trace_data, trace_points, snr
    
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
            raise ValueError("Autoencoder model not loaded.")
        
        if self.anomaly_threshold is None:
            raise ValueError("Anomaly threshold not set.")
        
        trace_data = np.asarray(trace_data)
        if denoise:
            trace_data = self.data_processor.denoise_signal(trace_data)
        
        data_autoencoder = trace_data.reshape(1, -1) if trace_data.ndim == 1 else trace_data
        reconstructions = self.autoencoder_model.predict(data_autoencoder)
        mse = np.mean(np.square(data_autoencoder - reconstructions), axis=1)
        is_anomaly = (mse > self.anomaly_threshold).astype(int)
        
        return {
            'reconstruction_error': mse.tolist(),
            'threshold': self.anomaly_threshold,
            'is_anomaly': is_anomaly.tolist()
        }
    
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
            raise ValueError("BiGRU model not loaded.")
        
        trace_data, trace_points_3d, snr = self.preprocess_trace(trace_data, denoise)

        model_inputs = [trace_points_3d, np.array([[snr]])] if include_snr else trace_points_3d
        class_probs, positions = self.bigru_model.predict(model_inputs)
        
        class_indices = np.argmax(class_probs, axis=1)
        class_names = [self.class_mapping.get(idx, "unknown") for idx in class_indices]
        
        return {
            'class_index': class_indices.tolist(),
            'class_name': class_names,
            'class_probabilities': class_probs.tolist(),
            'position': positions.flatten().tolist()
        }
    
    def predict(self, trace_data, denoise=True):
        """
        Make complete predictions using both models.
        
        Args:
            trace_data (numpy.ndarray): OTDR trace data
            denoise (bool): Whether to apply denoising
            
        Returns:
            dict: Complete prediction results
        """
        trace_data = np.asarray(trace_data)
        anomaly_results = self.detect_anomaly(trace_data, denoise)
        
        if any(anomaly_results['is_anomaly']):
            fault_results = self.diagnose_fault(trace_data, denoise=denoise)
        else:
            fault_results = {
                'class_index': [0],
                'class_name': ["normal"],
                'class_probabilities': [[1.0] + [0.0] * 7],
                'position': [0.0]
            }
        
        return {
            'anomaly_detection': anomaly_results,
            'fault_diagnosis': fault_results
        }
