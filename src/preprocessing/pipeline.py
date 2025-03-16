"""
Preprocessing pipeline for OTDR trace analysis.
This module provides a complete pipeline for preprocessing OTDR data.
"""

import os
import numpy as np
import pandas as pd
from .data_loader import OTDRDataLoader
from .data_processor import OTDRDataProcessor

class OTDRPreprocessingPipeline:
    """
    End-to-end preprocessing pipeline for OTDR data.
    """
    
    def __init__(self, data_path, random_state=42):
        """
        Initialize the preprocessing pipeline.
        
        Args:
            data_path (str): Path to the OTDR data CSV file
            random_state (int): Random seed for reproducibility
        """
        self.data_path = data_path
        self.random_state = random_state
        self.data_loader = OTDRDataLoader(data_path, random_state)
        self.data_processor = OTDRDataProcessor()
        
    def prepare_data_for_anomaly_detection(self, denoise=True, balance_classes=False):
        """
        Prepare data for autoencoder-based anomaly detection.
        
        Args:
            denoise (bool): Whether to apply denoising
            balance_classes (bool): Whether to balance classes
            
        Returns:
            dict: Dictionary containing prepared datasets
        """
        # Load and split data
        self.data_loader.load_data()
        X_train_normal, X_val, X_test, y_test = self.data_loader.prepare_data_for_autoencoder()
        
        # Apply denoising if requested
        if denoise:
            X_train_normal = self.data_processor.denoise_signal(X_train_normal)
            X_val = self.data_processor.denoise_signal(X_val)
            X_test = self.data_processor.denoise_signal(X_test)
        
        # Prepare data for autoencoder (reshape if needed)
        # For autoencoder, we keep the data in 2D format (samples, features)
        
        return {
            'X_train': X_train_normal,
            'X_val': X_val,
            'X_test': X_test,
            'y_test': y_test
        }
    
    def prepare_data_for_classification(self, denoise=True, balance_classes=True, augment_data=True):
        """
        Prepare data for fault classification and localization.
        
        Args:
            denoise (bool): Whether to apply denoising
            balance_classes (bool): Whether to balance classes
            augment_data (bool): Whether to augment training data
            
        Returns:
            dict: Dictionary containing prepared datasets
        """
        # Load and split data
        self.data_loader.load_data()
        X_train, X_val, X_test, y_train, y_val, y_test = self.data_loader.prepare_data_for_classification()
        
        # Apply denoising if requested
        if denoise:
            X_train = self.data_processor.denoise_signal(X_train)
            X_val = self.data_processor.denoise_signal(X_val)
            X_test = self.data_processor.denoise_signal(X_test)
        
        # Balance classes if requested
        if balance_classes:
            X_train, y_train = self.data_processor.balance_classes(X_train, y_train)
        
        # Augment training data if requested
        if augment_data:
            X_train_aug, y_train_aug = self.data_processor.augment_data(X_train, y_train)
            X_train = np.vstack([X_train, X_train_aug])
            if isinstance(y_train, pd.DataFrame):
                y_train = pd.concat([y_train, y_train_aug], ignore_index=True)
            else:
                y_train = np.vstack([y_train, y_train_aug])
        
        # Encode class labels
        y_train_class = y_train['Class'].values
        y_val_class = y_val['Class'].values
        y_test_class = y_test['Class'].values
        
        y_train_encoded = self.data_processor.encode_classes(y_train_class)
        y_val_encoded = self.data_processor.encode_classes(y_val_class)
        y_test_encoded = self.data_processor.encode_classes(y_test_class)
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'y_train_encoded': y_train_encoded,
            'y_val_encoded': y_val_encoded,
            'y_test_encoded': y_test_encoded
        }
    
    def prepare_data_for_rnn(self, denoise=True, balance_classes=True):
        """
        Prepare sequence data for RNN models (BiGRU).
        
        Args:
            denoise (bool): Whether to apply denoising
            balance_classes (bool): Whether to balance classes
            
        Returns:
            dict: Dictionary containing prepared datasets
        """
        # Load and split data
        self.data_loader.load_data()
        X_train_seq, X_val_seq, X_test_seq,snr_values, y_train, y_val, y_test = self.data_loader.prepare_sequences_for_rnn()
        
        # Apply denoising if requested
        if denoise:
            X_train_seq = self.data_processor.denoise_signal(X_train_seq)
            X_val_seq = self.data_processor.denoise_signal(X_val_seq)
            X_test_seq = self.data_processor.denoise_signal(X_test_seq)
        
        # Balance classes if requested
        if balance_classes:
            # Reshape sequences to 2D for balancing
            samples_train, time_steps, features = X_train_seq.shape
            X_train_2d = X_train_seq.reshape(samples_train, time_steps * features)
            
            X_train_balanced_2d, y_train_balanced = self.data_processor.balance_classes(X_train_2d, y_train)
            
            # Reshape back to 3D
            X_train_seq = X_train_balanced_2d.reshape(-1, time_steps, features)
            y_train = y_train_balanced
        
        # Encode class labels
        y_train_class = y_train['Class'].values
        y_val_class = y_val['Class'].values
        y_test_class = y_test['Class'].values
        
        y_train_encoded = self.data_processor.encode_classes(y_train_class)
        y_val_encoded = self.data_processor.encode_classes(y_val_class)
        y_test_encoded = self.data_processor.encode_classes(y_test_class)
        
        # Extract position values for localization
        y_train_pos = y_train['Position'].values.reshape(-1, 1)
        y_val_pos = y_val['Position'].values.reshape(-1, 1)
        y_test_pos = y_test['Position'].values.reshape(-1, 1)
        
        return {
            'X_train': X_train_seq,
            'X_val': X_val_seq,
            'X_test': X_test_seq,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'y_train_encoded': y_train_encoded,
            'y_val_encoded': y_val_encoded,
            'y_test_encoded': y_test_encoded,
            'y_train_pos': y_train_pos,
            'y_val_pos': y_val_pos,
            'y_test_pos': y_test_pos
        }
    
    def save_processed_data(self, output_dir, data_dict):
        """
        Save processed data to files.
        
        Args:
            output_dir (str): Directory to save processed data
            data_dict (dict): Dictionary containing datasets to save
            
        Returns:
            dict: Dictionary with paths to saved files
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = {}
        
        for key, data in data_dict.items():
            file_path = os.path.join(output_dir, f"{key}.npy")
            np.save(file_path, data)
            saved_paths[key] = file_path
            
        return saved_paths
