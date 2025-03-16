"""
Data loader module for OTDR trace analysis.
This module handles loading and initial processing of OTDR data.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class OTDRDataLoader:
    """
    Class for loading and preprocessing OTDR data.
    """
    
    def __init__(self, data_path, random_state=42):
        """
        Initialize the data loader.
        
        Args:
            data_path (str): Path to the OTDR data CSV file
            random_state (int): Random seed for reproducibility
        """
        self.data_path = data_path
        self.random_state = random_state
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
    def load_data(self):
        """
        Load the OTDR data from CSV file.
        
        Returns:
            pandas.DataFrame: Loaded data
        """
        self.data = pd.read_csv(self.data_path)
        print(f"Loaded data with shape: {self.data.shape}")
        return self.data
    
    def extract_features_and_labels(self):
        """
        Extract features (SNR and OTDR trace points) and labels (Class, Position, etc.).
        
        Returns:
            tuple: (X, y) where X contains features and y contains labels
        """
        if self.data is None:
            self.load_data()
            
        # Extract features: SNR and OTDR trace points (P1-P30)
        trace_columns = [f'P{i}' for i in range(1, 31)]
        self.X = self.data[['SNR'] + trace_columns].copy()
        
        # Extract labels: Class, Position, Reflectance, loss
        self.y = self.data[['Class', 'Position', 'Reflectance', 'loss']].copy()
        
        return self.X, self.y
    
    def split_data(self, test_size=0.2, val_size=0.25):
        """
        Split data into training, validation, and test sets.
        
        Args:
            test_size (float): Proportion of data to use for testing
            val_size (float): Proportion of training data to use for validation
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        if self.X is None or self.y is None:
            self.extract_features_and_labels()
            
        # First split: training + validation vs test
        X_train_val, self.X_test, y_train_val, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=self.random_state, stratify=self.y['Class']
        )
        
        # Second split: training vs validation
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size, random_state=self.random_state, stratify=y_train_val['Class']
        )
        
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Validation set: {self.X_val.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
    
    def get_class_distribution(self):
        """
        Get the distribution of fault classes in the dataset.
        
        Returns:
            pandas.Series: Class distribution
        """
        if self.y is None:
            self.extract_features_and_labels()
            
        class_mapping = {
            0.0: "normal",
            1.0: "fiber tapping",
            2.0: "bad splice",
            3.0: "bending event",
            4.0: "dirty connector",
            5.0: "fiber cut",
            6.0: "PC connector",
            7.0: "reflector"
        }
        
        class_counts = self.y['Class'].value_counts().sort_index()
        class_counts.index = class_counts.index.map(lambda x: f"{int(x)}: {class_mapping.get(x, 'unknown')}")
        
        return class_counts
    
    def normalize_features(self, scaler_type='standard'):
        """
        Normalize features using StandardScaler or MinMaxScaler.
        
        Args:
            scaler_type (str): Type of scaler to use ('standard' or 'minmax')
            
        Returns:
            tuple: (X_train_scaled, X_val_scaled, X_test_scaled, scaler)
        """
        if self.X_train is None:
            self.split_data()
            
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("scaler_type must be 'standard' or 'minmax'")
            
        # Fit scaler on training data only
        X_train_scaled = scaler.fit_transform(self.X_train)
        
        # Transform validation and test data
        X_val_scaled = scaler.transform(self.X_val)
        X_test_scaled = scaler.transform(self.X_test)
        
        return X_train_scaled, X_val_scaled, X_test_scaled, scaler
    
    def prepare_data_for_autoencoder(self, normal_class=0.0):
        """
        Prepare data for autoencoder-based anomaly detection.
        For training, only normal samples are used.
        
        Args:
            normal_class (float): Class value representing normal samples
            
        Returns:
            tuple: (X_train_normal, X_val, X_test, y_test)
        """
        if self.X_train is None:
            self.split_data()
            
        # Get indices of normal samples in training set
        normal_indices = self.y_train['Class'] == normal_class
        
        # Extract normal samples for training
        X_train_normal = self.X_train[normal_indices]
        
        # Normalize data
        X_train_normal_scaled, X_val_scaled, X_test_scaled, _ = self.normalize_features()
        
        return X_train_normal_scaled, X_val_scaled, X_test_scaled, self.y_test
    
    def prepare_data_for_classification(self):
        """
        Prepare data for fault classification and localization.
        
        Returns:
            tuple: (X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test)
        """
        if self.X_train is None:
            self.split_data()
            
        # Normalize data
        X_train_scaled, X_val_scaled, X_test_scaled, _ = self.normalize_features()
        
        return X_train_scaled, X_val_scaled, X_test_scaled, self.y_train, self.y_val, self.y_test
    
    def prepare_sequences_for_rnn(self, sequence_length=30):
        """
        Prepare sequence data for RNN models.
        
        Args:
            sequence_length (int): Length of sequences
            
        Returns:
            tuple: (X_train_seq, X_val_seq, X_test_seq, y_train, y_val, y_test)
        """
        if self.X_train is None:
            self.split_data()
            
        # Extract OTDR trace points only
        trace_columns = [f'P{i}' for i in range(1, 31)]
        
        X_train_traces = self.X_train[trace_columns].values
        X_val_traces = self.X_val[trace_columns].values
        X_test_traces = self.X_test[trace_columns].values
        
        # Reshape for RNN input: (samples, time_steps, features)
        X_train_seq = X_train_traces.reshape(X_train_traces.shape[0], sequence_length, 1)
        X_val_seq = X_val_traces.reshape(X_val_traces.shape[0], sequence_length, 1)
        X_test_seq = X_test_traces.reshape(X_test_traces.shape[0], sequence_length, 1)
        
        return X_train_seq, X_val_seq, X_test_seq, self.y_train, self.y_val, self.y_test
