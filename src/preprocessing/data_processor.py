"""
Data processor module for OTDR trace analysis.
This module handles advanced preprocessing of OTDR data.
"""

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import OneHotEncoder

class OTDRDataProcessor:
    """
    Class for advanced preprocessing of OTDR data.
    """
    
    def __init__(self):
        """
        Initialize the data processor.
        """
        self.class_encoder = None
        
    def denoise_signal(self, trace_data, method='savgol', window_length=5, polyorder=2):
        """
        Apply denoising to OTDR trace signals.
        
        Args:
            trace_data (numpy.ndarray): OTDR trace data with shape (samples, features)
            method (str): Denoising method ('savgol', 'moving_avg', or 'wavelet')
            window_length (int): Window length for filtering
            polyorder (int): Polynomial order for Savitzky-Golay filter
            
        Returns:
            numpy.ndarray: Denoised trace data
        """
        if method == 'savgol':
            # Apply Savitzky-Golay filter for smoothing
            if trace_data.ndim == 3:  # For RNN input (samples, time_steps, features)
                samples, time_steps, features = trace_data.shape
                denoised_data = np.zeros_like(trace_data)
                for i in range(samples):
                    for j in range(features):
                        denoised_data[i, :, j] = signal.savgol_filter(
                            trace_data[i, :, j], window_length, polyorder
                        )
            else:  # For 2D input (samples, features)
                denoised_data = np.zeros_like(trace_data)
                for i in range(trace_data.shape[0]):
                    # Extract only the trace points (assuming first column is SNR)
                    trace_points = trace_data[i, 1:]
                    denoised_trace = signal.savgol_filter(trace_points, window_length, polyorder)
                    denoised_data[i, 0] = trace_data[i, 0]  # Keep SNR value
                    denoised_data[i, 1:] = denoised_trace
                    
        elif method == 'moving_avg':
            # Apply moving average filter
            kernel = np.ones(window_length) / window_length
            if trace_data.ndim == 3:
                samples, time_steps, features = trace_data.shape
                denoised_data = np.zeros_like(trace_data)
                for i in range(samples):
                    for j in range(features):
                        denoised_data[i, :, j] = np.convolve(
                            trace_data[i, :, j], kernel, mode='same'
                        )
            else:
                denoised_data = np.zeros_like(trace_data)
                for i in range(trace_data.shape[0]):
                    trace_points = trace_data[i, 1:]
                    denoised_trace = np.convolve(trace_points, kernel, mode='same')
                    denoised_data[i, 0] = trace_data[i, 0]
                    denoised_data[i, 1:] = denoised_trace
                    
        else:
            # Default: return original data
            denoised_data = trace_data
            
        return denoised_data
    
    def augment_data(self, X, y, noise_level=0.05, num_augmented=None):
        """
        Augment training data by adding noise to existing samples.
        
        Args:
            X (numpy.ndarray): Feature data
            y (numpy.ndarray or pandas.DataFrame): Label data
            noise_level (float): Standard deviation of Gaussian noise
            num_augmented (int): Number of augmented samples to generate (default: same as input)
            
        Returns:
            tuple: (X_augmented, y_augmented)
        """
        if num_augmented is None:
            num_augmented = X.shape[0]
            
        # Randomly select samples to augment
        indices = np.random.choice(X.shape[0], num_augmented, replace=True)
        
        # Create augmented features by adding Gaussian noise
        X_selected = X[indices]
        noise = np.random.normal(0, noise_level, X_selected.shape)
        X_augmented = X_selected + noise
        
        # Keep corresponding labels
        if isinstance(y, pd.DataFrame):
            y_augmented = y.iloc[indices].reset_index(drop=True)
        else:
            y_augmented = y[indices]
            
        return X_augmented, y_augmented
    
    def balance_classes(self, X, y, method='oversample', target_column='Class'):
        """
        Balance class distribution in the dataset.
        
        Args:
            X (numpy.ndarray): Feature data
            y (pandas.DataFrame): Label data
            method (str): Balancing method ('oversample', 'undersample', or 'smote')
            target_column (str): Column name for class labels
            
        Returns:
            tuple: (X_balanced, y_balanced)
        """
        if method == 'oversample':
            # Simple random oversampling
            class_counts = y[target_column].value_counts()
            max_count = class_counts.max()
            
            X_balanced_list = []
            y_balanced_list = []
            
            for class_value, count in class_counts.items():
                # Get samples for this class
                class_indices = y[target_column] == class_value
                X_class = X[class_indices]
                y_class = y[class_indices]
                
                # Calculate how many additional samples needed
                n_oversample = max_count - count
                
                if n_oversample > 0:
                    # Randomly oversample with replacement
                    oversample_indices = np.random.choice(X_class.shape[0], n_oversample, replace=True)
                    X_oversampled = X_class[oversample_indices]
                    
                    if isinstance(y, pd.DataFrame):
                        y_oversampled = y_class.iloc[oversample_indices].reset_index(drop=True)
                    else:
                        y_oversampled = y_class[oversample_indices]
                        
                    # Combine original and oversampled data
                    X_balanced_list.append(np.vstack([X_class, X_oversampled]))
                    
                    if isinstance(y, pd.DataFrame):
                        y_balanced_list.append(pd.concat([y_class, y_oversampled], ignore_index=True))
                    else:
                        y_balanced_list.append(np.vstack([y_class, y_oversampled]))
                else:
                    X_balanced_list.append(X_class)
                    y_balanced_list.append(y_class)
            
            # Combine all classes
            X_balanced = np.vstack(X_balanced_list)
            
            if isinstance(y, pd.DataFrame):
                y_balanced = pd.concat(y_balanced_list, ignore_index=True)
            else:
                y_balanced = np.vstack(y_balanced_list)
                
        elif method == 'undersample':
            # Simple random undersampling
            class_counts = y[target_column].value_counts()
            min_count = class_counts.min()
            
            X_balanced_list = []
            y_balanced_list = []
            
            for class_value in class_counts.index:
                # Get samples for this class
                class_indices = y[target_column] == class_value
                X_class = X[class_indices]
                y_class = y[class_indices]
                
                # Randomly undersample without replacement
                undersample_indices = np.random.choice(X_class.shape[0], min_count, replace=False)
                X_undersampled = X_class[undersample_indices]
                
                if isinstance(y, pd.DataFrame):
                    y_undersampled = y_class.iloc[undersample_indices].reset_index(drop=True)
                else:
                    y_undersampled = y_class[undersample_indices]
                    
                X_balanced_list.append(X_undersampled)
                y_balanced_list.append(y_undersampled)
            
            # Combine all classes
            X_balanced = np.vstack(X_balanced_list)
            
            if isinstance(y, pd.DataFrame):
                y_balanced = pd.concat(y_balanced_list, ignore_index=True)
            else:
                y_balanced = np.vstack(y_balanced_list)
                
        else:
            # Default: return original data
            X_balanced = X
            y_balanced = y
            
        return X_balanced, y_balanced
    
    def encode_classes(self, y, column='Class'):
        """
        One-hot encode class labels.
        
        Args:
            y (pandas.DataFrame or numpy.ndarray): Label data
            column (str): Column name for class labels
            
        Returns:
            numpy.ndarray: One-hot encoded class labels
        """
        if self.class_encoder is None:
            self.class_encoder = OneHotEncoder(sparse=False)
            
            if isinstance(y, pd.DataFrame):
                y_encoded = self.class_encoder.fit_transform(y[[column]])
            else:
                y_encoded = self.class_encoder.fit_transform(y.reshape(-1, 1))
        else:
            if isinstance(y, pd.DataFrame):
                y_encoded = self.class_encoder.transform(y[[column]])
            else:
                y_encoded = self.class_encoder.transform(y.reshape(-1, 1))
                
        return y_encoded
    
    def extract_features(self, trace_data):
        """
        Extract additional features from OTDR trace data.
        
        Args:
            trace_data (numpy.ndarray): OTDR trace data
            
        Returns:
            numpy.ndarray: Extracted features
        """
        # Extract statistical features from each trace
        features = []
        
        for i in range(trace_data.shape[0]):
            trace = trace_data[i]
            
            # Basic statistics
            mean = np.mean(trace)
            std = np.std(trace)
            max_val = np.max(trace)
            min_val = np.min(trace)
            range_val = max_val - min_val
            
            # Gradient features
            gradient = np.gradient(trace)
            mean_gradient = np.mean(gradient)
            max_gradient = np.max(gradient)
            min_gradient = np.min(gradient)
            
            # Peak detection
            peaks, _ = signal.find_peaks(trace)
            num_peaks = len(peaks)
            
            # Combine features
            sample_features = [
                mean, std, max_val, min_val, range_val,
                mean_gradient, max_gradient, min_gradient,
                num_peaks
            ]
            
            features.append(sample_features)
            
        return np.array(features)
