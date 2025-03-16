"""
Autoencoder model for anomaly detection in OTDR traces.
This module implements an autoencoder for detecting anomalies in fiber optic networks.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GRU, Bidirectional, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

class OTDRAutoencoder:
    """
    Autoencoder model for anomaly detection in OTDR traces.
    """
    
    def __init__(self, input_dim=31, encoding_dim=16, random_state=42):
        """
        Initialize the autoencoder model.
        
        Args:
            input_dim (int): Input dimension (SNR + 30 OTDR trace points)
            encoding_dim (int): Dimension of the encoded representation
            random_state (int): Random seed for reproducibility
        """
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.random_state = random_state
        self.model = None
        self.encoder = None
        self.decoder = None
        self.history = None
        self.threshold = None
        
        # Set random seeds for reproducibility
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        
    def build_model(self):
        """
        Build the autoencoder model architecture.
        
        Returns:
            tensorflow.keras.models.Model: Built autoencoder model
        """
        # Input layer
        input_layer = Input(shape=(self.input_dim,), name='input')
        
        # Encoder layers
        encoded = Dense(64, activation='elu', name='encoder_1')(input_layer)
        encoded = BatchNormalization()(encoded)
        encoded = Dense(32, activation='elu', name='encoder_2')(encoded)
        encoded = BatchNormalization()(encoded)
        
        # Bottleneck layer
        bottleneck = Dense(self.encoding_dim, activation='elu', name='bottleneck')(encoded)
        
        # Decoder layers
        decoded = Dense(32, activation='elu', name='decoder_1')(bottleneck)
        decoded = BatchNormalization()(decoded)
        decoded = Dense(64, activation='elu', name='decoder_2')(decoded)
        decoded = BatchNormalization()(decoded)
        
        # Output layer
        output_layer = Dense(self.input_dim, activation='linear', name='output')(decoded)
        
        # Create model
        self.model = Model(inputs=input_layer, outputs=output_layer)
        
        # Create encoder model
        self.encoder = Model(inputs=input_layer, outputs=bottleneck)
        
        # Create decoder model (for potential future use)
        encoded_input = Input(shape=(self.encoding_dim,))
        decoder_layers = self.model.layers[-5:] # Get decoder layers
        decoded_output = encoded_input
        for layer in decoder_layers:
            decoded_output = layer(decoded_output)
        self.decoder = Model(inputs=encoded_input, outputs=decoded_output)
        
        # Compile model
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        return self.model
    
    def train(self, X_train, X_val, epochs=100, batch_size=32, patience=10, model_path=None):
        """
        Train the autoencoder model.
        
        Args:
            X_train (numpy.ndarray): Training data (normal samples only)
            X_val (numpy.ndarray): Validation data
            epochs (int): Maximum number of training epochs
            batch_size (int): Batch size for training
            patience (int): Patience for early stopping
            model_path (str): Path to save the best model
            
        Returns:
            tensorflow.keras.callbacks.History: Training history
        """
        if self.model is None:
            self.build_model()
            
        # Define callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        if model_path:
            callbacks.append(
                ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)
            )
            
        # Train model
        self.history = self.model.fit(
            X_train, X_train,  # Autoencoder reconstructs its input
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, X_val),
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def compute_threshold(self, X_normal, percentile=95):
        """
        Compute reconstruction error threshold for anomaly detection.
        
        Args:
            X_normal (numpy.ndarray): Normal data samples
            percentile (int): Percentile for threshold calculation
            
        Returns:
            float: Reconstruction error threshold
        """
        # Get reconstructions
        reconstructions = self.model.predict(X_normal)
        
        # Compute reconstruction errors (MSE)
        mse = np.mean(np.square(X_normal - reconstructions), axis=1)
        
        # Set threshold as percentile of reconstruction errors
        self.threshold = np.percentile(mse, percentile)
        
        return self.threshold
    
    def predict_anomalies(self, X_test):
        """
        Predict anomalies based on reconstruction error.
        
        Args:
            X_test (numpy.ndarray): Test data
            
        Returns:
            tuple: (anomaly_scores, anomaly_predictions)
        """
        if self.threshold is None:
            raise ValueError("Threshold not computed. Call compute_threshold() first.")
            
        # Get reconstructions
        reconstructions = self.model.predict(X_test)
        
        # Compute reconstruction errors (MSE)
        mse = np.mean(np.square(X_test - reconstructions), axis=1)
        
        # Classify as anomaly if error > threshold
        predictions = (mse > self.threshold).astype(int)
        
        return mse, predictions
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance.
        
        Args:
            X_test (numpy.ndarray): Test data
            y_test (pandas.DataFrame): Test labels
            
        Returns:
            dict: Performance metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        # Get anomaly predictions
        anomaly_scores, anomaly_preds = self.predict_anomalies(X_test)
        
        # Convert y_test to binary (0: normal, 1: anomaly)
        y_true = (y_test['Class'] != 0.0).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, anomaly_preds)
        precision = precision_score(y_true, anomaly_preds)
        recall = recall_score(y_true, anomaly_preds)
        f1 = f1_score(y_true, anomaly_preds)
        conf_matrix = confusion_matrix(y_true, anomaly_preds)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix,
            'threshold': self.threshold
        }
        
        return metrics
    
    def save_model(self, model_path):
        """
        Save the model to disk.
        
        Args:
            model_path (str): Path to save the model
        """
        self.model.save(model_path)
        
    def load_model(self, model_path):
        """
        Load the model from disk.
        
        Args:
            model_path (str): Path to the saved model
        """
        self.model = tf.keras.models.load_model(model_path)
        
        # Recreate encoder model
        bottleneck_layer_idx = [i for i, layer in enumerate(self.model.layers) if layer.name == 'bottleneck'][0]
        self.encoder = Model(
            inputs=self.model.inputs,
            outputs=self.model.layers[bottleneck_layer_idx].output
        )
