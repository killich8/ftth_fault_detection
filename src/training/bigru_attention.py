"""
Bidirectional GRU with Attention model for fault diagnosis and localization in OTDR traces.
This module implements a BiGRU model with attention mechanism for classifying and localizing faults.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GRU, Bidirectional, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

class AttentionLayer(tf.keras.layers.Layer):
    """
    Attention layer for focusing on relevant parts of the sequence.
    """
    
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight',
                                 shape=(input_shape[-1], 1),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(name='attention_bias',
                                 shape=(input_shape[1], 1),
                                 initializer='zeros',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)
        
    def call(self, x):
        # Alignment scores
        e = tf.tanh(tf.matmul(x, self.W) + self.b)
        
        # Compute attention weights
        a = tf.nn.softmax(e, axis=1)
        
        # Compute context vector
        context = x * a
        context = tf.reduce_sum(context, axis=1)
        
        return context, a
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
    
    def get_config(self):
        return super(AttentionLayer, self).get_config()


class BiGRUAttention:
    """
    Bidirectional GRU with Attention model for fault diagnosis and localization.
    """
    
    def __init__(self, sequence_length=30, num_features=1, num_classes=8, random_state=42):
        """
        Initialize the BiGRU with Attention model.
        
        Args:
            sequence_length (int): Length of input sequences
            num_features (int): Number of features per time step
            num_classes (int): Number of fault classes
            random_state (int): Random seed for reproducibility
        """
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.num_classes = num_classes
        self.random_state = random_state
        self.model = None
        self.history = None
        
        # Set random seeds for reproducibility
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        
    def build_model(self, include_snr=True):
        """
        Build the BiGRU with Attention model architecture.
        
        Args:
            include_snr (bool): Whether to include SNR as additional input
            
        Returns:
            tensorflow.keras.models.Model: Built model
        """
        # Sequence input
        sequence_input = Input(shape=(self.sequence_length, self.num_features), name='sequence_input')
        
        # BiGRU layers
        gru1 = Bidirectional(GRU(64, return_sequences=True))(sequence_input)
        gru1 = Dropout(0.3)(gru1)
        
        gru2 = Bidirectional(GRU(32, return_sequences=True))(gru1)
        gru2 = Dropout(0.3)(gru2)
        
        # Attention mechanism
        context_vector, attention_weights = AttentionLayer()(gru2)
        
        # Additional input for SNR if requested
        if include_snr:
            snr_input = Input(shape=(1,), name='snr_input')
            combined = Concatenate()([context_vector, snr_input])
            
            # Dense layers for classification
            dense1 = Dense(64, activation='relu')(combined)
            dense1 = BatchNormalization()(dense1)
            dense1 = Dropout(0.3)(dense1)
            
            # Classification output
            classification_output = Dense(self.num_classes, activation='softmax', name='classification_output')(dense1)
            
            # Regression output for localization
            localization_output = Dense(1, activation='sigmoid', name='localization_output')(dense1)
            
            # Create model with multiple inputs and outputs
            self.model = Model(
                inputs=[sequence_input, snr_input],
                outputs=[classification_output, localization_output]
            )
        else:
            # Dense layers for classification
            dense1 = Dense(64, activation='relu')(context_vector)
            dense1 = BatchNormalization()(dense1)
            dense1 = Dropout(0.3)(dense1)
            
            # Classification output
            classification_output = Dense(self.num_classes, activation='softmax', name='classification_output')(dense1)
            
            # Regression output for localization
            localization_output = Dense(1, activation='sigmoid', name='localization_output')(dense1)
            
            # Create model with single input and multiple outputs
            self.model = Model(
                inputs=sequence_input,
                outputs=[classification_output, localization_output]
            )
        
        # Compile model with appropriate loss functions and metrics
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={
                'classification_output': 'categorical_crossentropy',
                'localization_output': 'mse'
            },
            metrics={
                'classification_output': ['accuracy'],
                'localization_output': ['mae', 'mse']
            },
            loss_weights={
                'classification_output': 1.0,
                'localization_output': 0.5
            }
        )
        
        return self.model
    
    def train(self, X_train, y_train_class, y_train_pos, X_val=None, y_val_class=None, y_val_pos=None,
              epochs=100, batch_size=32, patience=10, model_path=None, include_snr=True):
        """
        Train the BiGRU with Attention model.
        
        Args:
            X_train (numpy.ndarray): Training sequences
            y_train_class (numpy.ndarray): Training class labels (one-hot encoded)
            y_train_pos (numpy.ndarray): Training position labels
            X_val (numpy.ndarray): Validation sequences
            y_val_class (numpy.ndarray): Validation class labels (one-hot encoded)
            y_val_pos (numpy.ndarray): Validation position labels
            epochs (int): Maximum number of training epochs
            batch_size (int): Batch size for training
            patience (int): Patience for early stopping
            model_path (str): Path to save the best model
            include_snr (bool): Whether to include SNR as additional input
            
        Returns:
            tensorflow.keras.callbacks.History: Training history
        """
        if self.model is None:
            self.build_model(include_snr=include_snr)
            
        # Define callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        if model_path:
            callbacks.append(
                ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)
            )
        
        # Prepare inputs based on model architecture
        if include_snr:
            # Extract SNR values (assuming they're stored separately or in the first column)
            train_snr = X_train[:, 0, 0].reshape(-1, 1)  # Reshape to (samples, 1)
            train_inputs = [X_train, train_snr]
            
            if X_val is not None:
                val_snr = X_val[:, 0, 0].reshape(-1, 1)
                val_inputs = [X_val, val_snr]
            else:
                val_inputs = None
        else:
            train_inputs = X_train
            val_inputs = X_val
        
        # Prepare validation data if provided
        if X_val is not None and y_val_class is not None and y_val_pos is not None:
            validation_data = (val_inputs, [y_val_class, y_val_pos])
        else:
            validation_data = None
            
        # Train model
        self.history = self.model.fit(
            train_inputs,
            [y_train_class, y_train_pos],
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def predict(self, X_test, include_snr=True):
        """
        Make predictions with the trained model.
        
        Args:
            X_test (numpy.ndarray): Test sequences
            include_snr (bool): Whether to include SNR as additional input
            
        Returns:
            tuple: (class_predictions, position_predictions)
        """
        if include_snr:
            # Extract SNR values
            test_snr = X_test[:, 0, 0].reshape(-1, 1)
            test_inputs = [X_test, test_snr]
        else:
            test_inputs = X_test
            
        # Get model predictions
        class_preds, pos_preds = self.model.predict(test_inputs)
        
        # Convert class probabilities to class indices
        class_indices = np.argmax(class_preds, axis=1)
        
        return class_indices, pos_preds
    
    def evaluate(self, X_test, y_test_class, y_test_pos, include_snr=True):
        """
        Evaluate model performance.
        
        Args:
            X_test (numpy.ndarray): Test sequences
            y_test_class (numpy.ndarray): Test class labels (one-hot encoded)
            y_test_pos (numpy.ndarray): Test position labels
            include_snr (bool): Whether to include SNR as additional input
            
        Returns:
            dict: Performance metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error, mean_absolute_error
        
        # Get predictions
        class_indices, pos_preds = self.predict(X_test, include_snr=include_snr)
        
        # Convert one-hot encoded test labels to indices
        y_true_class = np.argmax(y_test_class, axis=1)
        
        # Calculate classification metrics
        accuracy = accuracy_score(y_true_class, class_indices)
        precision = precision_score(y_true_class, class_indices, average='weighted')
        recall = recall_score(y_true_class, class_indices, average='weighted')
        f1 = f1_score(y_true_class, class_indices, average='weighted')
        conf_matrix = confusion_matrix(y_true_class, class_indices)
        
        # Calculate regression metrics for localization
        mse = mean_squared_error(y_test_pos, pos_preds)
        mae = mean_absolute_error(y_test_pos, pos_preds)
        rmse = np.sqrt(mse)
        
        metrics = {
            'classification': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': conf_matrix
            },
            'localization': {
                'mse': mse,
                'mae': mae,
                'rmse': rmse
            }
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
        # Custom objects needed for loading
        custom_objects = {'AttentionLayer': AttentionLayer}
        
        self.model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

