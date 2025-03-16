"""
Model training script for the autoencoder-based anomaly detection model.
This script trains an autoencoder to detect anomalies in OTDR traces.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import sys
sys.path.append('./')

from src.preprocessing import OTDRPreprocessingPipeline
from src.training import OTDRAutoencoder

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train autoencoder for OTDR anomaly detection')
    parser.add_argument('--data_path', type=str, default='data/OTDR_data.csv',
                        help='Path to the OTDR data CSV file')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Directory to save the trained model and results')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--encoding_dim', type=int, default=16,
                        help='Dimension of the encoded representation')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--denoise', action='store_true',
                        help='Apply denoising to OTDR traces')
    return parser.parse_args()

def main():
    """Main function to train the autoencoder model."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create timestamp for model versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(args.output_dir, f'autoencoder_{timestamp}')
    os.makedirs(model_dir, exist_ok=True)
    
    # Prepare data
    print("Preparing data...")
    pipeline = OTDRPreprocessingPipeline(args.data_path, args.random_state)
    data_dict = pipeline.prepare_data_for_anomaly_detection(denoise=args.denoise)
    
    X_train = data_dict['X_train']
    X_val = data_dict['X_val']
    X_test = data_dict['X_test']
    y_test = data_dict['y_test']
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Initialize and build model
    print("Building model...")
    input_dim = X_train.shape[1]
    model = OTDRAutoencoder(input_dim=input_dim, encoding_dim=args.encoding_dim, random_state=args.random_state)
    model.build_model()
    model.model.summary()
    
    # Train model
    print("Training model...")
    model_path = os.path.join(model_dir, 'autoencoder_model.h5')
    history = model.train(
        X_train, X_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_path=model_path
    )
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Autoencoder Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(model_dir, 'training_history.png'))
    
    # Compute threshold for anomaly detection
    print("Computing threshold...")
    threshold = model.compute_threshold(X_train)
    print(f"Anomaly threshold: {threshold:.6f}")
    
    # Evaluate model
    print("Evaluating model...")
    metrics = model.evaluate(X_test, y_test)
    
    # Print evaluation metrics
    print("\nEvaluation Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
    
    # Save evaluation metrics
    with open(os.path.join(model_dir, 'evaluation_metrics.txt'), 'w') as f:
        f.write(f"Anomaly Threshold: {threshold:.6f}\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1_score']:.4f}\n")
        f.write(f"Confusion Matrix:\n{metrics['confusion_matrix']}\n")
    
    # Save model configuration
    config = {
        'input_dim': input_dim,
        'encoding_dim': args.encoding_dim,
        'threshold': threshold,
        'random_state': args.random_state,
        'denoise': args.denoise,
        'timestamp': timestamp
    }
    np.save(os.path.join(model_dir, 'model_config.npy'), config)
    
    print(f"Model and results saved to {model_dir}")

if __name__ == '__main__':
    main()
