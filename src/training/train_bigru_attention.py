"""
Model training script for the BiGRU-Attention model for fault diagnosis and localization.
This script trains a bidirectional GRU with attention mechanism to classify and localize faults in OTDR traces.
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
from src.training import BiGRUAttention

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train BiGRU-Attention model for OTDR fault diagnosis and localization')
    parser.add_argument('--data_path', type=str, default='data/OTDR_data.csv',
                        help='Path to the OTDR data CSV file')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Directory to save the trained model and results')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--sequence_length', type=int, default=30,
                        help='Length of input sequences')
    parser.add_argument('--num_classes', type=int, default=8,
                        help='Number of fault classes')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--denoise', action='store_true',
                        help='Apply denoising to OTDR traces')
    parser.add_argument('--balance_classes', action='store_true',
                        help='Balance class distribution')
    parser.add_argument('--include_snr', action='store_true',
                        help='Include SNR as additional input')
    return parser.parse_args()

def main():
    """Main function to train the BiGRU-Attention model."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create timestamp for model versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(args.output_dir, f'bigru_attention_{timestamp}')
    os.makedirs(model_dir, exist_ok=True)
    
    # Prepare data
    print("Preparing data...")
    pipeline = OTDRPreprocessingPipeline(args.data_path, args.random_state)
    data_dict = pipeline.prepare_data_for_rnn(denoise=args.denoise, balance_classes=args.balance_classes)
    
    X_train = data_dict['X_train']
    X_val = data_dict['X_val']
    X_test = data_dict['X_test']
    y_train_encoded = data_dict['y_train_encoded']
    y_val_encoded = data_dict['y_val_encoded']
    y_test_encoded = data_dict['y_test_encoded']
    y_train_pos = data_dict['y_train_pos']
    y_val_pos = data_dict['y_val_pos']
    y_test_pos = data_dict['y_test_pos']
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Initialize and build model
    print("Building model...")
    model = BiGRUAttention(
        sequence_length=args.sequence_length,
        num_features=X_train.shape[2],
        num_classes=args.num_classes,
        random_state=args.random_state
    )
    model.build_model(include_snr=args.include_snr)
    model.model.summary()
    
    # Train model
    print("Training model...")
    model_path = os.path.join(model_dir, 'bigru_attention_model.h5')
    history = model.train(
        X_train, y_train_encoded, y_train_pos,
        X_val, y_val_encoded, y_val_pos,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_path=model_path,
        include_snr=args.include_snr
    )
    
    # Plot training history
    plt.figure(figsize=(12, 8))
    
    # Plot classification loss
    plt.subplot(2, 2, 1)
    plt.plot(history.history['classification_output_loss'], label='Training Loss')
    plt.plot(history.history['val_classification_output_loss'], label='Validation Loss')
    plt.title('Classification Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot classification accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history.history['classification_output_accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_classification_output_accuracy'], label='Validation Accuracy')
    plt.title('Classification Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot localization loss
    plt.subplot(2, 2, 3)
    plt.plot(history.history['localization_output_loss'], label='Training Loss')
    plt.plot(history.history['val_localization_output_loss'], label='Validation Loss')
    plt.title('Localization Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot localization MAE
    plt.subplot(2, 2, 4)
    plt.plot(history.history['localization_output_mae'], label='Training MAE')
    plt.plot(history.history['val_localization_output_mae'], label='Validation MAE')
    plt.title('Localization MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'training_history.png'))
    
    # Evaluate model
    print("Evaluating model...")
    metrics = model.evaluate(X_test, y_test_encoded, y_test_pos, include_snr=args.include_snr)
    
    # Print evaluation metrics
    print("\nClassification Metrics:")
    print(f"Accuracy: {metrics['classification']['accuracy']:.4f}")
    print(f"Precision: {metrics['classification']['precision']:.4f}")
    print(f"Recall: {metrics['classification']['recall']:.4f}")
    print(f"F1 Score: {metrics['classification']['f1_score']:.4f}")
    print(f"Confusion Matrix:\n{metrics['classification']['confusion_matrix']}")
    
    print("\nLocalization Metrics:")
    print(f"MSE: {metrics['localization']['mse']:.6f}")
    print(f"MAE: {metrics['localization']['mae']:.6f}")
    print(f"RMSE: {metrics['localization']['rmse']:.6f}")
    
    # Save evaluation metrics
    with open(os.path.join(model_dir, 'evaluation_metrics.txt'), 'w') as f:
        f.write("Classification Metrics:\n")
        f.write(f"Accuracy: {metrics['classification']['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['classification']['precision']:.4f}\n")
        f.write(f"Recall: {metrics['classification']['recall']:.4f}\n")
        f.write(f"F1 Score: {metrics['classification']['f1_score']:.4f}\n")
        f.write(f"Confusion Matrix:\n{metrics['classification']['confusion_matrix']}\n\n")
        
        f.write("Localization Metrics:\n")
        f.write(f"MSE: {metrics['localization']['mse']:.6f}\n")
        f.write(f"MAE: {metrics['localization']['mae']:.6f}\n")
        f.write(f"RMSE: {metrics['localization']['rmse']:.6f}\n")
    
    # Save model configuration
    config = {
        'sequence_length': args.sequence_length,
        'num_features': X_train.shape[2],
        'num_classes': args.num_classes,
        'random_state': args.random_state,
        'denoise': args.denoise,
        'balance_classes': args.balance_classes,
        'include_snr': args.include_snr,
        'timestamp': timestamp
    }
    np.save(os.path.join(model_dir, 'model_config.npy'), config)
    
    print(f"Model and results saved to {model_dir}")

if __name__ == '__main__':
    main()
