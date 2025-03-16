# FTTH Fiber Optic Fault Detection System

This project implements an AI-based system for automatically detecting, diagnosing, and localizing faults in FTTH fiber optic cables by analyzing OTDR traces. The entire infrastructure and deployment is automated on AWS using Terraform, Ansible, and Jenkins.

## Project Overview

The system uses a two-stage approach for fault detection:
1. An autoencoder-based anomaly detection model to detect any abnormalities in OTDR traces
2. A bidirectional GRU with attention mechanism for fault diagnosis and localization

The system is deployed as a containerized FastAPI application on AWS ECS, with infrastructure provisioned using Terraform and deployment automated with Ansible and Jenkins.

## Directory Structure

```
ftth_fault_detection/
├── data/                      # Data directory
│   └── OTDR_data.csv          # OTDR trace dataset
├── models/                    # Trained models directory
├── src/                       # Source code
│   ├── preprocessing/         # Data preprocessing modules
│   │   ├── __init__.py
│   │   ├── data_loader.py     # Data loading and splitting
│   │   ├── data_processor.py  # Advanced preprocessing techniques
│   │   └── pipeline.py        # End-to-end preprocessing pipeline
│   ├── training/              # Model training modules
│   │   ├── __init__.py
│   │   ├── autoencoder.py     # Autoencoder for anomaly detection
│   │   ├── bigru_attention.py # BiGRU with attention for fault diagnosis
│   │   ├── train_autoencoder.py # Training script for autoencoder
│   │   └── train_bigru_attention.py # Training script for BiGRU
│   ├── inference/             # Inference modules
│   │   ├── __init__.py
│   │   └── predictor.py       # Unified prediction interface
│   └── api/                   # FastAPI application
│       ├── __init__.py
│       ├── app.py             # FastAPI application
│       ├── main.py            # Entry point
│       └── requirements.txt   # API dependencies
├── infrastructure/            # Infrastructure automation
│   ├── terraform/             # Terraform configuration
│   │   ├── main.tf            # Main Terraform configuration
│   │   └── modules/           # Terraform modules
│   │       └── vpc/           # VPC module
│   │           └── main.tf    # VPC configuration
│   └── ansible/               # Ansible configuration
│       └── deploy.yml         # Deployment playbook
├── deployment/                # Deployment artifacts
├── docs/                      # Documentation
├── notebooks/                 # Jupyter notebooks
├── Dockerfile                 # Docker configuration
├── docker-compose.yml         # Docker Compose configuration
└── Jenkinsfile                # Jenkins pipeline configuration
```

## Technical Approach

### Data Preprocessing

The preprocessing module handles:
- Loading and splitting the OTDR dataset
- Normalizing features
- Denoising OTDR traces
- Data augmentation and class balancing
- Preparing data for both autoencoder and BiGRU models

### AI Models

1. **Autoencoder for Anomaly Detection**
   - Detects any abnormalities in OTDR traces
   - Trained only on normal samples
   - Uses reconstruction error to identify anomalies

2. **BiGRU with Attention for Fault Diagnosis and Localization**
   - Classifies the specific fault type (7 fault classes + normal)
   - Localizes the fault position along the fiber
   - Uses attention mechanism to focus on relevant parts of the trace

### API Implementation

The FastAPI application provides:
- Health check endpoint
- Single prediction endpoint for real-time analysis
- Batch processing endpoint for multiple traces
- Model reloading capability

### Infrastructure Automation

- **Terraform**: Provisions AWS resources (VPC, ECS, ECR, S3, etc.)
- **Ansible**: Automates deployment process
- **Jenkins**: Implements CI/CD pipeline for model training and deployment

## Fault Types

The system can detect and diagnose the following fault types:
- Normal (no fault)
- Fiber tapping
- Bad splice
- Bending event
- Dirty connector
- Fiber cut
- PC connector
- Reflector

## Getting Started

### Prerequisites

- Python 3.10+
- Docker and Docker Compose
- AWS CLI configured with appropriate permissions
- Terraform 1.0+
- Ansible 2.9+
- Jenkins (for CI/CD)

### Local Development

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r src/api/requirements.txt
   ```
3. Train models:
   ```
   python src/training/train_autoencoder.py
   python src/training/train_bigru_attention.py
   ```
4. Run the API locally:
   ```
   uvicorn src.api.app:app --host 0.0.0.0 --port 8000
   ```
5. Or use Docker Compose:
   ```
   docker-compose up --build
   ```

### Deployment

The system can be deployed to AWS using:

1. Manual deployment with Ansible:
   ```
   cd infrastructure/ansible
   ansible-playbook deploy.yml
   ```

2. Automated deployment with Jenkins:
   - Configure Jenkins with AWS credentials
   - Create a new pipeline using the provided Jenkinsfile
   - Run the pipeline

## API Usage

### Single Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"snr": 10.0, "trace_points": [0.9, 0.8, 0.7, ...]}'
```

### Batch Processing

```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -F "file=@batch_data.csv"
```

## Performance

Based on the reference paper implementation:
- Anomaly detection: F1 score of 96.86%
- Fault diagnosis: Average accuracy of 98.2%
- Fault localization: Average RMSE of 0.19m

## References

- Based on the approach described in the reference paper (2204.07059v1.pdf)
- Uses a combination of autoencoder for anomaly detection and BiGRU with attention for fault diagnosis
