# FTTH Fiber Optic Fault Detection - Technical Documentation

This document provides detailed technical information about the FTTH Fiber Optic Fault Detection system implementation.

## Data Processing Pipeline

### Data Loading and Preprocessing

The data preprocessing pipeline handles the OTDR trace data with the following steps:

1. **Data Loading**: The `OTDRDataLoader` class loads the CSV data containing SNR values and 30-point OTDR traces.
2. **Feature Extraction**: Extracts SNR and trace points as features, and fault class, position, reflectance, and loss as labels.
3. **Data Splitting**: Splits data into training, validation, and test sets with stratification by fault class.
4. **Normalization**: Applies StandardScaler or MinMaxScaler to normalize features.
5. **Denoising**: The `OTDRDataProcessor` class provides signal denoising using Savitzky-Golay filter, moving average, or wavelet methods.
6. **Class Balancing**: Handles class imbalance through oversampling or undersampling techniques.
7. **Data Augmentation**: Generates additional training samples by adding controlled noise to existing samples.

## AI Models Architecture

### Autoencoder for Anomaly Detection

The autoencoder model is designed to detect anomalies by learning the normal pattern of OTDR traces:

```
Input Layer (31 neurons) → Dense(64) + BatchNorm → Dense(32) + BatchNorm → 
Dense(16) [Bottleneck] → Dense(32) + BatchNorm → Dense(64) + BatchNorm → 
Output Layer (31 neurons)
```

- **Training**: Trained only on normal samples to learn the normal pattern
- **Anomaly Detection**: Uses reconstruction error (MSE) compared against a threshold
- **Threshold Calculation**: Set at the 95th percentile of reconstruction errors on normal samples

### BiGRU with Attention for Fault Diagnosis and Localization

The BiGRU-Attention model performs both classification and regression tasks:

```
Sequence Input (30×1) → Bidirectional GRU(64) → Dropout(0.3) → 
Bidirectional GRU(32) → Dropout(0.3) → Attention Layer → 
[Optional SNR Input → Concatenate] → Dense(64) + BatchNorm + Dropout(0.3) → 
[Classification Output (8 neurons, softmax), Localization Output (1 neuron, sigmoid)]
```

- **Attention Mechanism**: Focuses on the most relevant parts of the OTDR trace
- **Multi-task Learning**: Simultaneously performs fault classification and localization
- **Loss Functions**: Categorical cross-entropy for classification, MSE for localization

## API Implementation Details

### Endpoints

1. **GET /health**
   - Health check endpoint
   - Returns model loading status

2. **POST /predict**
   - Input: JSON with SNR and trace points
   - Process: 
     1. Preprocesses input data
     2. Detects anomalies using autoencoder
     3. If anomaly detected, diagnoses fault type and location using BiGRU-Attention
   - Output: Anomaly status, fault type, location, confidence, and raw predictions

3. **POST /predict/batch**
   - Input: CSV file with multiple OTDR traces
   - Process: Processes each trace as in the single prediction endpoint
   - Output: Array of prediction results

4. **POST /models/reload**
   - Reloads models from disk
   - Useful after model updates

### Error Handling

- Proper HTTP status codes (400 for client errors, 500 for server errors)
- Detailed error messages
- Logging of all API operations and errors

## Infrastructure Architecture

### AWS Resources

1. **Networking**:
   - VPC with public and private subnets across 2 availability zones
   - Internet Gateway for public access
   - NAT Gateway for private subnet outbound traffic
   - Security groups for ALB and ECS tasks

2. **Compute and Container Services**:
   - ECS Cluster with Fargate launch type
   - Task Definition with CPU and memory allocation
   - ECS Service with desired count of 2 for high availability

3. **Storage**:
   - S3 bucket for OTDR data storage
   - S3 bucket for model storage
   - CloudWatch Logs for application logging

4. **Load Balancing**:
   - Application Load Balancer
   - Target Group with health checks
   - HTTP Listener on port 80

5. **Security**:
   - IAM roles with least privilege
   - Security groups with restricted access
   - S3 bucket versioning enabled

### CI/CD Pipeline

The Jenkins pipeline automates the following workflow:

1. **Code Checkout**: Retrieves code from repository
2. **Dependencies Installation**: Installs required Python packages
3. **Testing**: Runs unit tests with coverage reporting
4. **Model Training** (optional): Trains models with latest data
5. **Docker Build**: Builds container image
6. **ECR Push**: Pushes image to Amazon ECR
7. **Model Upload**: Uploads trained models to S3
8. **Infrastructure Deployment**: Applies Terraform configuration
9. **Application Deployment**: Uses Ansible to deploy the application

## Performance Optimization

1. **Model Optimization**:
   - Early stopping during training
   - Learning rate reduction on plateau
   - Batch normalization for faster convergence

2. **API Performance**:
   - Asynchronous processing for batch requests
   - Model loading at startup
   - Proper error handling and logging

3. **Infrastructure Optimization**:
   - Auto-scaling based on CPU/memory usage
   - Load balancing across multiple containers
   - Proper resource allocation for ECS tasks

## Security Considerations

1. **Data Security**:
   - S3 bucket versioning
   - IAM roles with least privilege
   - Data encryption in transit and at rest

2. **Application Security**:
   - Input validation
   - Proper error handling
   - CORS configuration

3. **Infrastructure Security**:
   - Security groups with restricted access
   - Private subnets for ECS tasks
   - Regular security updates

## Monitoring and Logging

1. **Application Logging**:
   - Request/response logging
   - Error logging
   - Performance metrics

2. **Infrastructure Monitoring**:
   - CloudWatch metrics for ECS, ALB, and S3
   - Health checks for ECS tasks
   - ALB access logs

## Future Enhancements

1. **Model Improvements**:
   - Implement online learning for continuous model improvement
   - Add support for different OTDR trace lengths
   - Explore transformer-based models for improved accuracy

2. **API Enhancements**:
   - Add authentication and authorization
   - Implement rate limiting
   - Add support for different input formats

3. **Infrastructure Enhancements**:
   - Add HTTPS support with ACM certificates
   - Implement blue/green deployments
   - Add disaster recovery capabilities
