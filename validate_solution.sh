#!/bin/bash

# Validation script for FTTH Fiber Optic Fault Detection System
# This script checks that all components are properly implemented and integrated

echo "Starting validation of FTTH Fiber Optic Fault Detection System..."
echo "=================================================================="

# Check directory structure
echo -e "\n[1/7] Checking directory structure..."
DIRECTORIES=(
    "data"
    "models"
    "src/preprocessing"
    "src/training"
    "src/inference"
    "src/api"
    "infrastructure/terraform"
    "infrastructure/terraform/modules/vpc"
    "infrastructure/ansible"
    "deployment"
    "docs"
    "notebooks"
)

MISSING_DIRS=0
for DIR in "${DIRECTORIES[@]}"; do
    if [ ! -d "ftth_fault_detection/$DIR" ]; then
        echo "❌ Missing directory: ftth_fault_detection/$DIR"
        MISSING_DIRS=$((MISSING_DIRS+1))
    else
        echo "✅ Directory exists: ftth_fault_detection/$DIR"
    fi
done

if [ $MISSING_DIRS -eq 0 ]; then
    echo "✅ All required directories are present."
else
    echo "❌ $MISSING_DIRS directories are missing."
fi

# Check key files
echo -e "\n[2/7] Checking key files..."
FILES=(
    "data/OTDR_data.csv"
    "src/preprocessing/__init__.py"
    "src/preprocessing/data_loader.py"
    "src/preprocessing/data_processor.py"
    "src/preprocessing/pipeline.py"
    "src/training/__init__.py"
    "src/training/autoencoder.py"
    "src/training/bigru_attention.py"
    "src/training/train_autoencoder.py"
    "src/training/train_bigru_attention.py"
    "src/inference/__init__.py"
    "src/inference/predictor.py"
    "src/api/__init__.py"
    "src/api/app.py"
    "src/api/main.py"
    "src/api/requirements.txt"
    "infrastructure/terraform/main.tf"
    "infrastructure/terraform/modules/vpc/main.tf"
    "infrastructure/ansible/deploy.yml"
    "Dockerfile"
    "docker-compose.yml"
    "Jenkinsfile"
    "README.md"
    "docs/technical_documentation.md"
)

MISSING_FILES=0
for FILE in "${FILES[@]}"; do
    if [ ! -f "ftth_fault_detection/$FILE" ]; then
        echo "❌ Missing file: ftth_fault_detection/$FILE"
        MISSING_FILES=$((MISSING_FILES+1))
    else
        echo "✅ File exists: ftth_fault_detection/$FILE"
    fi
done

if [ $MISSING_FILES -eq 0 ]; then
    echo "✅ All required files are present."
else
    echo "❌ $MISSING_FILES files are missing."
fi

# Check Python modules
echo -e "\n[3/7] Checking Python modules..."
PYTHON_MODULES=(
    "src/preprocessing/data_loader.py:OTDRDataLoader"
    "src/preprocessing/data_processor.py:OTDRDataProcessor"
    "src/preprocessing/pipeline.py:OTDRPreprocessingPipeline"
    "src/training/autoencoder.py:OTDRAutoencoder"
    "src/training/bigru_attention.py:BiGRUAttention"
    "src/inference/predictor.py:OTDRFaultPredictor"
)

MISSING_MODULES=0
for MODULE in "${PYTHON_MODULES[@]}"; do
    FILE=$(echo $MODULE | cut -d':' -f1)
    CLASS=$(echo $MODULE | cut -d':' -f2)
    if [ -f "ftth_fault_detection/$FILE" ]; then
        if grep -q "class $CLASS" "ftth_fault_detection/$FILE"; then
            echo "✅ Module $CLASS found in $FILE"
        else
            echo "❌ Module $CLASS not found in $FILE"
            MISSING_MODULES=$((MISSING_MODULES+1))
        fi
    else
        echo "❌ File $FILE not found, cannot check for module $CLASS"
        MISSING_MODULES=$((MISSING_MODULES+1))
    fi
done

if [ $MISSING_MODULES -eq 0 ]; then
    echo "✅ All required Python modules are present."
else
    echo "❌ $MISSING_MODULES Python modules are missing."
fi

# Check FastAPI endpoints
echo -e "\n[4/7] Checking FastAPI endpoints..."
ENDPOINTS=(
    "app.py:@app.get(\"/\")"
    "app.py:@app.get(\"/health\")"
    "app.py:@app.post(\"/predict\""
    "app.py:@app.post(\"/predict/batch\")"
    "app.py:@app.post(\"/models/reload\")"
)

MISSING_ENDPOINTS=0
for ENDPOINT in "${ENDPOINTS[@]}"; do
    FILE=$(echo $ENDPOINT | cut -d':' -f1)
    PATTERN=$(echo $ENDPOINT | cut -d':' -f2)
    if [ -f "ftth_fault_detection/src/api/$FILE" ]; then
        if grep -q "$PATTERN" "ftth_fault_detection/src/api/$FILE"; then
            echo "✅ Endpoint $PATTERN found in $FILE"
        else
            echo "❌ Endpoint $PATTERN not found in $FILE"
            MISSING_ENDPOINTS=$((MISSING_ENDPOINTS+1))
        fi
    else
        echo "❌ File $FILE not found, cannot check for endpoint $PATTERN"
        MISSING_ENDPOINTS=$((MISSING_ENDPOINTS+1))
    fi
done

if [ $MISSING_ENDPOINTS -eq 0 ]; then
    echo "✅ All required FastAPI endpoints are present."
else
    echo "❌ $MISSING_ENDPOINTS FastAPI endpoints are missing."
fi

# Check Terraform resources
echo -e "\n[5/7] Checking Terraform resources..."
TERRAFORM_RESOURCES=(
    "main.tf:resource \"aws_s3_bucket\" \"data_bucket\""
    "main.tf:resource \"aws_s3_bucket\" \"model_bucket\""
    "main.tf:resource \"aws_ecr_repository\" \"app_repository\""
    "main.tf:resource \"aws_ecs_cluster\" \"app_cluster\""
    "main.tf:resource \"aws_ecs_task_definition\" \"app_task\""
    "main.tf:resource \"aws_ecs_service\" \"app_service\""
    "main.tf:resource \"aws_lb\" \"app_lb\""
    "modules/vpc/main.tf:resource \"aws_vpc\" \"main\""
)

MISSING_RESOURCES=0
for RESOURCE in "${TERRAFORM_RESOURCES[@]}"; do
    FILE=$(echo $RESOURCE | cut -d':' -f1)
    PATTERN=$(echo $RESOURCE | cut -d':' -f2)
    if [ -f "ftth_fault_detection/infrastructure/terraform/$FILE" ]; then
        if grep -q "$PATTERN" "ftth_fault_detection/infrastructure/terraform/$FILE"; then
            echo "✅ Terraform resource $PATTERN found in $FILE"
        else
            echo "❌ Terraform resource $PATTERN not found in $FILE"
            MISSING_RESOURCES=$((MISSING_RESOURCES+1))
        fi
    else
        echo "❌ File $FILE not found, cannot check for Terraform resource $PATTERN"
        MISSING_RESOURCES=$((MISSING_RESOURCES+1))
    fi
done

if [ $MISSING_RESOURCES -eq 0 ]; then
    echo "✅ All required Terraform resources are present."
else
    echo "❌ $MISSING_RESOURCES Terraform resources are missing."
fi

# Check Ansible tasks
echo -e "\n[6/7] Checking Ansible tasks..."
ANSIBLE_TASKS=(
    "deploy.yml:Build Docker image"
    "deploy.yml:Push Docker image to ECR"
    "deploy.yml:Upload models to S3"
    "deploy.yml:Apply Terraform configuration"
)

MISSING_TASKS=0
for TASK in "${ANSIBLE_TASKS[@]}"; do
    FILE=$(echo $TASK | cut -d':' -f1)
    PATTERN=$(echo $TASK | cut -d':' -f2)
    if [ -f "ftth_fault_detection/infrastructure/ansible/$FILE" ]; then
        if grep -q "$PATTERN" "ftth_fault_detection/infrastructure/ansible/$FILE"; then
            echo "✅ Ansible task $PATTERN found in $FILE"
        else
            echo "❌ Ansible task $PATTERN not found in $FILE"
            MISSING_TASKS=$((MISSING_TASKS+1))
        fi
    else
        echo "❌ File $FILE not found, cannot check for Ansible task $PATTERN"
        MISSING_TASKS=$((MISSING_TASKS+1))
    fi
done

if [ $MISSING_TASKS -eq 0 ]; then
    echo "✅ All required Ansible tasks are present."
else
    echo "❌ $MISSING_TASKS Ansible tasks are missing."
fi

# Check Jenkins pipeline stages
echo -e "\n[7/7] Checking Jenkins pipeline stages..."
JENKINS_STAGES=(
    "Jenkinsfile:stage('Checkout')"
    "Jenkinsfile:stage('Install Dependencies')"
    "Jenkinsfile:stage('Run Tests')"
    "Jenkinsfile:stage('Train Models')"
    "Jenkinsfile:stage('Build Docker Image')"
    "Jenkinsfile:stage('Upload Models to S3')"
    "Jenkinsfile:stage('Deploy Infrastructure')"
    "Jenkinsfile:stage('Deploy Application with Ansible')"
)

MISSING_STAGES=0
for STAGE in "${JENKINS_STAGES[@]}"; do
    FILE=$(echo $STAGE | cut -d':' -f1)
    PATTERN=$(echo $STAGE | cut -d':' -f2)
    if [ -f "ftth_fault_detection/$FILE" ]; then
        if grep -q "$PATTERN" "ftth_fault_detection/$FILE"; then
            echo "✅ Jenkins stage $PATTERN found in $FILE"
        else
            echo "❌ Jenkins stage $PATTERN not found in $FILE"
            MISSING_STAGES=$((MISSING_STAGES+1))
        fi
    else
        echo "❌ File $FILE not found, cannot check for Jenkins stage $PATTERN"
        MISSING_STAGES=$((MISSING_STAGES+1))
    fi
done

if [ $MISSING_STAGES -eq 0 ]; then
    echo "✅ All required Jenkins pipeline stages are present."
else
    echo "❌ $MISSING_STAGES Jenkins pipeline stages are missing."
fi

# Final validation summary
echo -e "\n=================================================================="
echo "VALIDATION SUMMARY:"
echo "------------------------------------------------------------------"

if [ $MISSING_DIRS -eq 0 ] && [ $MISSING_FILES -eq 0 ] && [ $MISSING_MODULES -eq 0 ] && [ $MISSING_ENDPOINTS -eq 0 ] && [ $MISSING_RESOURCES -eq 0 ] && [ $MISSING_TASKS -eq 0 ] && [ $MISSING_STAGES -eq 0 ]; then
    echo "✅ VALIDATION PASSED: All components are properly implemented and integrated."
else
    echo "❌ VALIDATION FAILED: Some components are missing or not properly implemented."
    echo "   - Missing directories: $MISSING_DIRS"
    echo "   - Missing files: $MISSING_FILES"
    echo "   - Missing Python modules: $MISSING_MODULES"
    echo "   - Missing FastAPI endpoints: $MISSING_ENDPOINTS"
    echo "   - Missing Terraform resources: $MISSING_RESOURCES"
    echo "   - Missing Ansible tasks: $MISSING_TASKS"
    echo "   - Missing Jenkins pipeline stages: $MISSING_STAGES"
fi

echo "=================================================================="
