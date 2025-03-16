pipeline {
    agent any
    
    environment {
        PROJECT_NAME = 'ftth-fault-detection'
        AWS_REGION = 'us-east-1'
        ECR_REPOSITORY_URL = ''
        MODEL_BUCKET_NAME = ''
        DATA_BUCKET_NAME = ''
        ENVIRONMENT = 'production'
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Install Dependencies') {
            steps {
                sh '''
                    python3 -m pip install --upgrade pip
                    python3 -m pip install -r src/api/requirements.txt
                    python3 -m pip install pytest pytest-cov
                '''
            }
        }
        
        stage('Run Tests') {
            steps {
                sh '''
                    # Run unit tests with coverage
                    python3 -m pytest --cov=src tests/
                '''
            }
        }
        
        stage('Train Models') {
            when {
                expression { params.TRAIN_MODELS == true }
            }
            steps {
                sh '''
                    # Train autoencoder model
                    python3 src/training/train_autoencoder.py --data_path data/OTDR_data.csv --output_dir models --epochs 50 --batch_size 32
                    
                    # Train BiGRU-Attention model
                    python3 src/training/train_bigru_attention.py --data_path data/OTDR_data.csv --output_dir models --epochs 50 --batch_size 32 --include_snr
                    
                    # Create symbolic links for latest models
                    mkdir -p models/autoencoder_latest
                    mkdir -p models/bigru_attention_latest
                    
                    # Find latest model directories
                    LATEST_AUTOENCODER=$(ls -td models/autoencoder_* | grep -v latest | head -1)
                    LATEST_BIGRU=$(ls -td models/bigru_attention_* | grep -v latest | head -1)
                    
                    # Create symbolic links
                    ln -sf $LATEST_AUTOENCODER/autoencoder_model.h5 models/autoencoder_latest/autoencoder_model.h5
                    ln -sf $LATEST_AUTOENCODER/model_config.npy models/autoencoder_latest/model_config.npy
                    
                    ln -sf $LATEST_BIGRU/bigru_attention_model.h5 models/bigru_attention_latest/bigru_attention_model.h5
                    ln -sf $LATEST_BIGRU/model_config.npy models/bigru_attention_latest/model_config.npy
                '''
            }
        }
        
        stage('Build Docker Image') {
            steps {
                script {
                    // Build Docker image
                    sh 'docker build -t ${PROJECT_NAME}:${BUILD_NUMBER} .'
                    
                    // Get ECR repository URL from Terraform output
                    def terraformOutput = sh(
                        script: 'cd infrastructure/terraform && terraform output -json',
                        returnStdout: true
                    ).trim()
                    
                    def outputs = readJSON text: terraformOutput
                    env.ECR_REPOSITORY_URL = outputs.ecr_repository_url.value
                    env.MODEL_BUCKET_NAME = outputs.model_bucket_name.value
                    env.DATA_BUCKET_NAME = outputs.data_bucket_name.value
                    
                    // Login to ECR
                    sh '''
                        aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_REPOSITORY_URL.split('/')[0]}
                    '''
                    
                    // Tag and push Docker image
                    sh '''
                        docker tag ${PROJECT_NAME}:${BUILD_NUMBER} ${ECR_REPOSITORY_URL}:${BUILD_NUMBER}
                        docker tag ${PROJECT_NAME}:${BUILD_NUMBER} ${ECR_REPOSITORY_URL}:latest
                        docker push ${ECR_REPOSITORY_URL}:${BUILD_NUMBER}
                        docker push ${ECR_REPOSITORY_URL}:latest
                    '''
                }
            }
        }
        
        stage('Upload Models to S3') {
            when {
                expression { params.TRAIN_MODELS == true }
            }
            steps {
                sh '''
                    # Upload models to S3
                    aws s3 sync models/ s3://${MODEL_BUCKET_NAME}/models/ --delete
                '''
            }
        }
        
        stage('Deploy Infrastructure') {
            steps {
                script {
                    // Initialize Terraform
                    sh 'cd infrastructure/terraform && terraform init'
                    
                    // Apply Terraform configuration
                    sh '''
                        cd infrastructure/terraform && terraform apply -auto-approve \
                        -var="aws_region=${AWS_REGION}" \
                        -var="environment=${ENVIRONMENT}" \
                        -var="project_name=${PROJECT_NAME}"
                    '''
                }
            }
        }
        
        stage('Deploy Application with Ansible') {
            steps {
                sh '''
                    cd infrastructure/ansible && \
                    ansible-playbook deploy.yml \
                    -e "aws_region=${AWS_REGION}" \
                    -e "environment=${ENVIRONMENT}" \
                    -e "project_name=${PROJECT_NAME}" \
                    -e "ecr_repository_url=${ECR_REPOSITORY_URL}" \
                    -e "model_bucket_name=${MODEL_BUCKET_NAME}" \
                    -e "data_bucket_name=${DATA_BUCKET_NAME}"
                '''
            }
        }
    }
    
    post {
        success {
            echo 'Deployment completed successfully!'
            
            script {
                // Get load balancer DNS from Terraform output
                def terraformOutput = sh(
                    script: 'cd infrastructure/terraform && terraform output -json',
                    returnStdout: true
                ).trim()
                
                def outputs = readJSON text: terraformOutput
                def loadBalancerDns = outputs.load_balancer_dns.value
                
                echo "Application is deployed and available at: http://${loadBalancerDns}"
            }
        }
        failure {
            echo 'Deployment failed!'
        }
        always {
            // Clean up Docker images
            sh '''
                docker rmi ${PROJECT_NAME}:${BUILD_NUMBER} || true
                docker rmi ${ECR_REPOSITORY_URL}:${BUILD_NUMBER} || true
                docker rmi ${ECR_REPOSITORY_URL}:latest || true
            '''
        }
    }
    
    parameters {
        booleanParam(name: 'TRAIN_MODELS', defaultValue: false, description: 'Train models before deployment')
    }
}
