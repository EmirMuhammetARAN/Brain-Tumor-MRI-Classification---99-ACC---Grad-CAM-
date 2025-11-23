# Deployment Guide

## Table of Contents
1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Infrastructure Requirements](#infrastructure-requirements)
3. [Docker Deployment](#docker-deployment)
4. [Cloud Deployment](#cloud-deployment)
5. [Monitoring & Logging](#monitoring--logging)
6. [Troubleshooting](#troubleshooting)

## Pre-Deployment Checklist

### ✅ Technical Requirements
- [ ] Python 3.10+ installed
- [ ] Docker & Docker Compose installed (for containerized deployment)
- [ ] GPU support configured (optional but recommended)
- [ ] Minimum 4GB RAM available
- [ ] 5GB disk space for model and dependencies

### ✅ Model Validation
- [ ] Clinical validation metrics reviewed
- [ ] Error analysis completed
- [ ] Performance benchmarks meet requirements (<500ms inference)
- [ ] Test suite passing (90%+ coverage)

### ✅ Security
- [ ] API authentication configured (if public-facing)
- [ ] HTTPS/TLS certificates obtained
- [ ] Rate limiting configured
- [ ] Input validation tested
- [ ] HIPAA compliance reviewed (if applicable)

### ✅ Documentation
- [ ] Model limitations documented
- [ ] Clinical workflow defined
- [ ] Incident response plan created
- [ ] User training materials prepared

## Infrastructure Requirements

### Minimum Specifications

**Development/Testing:**
- CPU: 4 cores
- RAM: 8GB
- Storage: 10GB
- GPU: Optional

**Production:**
- CPU: 8+ cores
- RAM: 16GB+
- Storage: 50GB+ (includes logs)
- GPU: NVIDIA GPU with 8GB+ VRAM (recommended)
- Network: 100Mbps+

### Recommended Cloud Instances

**AWS:**
- `g4dn.xlarge` (GPU) or `c5.2xlarge` (CPU)
- EBS storage: 50GB gp3

**Google Cloud:**
- `n1-standard-4` with NVIDIA T4
- Persistent disk: 50GB SSD

**Azure:**
- `Standard_NC6s_v3` (GPU) or `Standard_D4s_v3` (CPU)

## Docker Deployment

### 1. Local Docker Deployment

```bash
# Clone repository
git clone https://github.com/your-username/BrainTumor-Classification-CNN
cd BrainTumor-Classification-CNN

# Build image
docker build -t brain-tumor-api:latest .

# Run container
docker run -d \
  --name brain-tumor-classifier \
  -p 8000:8000 \
  -v $(pwd)/logs:/app/logs \
  brain-tumor-api:latest

# Check logs
docker logs -f brain-tumor-classifier

# Test health endpoint
curl http://localhost:8000/health
```

### 2. Docker Compose (Production)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Scale API service
docker-compose up -d --scale brain-tumor-api=3

# Stop services
docker-compose down
```

### 3. Docker Compose with GPU

```yaml
# docker-compose.gpu.yml
version: '3.8'

services:
  brain-tumor-api:
    build: .
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

```bash
docker-compose -f docker-compose.gpu.yml up -d
```

## Cloud Deployment

### AWS Deployment (ECS/Fargate)

#### 1. Push to ECR

```bash
# Authenticate
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Create repository
aws ecr create-repository --repository-name brain-tumor-classifier

# Tag and push
docker tag brain-tumor-api:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/brain-tumor-classifier:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/brain-tumor-classifier:latest
```

#### 2. Create ECS Task Definition

```json
{
  "family": "brain-tumor-classifier",
  "networkMode": "awsvpc",
  "containerDefinitions": [
    {
      "name": "api",
      "image": "<account-id>.dkr.ecr.us-east-1.amazonaws.com/brain-tumor-classifier:latest",
      "cpu": 2048,
      "memory": 4096,
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "LOG_LEVEL",
          "value": "INFO"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/brain-tumor-classifier",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ],
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096"
}
```

#### 3. Create Service

```bash
aws ecs create-service \
  --cluster production \
  --service-name brain-tumor-api \
  --task-definition brain-tumor-classifier \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
```

### Google Cloud Run Deployment

```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/PROJECT_ID/brain-tumor-api

# Deploy to Cloud Run
gcloud run deploy brain-tumor-api \
  --image gcr.io/PROJECT_ID/brain-tumor-api \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2 \
  --max-instances 10 \
  --allow-unauthenticated
```

### Azure Container Instances

```bash
# Create resource group
az group create --name brain-tumor-rg --location eastus

# Create container registry
az acr create --resource-group brain-tumor-rg --name braintumoracr --sku Basic

# Push image
az acr login --name braintumoracr
docker tag brain-tumor-api:latest braintumoracr.azurecr.io/brain-tumor-api:latest
docker push braintumoracr.azurecr.io/brain-tumor-api:latest

# Deploy container
az container create \
  --resource-group brain-tumor-rg \
  --name brain-tumor-api \
  --image braintumoracr.azurecr.io/brain-tumor-api:latest \
  --cpu 2 \
  --memory 4 \
  --registry-login-server braintumoracr.azurecr.io \
  --ports 8000 \
  --dns-name-label brain-tumor-api
```

## Monitoring & Logging

### Application Logging

Logs are written to:
- Console (stdout/stderr)
- `logs/inference.log` (file)

Configure log level in `config.json`:
```json
{
  "logging": {
    "level": "INFO",  // DEBUG, INFO, WARNING, ERROR
    "format": "json"
  }
}
```

### Health Checks

```bash
# Basic health
curl http://localhost:8000/health

# Detailed performance stats
curl http://localhost:8000/performance
```

### Prometheus Metrics (Optional)

Add to `requirements-prod.txt`:
```
prometheus-client==0.19.0
```

Add to `api.py`:
```python
from prometheus_client import Counter, Histogram, make_asgi_app

# Metrics
prediction_counter = Counter('predictions_total', 'Total predictions')
inference_time = Histogram('inference_seconds', 'Inference time')

# Mount metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
```

### Log Aggregation

**ELK Stack (Elasticsearch, Logstash, Kibana):**
```yaml
# docker-compose.logging.yml
version: '3.8'
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    environment:
      - discovery.type=single-node
    ports:
      - "9200:9200"
  
  kibana:
    image: docker.elastic.co/kibana/kibana:8.11.0
    ports:
      - "5601:5601"
    depends_on:
      - elasticsearch
```

## Troubleshooting

### Common Issues

#### 1. Model fails to load
```
Error: "Model weights not found"
```

**Solution:**
```bash
# Verify weights file exists
ls -lh best_weights_balanced.h5

# Check file permissions
chmod 644 best_weights_balanced.h5

# Verify in Docker
docker exec -it brain-tumor-classifier ls -lh /app/best_weights_balanced.h5
```

#### 2. Out of memory errors
```
Error: "ResourceExhaustedError: OOM when allocating tensor"
```

**Solution:**
```python
# In config.json, reduce batch size
"inference": {
  "batch_size": 8  # Reduce from 16
}

# Or limit TensorFlow memory growth
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
```

#### 3. Slow inference times
```
P95 inference time > 500ms
```

**Solution:**
```bash
# Enable GPU
docker run --gpus all ...

# Use TensorRT optimization (advanced)
python -c "import tensorflow as tf; tf.experimental.tensorrt.Converter(...)"

# Reduce image size in preprocessing
# Update config.json: "image_size": [160, 160]
```

#### 4. API connection timeouts

**Solution:**
```python
# In api.py, increase timeout
uvicorn.run(..., timeout_keep_alive=120)

# Or in docker-compose.yml
environment:
  - TIMEOUT=120
```

#### 5. Container keeps restarting

**Solution:**
```bash
# Check logs
docker logs brain-tumor-classifier

# Common causes:
# - Port already in use: change port mapping
# - Missing dependencies: rebuild image
# - Config file not found: check volume mounts
```

### Performance Tuning

#### CPU Optimization
```python
# Set number of threads
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)
```

#### GPU Optimization
```python
# Mixed precision for faster inference
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
```

#### Batch Processing
```python
# Process multiple images at once
results = model_inference.batch_predict(images)  # More efficient
```

### Disaster Recovery

#### Backup Strategy
```bash
# Backup model weights
aws s3 cp best_weights_balanced.h5 s3://backup-bucket/models/

# Backup logs
tar -czf logs_$(date +%Y%m%d).tar.gz logs/
aws s3 cp logs_*.tar.gz s3://backup-bucket/logs/
```

#### Rollback Procedure
```bash
# Tag current version
docker tag brain-tumor-api:latest brain-tumor-api:v1.0.1

# If issues arise, rollback
docker run brain-tumor-api:v1.0.0  # Previous stable version

# In Kubernetes
kubectl rollout undo deployment/brain-tumor-api
```

## Security Best Practices

### 1. API Authentication
```python
# Add to api.py
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

@app.post("/predict")
async def predict(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    # Verify token
    ...
```

### 2. Rate Limiting
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/predict")
@limiter.limit("10/minute")
async def predict():
    ...
```

### 3. HTTPS Setup
```bash
# Using Let's Encrypt with Nginx
docker run -d \
  --name nginx \
  -p 443:443 \
  -v /etc/letsencrypt:/etc/letsencrypt \
  nginx
```

## Support

For issues or questions:
- GitHub Issues: [Repository Issues]
- Email: [support@example.com]
- Documentation: [docs.example.com]
