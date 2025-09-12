# 🚀 DeepRAG Clinical Application Deployment Guide

This guide covers multiple deployment strategies for your DeepRAG clinical question-answering system.

## 📋 Table of Contents

1. [Local Development](#local-development)
2. [Docker Deployment](#docker-deployment)
3. [Cloud Deployment Options](#cloud-deployment-options)
4. [Production Considerations](#production-considerations)
5. [Monitoring & Scaling](#monitoring--scaling)

---

## 🏠 Local Development

### Quick Start
```bash
# Clone and setup
git clone https://github.com/DhruvMiyani/RAG-On-Clinical-Data.git
cd RAG-On-Clinical-Data

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env with your OpenAI API key

# Run the application
python3 deeprag_pipeline.py
```

### API Server (Flask/FastAPI)
```bash
# Install web framework
pip install fastapi uvicorn flask

# Run API server
python3 api_server.py
```

---

## 🐳 Docker Deployment

### 1. Create Dockerfile
```dockerfile
# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV LOG_LEVEL=INFO

# Run the application
CMD ["python", "api_server.py"]
```

### 2. Docker Compose Setup
```yaml
# docker-compose.yml
version: '3.8'

services:
  deeprag-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DEFAULT_MODEL=gpt-5
      - CHUNK_SIZE=750
      - CHUNK_OVERLAP=100
    volumes:
      - ./nosocomial-risk-datasets-from-mimic-iii-1.0:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Optional: Add Redis for caching
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    restart: unless-stopped

  # Optional: Add vector database
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
    volumes:
      - ./qdrant_storage:/qdrant/storage
    restart: unless-stopped
```

### 3. Build and Run
```bash
# Build and start
docker-compose up -d

# Check logs
docker-compose logs -f deeprag-api

# Scale replicas
docker-compose up -d --scale deeprag-api=3
```

---

## ☁️ Cloud Deployment Options

### 1. **AWS Deployment**

#### A. **AWS ECS (Recommended)**
```bash
# Install AWS CLI and ECS CLI
pip install awscli
curl -Lo ecs-cli https://amazon-ecs-cli.s3.amazonaws.com/ecs-cli-linux-amd64-latest
chmod +x ecs-cli && sudo mv ecs-cli /usr/local/bin

# Configure ECS
ecs-cli configure --cluster deeprag-cluster --default-launch-type EC2 --region us-east-1

# Deploy
ecs-cli compose --file docker-compose-aws.yml service up
```

#### B. **AWS Lambda (Serverless)**
```python
# lambda_handler.py
import json
from deeprag_pipeline import DeepRAGPipeline

# Initialize once (cold start optimization)
pipeline = None

def lambda_handler(event, context):
    global pipeline
    
    if pipeline is None:
        pipeline = DeepRAGPipeline()
    
    question = event.get('question', '')
    result = pipeline.process_question(question, use_deeprag=True)
    
    return {
        'statusCode': 200,
        'body': json.dumps(result),
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        }
    }
```

#### C. **AWS Fargate**
```yaml
# docker-compose-aws.yml
version: '3'
services:
  deeprag:
    image: your-account.dkr.ecr.us-east-1.amazonaws.com/deeprag:latest
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY
    logging:
      driver: awslogs
      options:
        awslogs-group: /ecs/deeprag
        awslogs-region: us-east-1
        awslogs-stream-prefix: ecs
```

### 2. **Google Cloud Platform**

#### A. **Cloud Run (Serverless)**
```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/PROJECT-ID/deeprag

# Deploy to Cloud Run
gcloud run deploy deeprag-api \
  --image gcr.io/PROJECT-ID/deeprag \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars OPENAI_API_KEY=$OPENAI_API_KEY
```

#### B. **GKE (Kubernetes)**
```yaml
# k8s-deployment.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: deeprag-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: deeprag
  template:
    metadata:
      labels:
        app: deeprag
    spec:
      containers:
      - name: deeprag
        image: gcr.io/PROJECT-ID/deeprag:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: openai-secret
              key: api-key
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"

---
apiVersion: v1
kind: Service
metadata:
  name: deeprag-service
spec:
  selector:
    app: deeprag
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### 3. **Microsoft Azure**

#### A. **Azure Container Instances**
```bash
# Create resource group
az group create --name deeprag-rg --location eastus

# Deploy container
az container create \
  --resource-group deeprag-rg \
  --name deeprag-app \
  --image your-registry/deeprag:latest \
  --dns-name-label deeprag-clinical \
  --ports 8000 \
  --environment-variables OPENAI_API_KEY=$OPENAI_API_KEY
```

#### B. **Azure App Service**
```bash
# Create App Service plan
az appservice plan create --name deeprag-plan --resource-group deeprag-rg --is-linux

# Create web app
az webapp create --resource-group deeprag-rg --plan deeprag-plan --name deeprag-clinical --deployment-container-image-name your-registry/deeprag:latest
```

### 4. **Heroku (Simple)**
```bash
# Install Heroku CLI
# Create Procfile
echo "web: python api_server.py" > Procfile

# Deploy
heroku create deeprag-clinical
heroku config:set OPENAI_API_KEY=$OPENAI_API_KEY
git push heroku main
```

### 5. **DigitalOcean App Platform**
```yaml
# .do/app.yaml
name: deeprag-clinical
services:
- name: api
  source_dir: /
  github:
    repo: DhruvMiyani/RAG-On-Clinical-Data
    branch: main
  run_command: python api_server.py
  environment_slug: python
  instance_count: 1
  instance_size_slug: basic-xxs
  envs:
  - key: OPENAI_API_KEY
    value: ${OPENAI_API_KEY}
    type: SECRET
  http_port: 8000
```

---

## 🏗️ Production API Server

Create a production-ready API server:

```python
# api_server.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import logging
from deeprag_pipeline import DeepRAGPipeline
from config import Config

# Initialize FastAPI app
app = FastAPI(
    title="DeepRAG Clinical API",
    description="Advanced RAG system for clinical question answering",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline (singleton)
pipeline = None

class QuestionRequest(BaseModel):
    question: str
    use_deeprag: bool = True
    max_tokens: int = 500

class QuestionResponse(BaseModel):
    answer: str
    success: bool
    latency_ms: float
    retrievals: int
    confidence: float

@app.on_event("startup")
async def startup_event():
    global pipeline
    pipeline = DeepRAGPipeline()
    logging.info("DeepRAG pipeline initialized")

@app.get("/")
async def root():
    return {"message": "DeepRAG Clinical API", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": Config.DEFAULT_MODEL}

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    try:
        result = pipeline.process_question(
            request.question, 
            use_deeprag=request.use_deeprag
        )
        
        return QuestionResponse(
            answer=result["answer"],
            success=result["success"],
            latency_ms=result["latency_ms"],
            retrievals=result.get("retrievals", 0),
            confidence=result.get("confidence", 0.0)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    # Add your metrics collection here
    return {
        "total_requests": 0,
        "avg_latency": 0,
        "success_rate": 0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 📊 Production Considerations

### 1. **Environment Variables**
```bash
# Production .env
OPENAI_API_KEY=your-production-key
DEFAULT_MODEL=gpt-5
LOG_LEVEL=WARNING
CHUNK_SIZE=750
CHUNK_OVERLAP=100
REDIS_URL=redis://redis:6379
DATABASE_URL=postgresql://user:pass@host:5432/db
MAX_CONCURRENT_REQUESTS=50
RATE_LIMIT_PER_MINUTE=100
```

### 2. **Caching Strategy**
```python
# Add Redis caching
import redis
import json
import hashlib

redis_client = redis.Redis.from_url(Config.REDIS_URL)

def get_cached_response(question: str):
    cache_key = hashlib.md5(question.encode()).hexdigest()
    cached = redis_client.get(f"deeprag:{cache_key}")
    if cached:
        return json.loads(cached)
    return None

def cache_response(question: str, response: dict):
    cache_key = hashlib.md5(question.encode()).hexdigest()
    redis_client.setex(
        f"deeprag:{cache_key}", 
        3600,  # 1 hour TTL
        json.dumps(response)
    )
```

### 3. **Database Integration**
```python
# Store query logs
from sqlalchemy import create_engine, Column, String, DateTime, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class QueryLog(Base):
    __tablename__ = "query_logs"
    
    id = Column(String, primary_key=True)
    question = Column(Text)
    answer = Column(Text)
    latency_ms = Column(Float)
    timestamp = Column(DateTime)
    success = Column(String)
```

### 4. **Load Balancer Configuration**
```nginx
# nginx.conf
upstream deeprag_backend {
    server deeprag-api-1:8000;
    server deeprag-api-2:8000;
    server deeprag-api-3:8000;
}

server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://deeprag_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }
}
```

---

## 📈 Monitoring & Scaling

### 1. **Application Monitoring**
```python
# Add Prometheus metrics
from prometheus_client import Counter, Histogram, generate_latest

REQUEST_COUNT = Counter('requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('request_latency_seconds', 'Request latency')

@app.middleware("http")
async def add_metrics(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path).inc()
    REQUEST_LATENCY.observe(process_time)
    
    return response

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

### 2. **Auto-scaling (Kubernetes)**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: deeprag-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: deeprag-deployment
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### 3. **Logging & Observability**
```python
import structlog

logger = structlog.get_logger()

# Structured logging
logger.info(
    "question_processed",
    question_length=len(question),
    response_time_ms=latency,
    model_used=Config.DEFAULT_MODEL,
    success=result["success"]
)
```

---

## 🔒 Security Considerations

### 1. **API Security**
```python
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Implement your authentication logic
    if not verify_api_token(credentials.credentials):
        raise HTTPException(status_code=401, detail="Invalid token")
```

### 2. **Rate Limiting**
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/ask")
@limiter.limit("10/minute")
async def ask_question(request: Request, question_req: QuestionRequest):
    # Your endpoint logic
    pass
```

---

## 💰 Cost Optimization

### 1. **OpenAI API Costs**
- Use caching to avoid duplicate queries
- Implement request deduplication
- Monitor token usage
- Set up usage alerts

### 2. **Infrastructure Costs**
- Use spot instances for non-critical workloads
- Implement auto-scaling
- Choose right-sized instances
- Use reserved instances for predictable workloads

---

## 🚀 Quick Deployment Commands

### Local Docker:
```bash
docker build -t deeprag-clinical .
docker run -p 8000:8000 --env-file .env deeprag-clinical
```

### AWS ECS:
```bash
aws ecs create-cluster --cluster-name deeprag-cluster
ecs-cli compose service up --cluster deeprag-cluster
```

### Google Cloud Run:
```bash
gcloud run deploy --image gcr.io/PROJECT/deeprag --allow-unauthenticated
```

### Kubernetes:
```bash
kubectl apply -f k8s-deployment.yml
kubectl expose deployment deeprag-deployment --type=LoadBalancer --port=80
```

---

Choose the deployment option that best fits your infrastructure requirements, budget, and scaling needs! 🎯