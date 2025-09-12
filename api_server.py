"""
Production-ready FastAPI server for DeepRAG Clinical Question Answering
"""

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
import asyncio
import logging
import time
import hashlib
import json
from typing import Optional
import uvicorn

# Import your pipeline modules
from deeprag_pipeline import DeepRAGPipeline
from config import Config

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="DeepRAG Clinical API",
    description="Advanced RAG system for MIMIC-III clinical question answering using DeepRAG methodology",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline = None

# Request/Response models
class QuestionRequest(BaseModel):
    question: str
    use_deeprag: bool = True
    log_details: bool = False
    max_tokens: Optional[int] = 500

class QuestionResponse(BaseModel):
    answer: str
    success: bool
    latency_ms: float
    retrievals: int
    confidence: Optional[float] = None
    model_used: str
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    model: str
    version: str
    uptime_seconds: float

# Global variables for metrics
app_start_time = time.time()
request_count = 0
total_latency = 0.0

@app.on_event("startup")
async def startup_event():
    """Initialize the DeepRAG pipeline on startup"""
    global pipeline
    
    try:
        logger.info("Initializing DeepRAG Clinical Pipeline...")
        pipeline = DeepRAGPipeline()
        logger.info("✅ DeepRAG pipeline successfully initialized")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize pipeline: {e}")
        raise e

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down DeepRAG Clinical API")

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "DeepRAG Clinical Question Answering API",
        "status": "healthy",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    uptime = time.time() - app_start_time
    
    return HealthResponse(
        status="healthy" if pipeline else "unhealthy",
        model=Config.DEFAULT_MODEL,
        version="1.0.0",
        uptime_seconds=round(uptime, 2)
    )

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Process a clinical question using DeepRAG methodology
    
    - **question**: The clinical question to answer
    - **use_deeprag**: Whether to use DeepRAG (MDP) reasoning (default: True)
    - **log_details**: Whether to log detailed processing information
    - **max_tokens**: Maximum tokens for the response (optional)
    """
    global request_count, total_latency
    
    if not pipeline:
        raise HTTPException(
            status_code=503, 
            detail="DeepRAG pipeline not initialized. Please check server logs."
        )
    
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )
    
    start_time = time.time()
    request_count += 1
    
    try:
        logger.info(f"Processing question: {request.question[:100]}...")
        
        # Process the question
        result = pipeline.process_question(
            request.question, 
            use_deeprag=request.use_deeprag,
            log_details=request.log_details
        )
        
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        total_latency += latency_ms
        
        # Log the interaction
        logger.info(
            f"Question processed successfully - "
            f"Latency: {latency_ms:.1f}ms, "
            f"Success: {result.get('success', False)}, "
            f"Retrievals: {result.get('retrievals', 0)}"
        )
        
        return QuestionResponse(
            answer=result.get("answer") or "No answer generated",
            success=result.get("success", False),
            latency_ms=round(latency_ms, 2),
            retrievals=result.get("retrievals", 0),
            confidence=result.get("confidence"),
            model_used=Config.DEFAULT_MODEL,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/metrics")
async def get_metrics():
    """
    Basic metrics endpoint for monitoring
    """
    uptime = time.time() - app_start_time
    avg_latency = total_latency / request_count if request_count > 0 else 0
    
    return {
        "uptime_seconds": round(uptime, 2),
        "total_requests": request_count,
        "avg_latency_ms": round(avg_latency, 2),
        "requests_per_second": round(request_count / uptime, 2) if uptime > 0 else 0,
        "pipeline_status": "healthy" if pipeline else "unhealthy",
        "model": Config.DEFAULT_MODEL
    }

@app.get("/info")
async def get_info():
    """
    System information endpoint
    """
    return {
        "system": {
            "model": Config.DEFAULT_MODEL,
            "chunk_size": Config.CHUNK_SIZE,
            "chunk_overlap": Config.CHUNK_OVERLAP,
            "retriever_k": Config.RETRIEVER_K
        },
        "capabilities": {
            "deeprag_reasoning": True,
            "clinical_data": "MIMIC-III",
            "conditions_supported": ["HAPI", "HAAKI", "HAA"],
            "max_concurrent_requests": 10
        },
        "endpoints": {
            "ask": "/ask - Process clinical questions",
            "health": "/health - Health check",
            "metrics": "/metrics - Performance metrics",
            "docs": "/docs - API documentation"
        }
    }

# Add middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Process the request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    # Log the request
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time*1000:.1f}ms"
    )
    
    # Add timing header
    response.headers["X-Process-Time"] = str(round(process_time * 1000, 2))
    
    return response

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return Response(
        content=json.dumps({
            "error": "Endpoint not found",
            "message": f"The endpoint {request.url.path} does not exist",
            "available_endpoints": ["/", "/health", "/ask", "/metrics", "/info", "/docs"]
        }),
        status_code=404,
        media_type="application/json"
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error(f"Internal server error: {exc}")
    return Response(
        content=json.dumps({
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please check server logs."
        }),
        status_code=500,
        media_type="application/json"
    )

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False,  # Set to True for development
        workers=1      # Increase for production
    )