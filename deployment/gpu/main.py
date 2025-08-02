#!/usr/bin/env python3
"""
GPU deployment main entry point for hierarchical RAG system using vLLM
"""
import os
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from src.agent.beps_agent import BEPSAgent
from src.config.config_loader import ConfigLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="BEPS RAG System - GPU Deployment",
    description="Hierarchical RAG system with vLLM backend",
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

# Global agent instance
agent: Optional[BEPSAgent] = None

class QueryRequest(BaseModel):
    query: str
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9

class QueryResponse(BaseModel):
    response: str
    confidence: float
    sources: list
    processing_time: float

@app.on_event("startup")
async def startup_event():
    """Initialize the agent on startup"""
    global agent
    
    logger.info("Starting BEPS RAG System - GPU Deployment")
    
    # Load configuration
    config_path = Path("/app/config/processing_config.yaml")
    if not config_path.exists():
        config_path = Path("config/processing_config.yaml")
    
    config = ConfigLoader.load_config(str(config_path))
    
    # Configure for GPU usage with vLLM
    config.llm.device = "cuda"
    config.llm.model_name = os.getenv("MODEL_NAME", "microsoft/DialoGPT-medium")
    config.llm.max_tokens = int(os.getenv("MAX_TOKENS", 2048))
    config.llm.temperature = float(os.getenv("TEMPERATURE", 0.7))
    
    # Initialize agent
    agent = BEPSAgent(config)
    logger.info("Agent initialized successfully with vLLM backend")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "deployment": "gpu", "cuda_available": True}

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a query using the RAG system"""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        result = await agent.process_query(
            query=request.query,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        return QueryResponse(
            response=result["response"],
            confidence=result["confidence"],
            sources=result.get("sources", []),
            processing_time=result.get("processing_time", 0.0)
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    """List available models"""
    models_dir = Path("/app/models")
    if models_dir.exists():
        models = [f.name for f in models_dir.iterdir() if f.is_dir() or f.suffix in ['.bin', '.safetensors']]
    else:
        models = []
    
    return {"available_models": models}

@app.post("/reload_model")
async def reload_model(model_name: str):
    """Reload a different model"""
    global agent
    
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        agent.config.llm.model_name = model_name
        await agent.reload_model()
        return {"status": "success", "model": model_name}
    except Exception as e:
        logger.error(f"Error reloading model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/gpu_info")
async def gpu_info():
    """Get GPU information"""
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "cuda_available": True,
                "gpu_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(),
                "memory_allocated": torch.cuda.memory_allocated(),
                "memory_reserved": torch.cuda.memory_reserved()
            }
        else:
            return {"cuda_available": False}
    except ImportError:
        return {"error": "PyTorch not available"}

if __name__ == "__main__":
    uvicorn.run(
        "deployment.gpu.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )