# Deployment Guide

This directory contains deployment scripts and configurations for running the hierarchical RAG system with different inference backends.

## Deployment Options

### 1. CPU Deployment with llama.cpp
- **Location**: `deployment/cpu/`
- **Backend**: llama.cpp via llama-cpp-python
- **Use Case**: Cost-effective CPU inference
- **Requirements**: CPU-only machine

### 2. GPU Deployment with vLLM
- **Location**: `deployment/gpu/`
- **Backend**: vLLM for high-throughput GPU inference
- **Use Case**: High-performance GPU inference
- **Requirements**: NVIDIA GPU with CUDA support

## Quick Start

Choose your deployment method:

```bash
# For CPU deployment
cd deployment/cpu
./deploy_cpu.sh

# For GPU deployment
cd deployment/gpu
./deploy_gpu.sh
```

## Configuration

Each deployment includes:
- Docker configurations
- Environment setup scripts
- Model download utilities
- Health check endpoints
- Monitoring setup

## System Requirements

### CPU Deployment
- 8GB+ RAM recommended
- 10GB+ disk space for models
- Docker and Docker Compose

### GPU Deployment
- NVIDIA GPU with 8GB+ VRAM
- CUDA 11.8+ and cuDNN
- 16GB+ RAM recommended
- Docker with NVIDIA Container Toolkit