#!/bin/bash
set -e

# BEPS RAG System GPU Deployment Script
# This script sets up and deploys the hierarchical RAG system using vLLM on GPU

echo "🚀 Starting BEPS RAG System GPU Deployment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is not installed. Please install Docker Compose first.${NC}"
    exit 1
fi

# Check for NVIDIA Container Toolkit
if ! docker info | grep -q "nvidia"; then
    echo -e "${RED}❌ NVIDIA Container Toolkit not found. Please install it:${NC}"
    echo "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    exit 1
fi

# Check GPU availability
echo -e "${YELLOW}🔍 Checking GPU availability...${NC}"
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✅ NVIDIA GPU detected:${NC}"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
else
    echo -e "${RED}❌ NVIDIA GPU not detected or drivers not installed${NC}"
    exit 1
fi

# Create necessary directories
echo -e "${YELLOW}📁 Creating directories...${NC}"
mkdir -p models data logs config

# Download default model if not exists
MODEL_NAME="microsoft/DialoGPT-medium"
MODEL_DIR="models/microsoft--DialoGPT-medium"

if [ ! -d "${MODEL_DIR}" ]; then
    echo -e "${YELLOW}📥 Downloading default model: ${MODEL_NAME}...${NC}"
    
    # Install git-lfs if not available
    if ! command -v git-lfs &> /dev/null; then
        echo -e "${YELLOW}📦 Installing git-lfs...${NC}"
        if command -v apt-get &> /dev/null; then
            sudo apt-get update && sudo apt-get install -y git-lfs
        elif command -v yum &> /dev/null; then
            sudo yum install -y git-lfs
        elif command -v brew &> /dev/null; then
            brew install git-lfs
        fi
        git lfs install
    fi
    
    # Clone the model
    cd models
    git lfs clone https://huggingface.co/${MODEL_NAME} microsoft--DialoGPT-medium
    cd ..
    
    echo -e "${GREEN}✅ Model downloaded successfully${NC}"
else
    echo -e "${GREEN}✅ Model already exists${NC}"
fi

# Copy configuration files
echo -e "${YELLOW}⚙️  Setting up configuration...${NC}"
cp ../../config/processing_config.yaml config/

# Build and start services
echo -e "${YELLOW}🐳 Building and starting Docker containers...${NC}"
docker-compose up --build -d

# Wait for services to be ready
echo -e "${YELLOW}⏳ Waiting for services to start...${NC}"
sleep 60

# Check if services are running
if docker-compose ps | grep -q "Up"; then
    echo -e "${GREEN}✅ Services are running successfully!${NC}"
    echo -e "${GREEN}🌐 API is available at: http://localhost:8000${NC}"
    echo -e "${GREEN}📊 Health check: http://localhost:8000/health${NC}"
    echo -e "${GREEN}🎮 GPU info: http://localhost:8000/gpu_info${NC}"
else
    echo -e "${RED}❌ Services failed to start. Check logs with: docker-compose logs${NC}"
    exit 1
fi

# Display useful commands
echo ""
echo -e "${YELLOW}📋 Useful commands:${NC}"
echo "  View logs: docker-compose logs -f"
echo "  Stop services: docker-compose down"
echo "  Restart services: docker-compose restart"
echo "  Access container: docker-compose exec beps-gpu bash"
echo "  Monitor GPU: docker-compose exec beps-gpu nvidia-smi"
echo ""
echo -e "${BLUE}🎯 Performance tips:${NC}"
echo "  - Monitor GPU memory usage: nvidia-smi"
echo "  - Adjust batch size in config for your GPU memory"
echo "  - Use smaller models if running out of memory"
echo ""
echo -e "${GREEN}🎉 Deployment complete!${NC}"