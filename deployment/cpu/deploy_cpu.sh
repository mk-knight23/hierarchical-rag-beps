#!/bin/bash
set -e

# BEPS RAG System CPU Deployment Script
# This script sets up and deploys the hierarchical RAG system using llama.cpp on CPU

echo "🚀 Starting BEPS RAG System CPU Deployment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

# Create necessary directories
echo -e "${YELLOW}📁 Creating directories...${NC}"
mkdir -p models data logs config

# Download default model if not exists
MODEL_NAME="llama-2-7b-chat.Q4_K_M.gguf"
MODEL_URL="https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/${MODEL_NAME}"

if [ ! -f "models/${MODEL_NAME}" ]; then
    echo -e "${YELLOW}📥 Downloading default model: ${MODEL_NAME}...${NC}"
    
    # Check if curl is available
    if command -v curl &> /dev/null; then
        curl -L -o "models/${MODEL_NAME}" "${MODEL_URL}"
    elif command -v wget &> /dev/null; then
        wget -O "models/${MODEL_NAME}" "${MODEL_URL}"
    else
        echo -e "${RED}❌ Neither curl nor wget found. Please install one of them.${NC}"
        exit 1
    fi
    
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
sleep 30

# Check if services are running
if docker-compose ps | grep -q "Up"; then
    echo -e "${GREEN}✅ Services are running successfully!${NC}"
    echo -e "${GREEN}🌐 API is available at: http://localhost:8000${NC}"
    echo -e "${GREEN}📊 Health check: http://localhost:8000/health${NC}"
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
echo "  Access container: docker-compose exec beps-cpu bash"
echo ""
echo -e "${GREEN}🎉 Deployment complete!${NC}"