# Multi-stage Dockerfile for Hierarchical RAG Document Processing Pipeline

# Build stage
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY setup.py .
COPY README.md .

# Install the package
RUN pip install -e .

# Production stage
FROM python:3.11-slim as production

# Install system dependencies for PDF processing
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash app

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --from=builder /app/src ./src
COPY --from=builder /app/config ./config
COPY --from=builder /app/examples ./examples

# Create necessary directories
RUN mkdir -p /app/data /app/output /app/logs && \
    chown -R app:app /app

# Switch to non-root user
USER app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port for potential web interface
EXPOSE 8000

# Default command
CMD ["python", "examples/basic_usage.py"]

# Development stage
FROM python:3.11-slim as development

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    libmagic1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies including dev dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir pytest pytest-asyncio black flake8 mypy

# Copy all source code
COPY . .

# Install in development mode
RUN pip install -e ".[dev]"

# Create necessary directories
RUN mkdir -p /app/data /app/output /app/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command for development
CMD ["python", "run_tests.py"]