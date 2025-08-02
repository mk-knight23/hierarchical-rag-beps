FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install llama-cpp-python for CPU inference
RUN pip install llama-cpp-python --force-reinstall --no-cache-dir

# Copy application code
COPY . .

# Create directories for models and data
RUN mkdir -p /app/models /app/data /app/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models
ENV DEVICE=cpu

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Start the application
CMD ["python", "-m", "deployment.cpu.main"]