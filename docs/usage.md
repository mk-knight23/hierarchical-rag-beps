# Usage Guide

This guide provides comprehensive instructions for using the Hierarchical RAG Document Processing Pipeline.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/hierarchical-rag-beps.git
cd hierarchical-rag-beps

# Install dependencies
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Basic Usage

```python
from hierarchical_rag_beps import DocumentProcessor
from hierarchical_rag_beps.config import load_config

# Load configuration
config = load_config("config/processing_config.yaml")

# Initialize processor
processor = DocumentProcessor(config)

# Process a single document
result = processor.process_document("path/to/document.pdf")

# Access processed data
print(f"Document ID: {result.document_id}")
print(f"Number of chunks: {len(result.chunks)}")
print(f"Summary: {result.summary}")
```

## Configuration

### Basic Configuration

Create a `config.yaml` file:

```yaml
# Document Processing Configuration
document_processing:
  max_file_size_mb: 100
  supported_formats: ["pdf", "txt", "docx"]
  
# Chunking Configuration
chunking:
  method: "semantic"
  max_chunk_size: 512
  overlap_size: 50
  min_chunk_size: 100
  
# Summary Configuration
summary:
  model_name: "microsoft/Phi-3-mini-4k-instruct"
  max_length: 150
  min_length: 50
  temperature: 0.7
  
# Storage Configuration
storage:
  output_dir: "./output"
  format: "json"
```

### Advanced Configuration

For more complex scenarios:

```yaml
# Advanced configuration with custom parameters
document_processing:
  max_file_size_mb: 500
  supported_formats: ["pdf", "txt", "docx", "md"]
  ocr_enabled: true
  extract_images: false
  
chunking:
  method: "hybrid"
  max_chunk_size: 1024
  overlap_size: 100
  min_chunk_size: 200
  semantic_threshold: 0.7
  use_sentence_boundaries: true
  
summary:
  model_name: "microsoft/Phi-3-mini-4k-instruct"
  max_length: 300
  min_length: 100
  temperature: 0.5
  top_p: 0.9
  do_sample: true
  
storage:
  output_dir: "./output"
  format: "json"
  compress: true
  include_metadata: true
  
logging:
  level: "INFO"
  file: "./logs/processing.log"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

## Processing Documents

### Single Document Processing

```python
from hierarchical_rag_beps import DocumentProcessor
from hierarchical_rag_beps.config import load_config

# Initialize processor
config = load_config("config/processing_config.yaml")
processor = DocumentProcessor(config)

# Process document
result = processor.process_document(
    "data/oecd_beps_pillar_two.pdf",
    output_dir="./output"
)

# Save results
result.save_to_file("./output/processed_document.json")
```

### Batch Processing

```python
from hierarchical_rag_beps import BatchProcessor
import glob

# Initialize batch processor
batch_processor = BatchProcessor(config)

# Process multiple documents
documents = glob.glob("data/*.pdf")
results = batch_processor.process_batch(
    documents,
    output_dir="./output/batch",
    num_workers=4
)

# Generate summary report
batch_processor.generate_report(results, "./output/batch_report.json")
```

### Streaming Processing

For large documents or real-time processing:

```python
from hierarchical_rag_beps import StreamingProcessor

# Initialize streaming processor
stream_processor = StreamingProcessor(config)

# Process document in chunks
async for chunk_result in stream_processor.process_stream("large_document.pdf"):
    # Handle each chunk as it's processed
    print(f"Processed chunk {chunk_result.chunk_id}")
    
    # Save intermediate results
    chunk_result.save_to_file(f"./output/chunk_{chunk_result.chunk_id}.json")
```

## Working with Results

### Accessing Processed Data

```python
# Load processed document
from hierarchical_rag_beps.models import ProcessedDocument

processed_doc = ProcessedDocument.load_from_file("output/processed_document.json")

# Access document information
print(f"Title: {processed_doc.title}")
print(f"Total chunks: {len(processed_doc.chunks)}")
print(f"Document summary: {processed_doc.summary}")

# Iterate through chunks
for chunk in processed_doc.chunks:
    print(f"Chunk {chunk.chunk_id}: {chunk.text[:100]}...")
    print(f"Summary: {chunk.summary}")
    print(f"Metadata: {chunk.metadata}")
```

### Searching Processed Documents

```python
from hierarchical_rag_beps import DocumentSearcher

# Initialize searcher
searcher = DocumentSearcher()

# Index processed documents
searcher.index_directory("./output")

# Search for content
results = searcher.search("minimum tax rate", top_k=5)

for result in results:
    print(f"Score: {result.score}")
    print(f"Document: {result.document_id}")
    print(f"Chunk: {result.chunk_id}")
    print(f"Text: {result.text}")
```

## Docker Usage

### Using Docker Compose

```bash
# Start the application
docker-compose up hierarchical-rag

# Run in development mode
docker-compose up hierarchical-rag-dev

# Run tests
docker-compose run test-runner

# Start Jupyter for experimentation
docker-compose up jupyter
```

### Using Docker directly

```bash
# Build image
docker build -t hierarchical-rag-beps .

# Run container
docker run -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output hierarchical-rag-beps

# Run with custom config
docker run -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output -v $(pwd)/config:/app/config hierarchical-rag-beps python examples/basic_usage.py
```

## API Usage

### REST API (Future Feature)

```python
import requests

# Start processing
response = requests.post(
    "http://localhost:8000/process",
    json={"document_path": "data/document.pdf"}
)
job_id = response.json()["job_id"]

# Check status
status = requests.get(f"http://localhost:8000/status/{job_id}")
print(status.json())

# Get results
results = requests.get(f"http://localhost:8000/results/{job_id}")
```

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce `max_chunk_size` in configuration
   - Process documents in smaller batches
   - Use streaming processing for large files

2. **Model Loading Issues**
   - Ensure sufficient disk space for model cache
   - Check internet connection for initial model download
   - Use local model cache by setting `TRANSFORMERS_CACHE`

3. **PDF Processing Issues**
   - Install system dependencies: `apt-get install poppler-utils tesseract-ocr`
   - Check PDF permissions and encryption
   - Use OCR for scanned documents

### Performance Optimization

1. **Parallel Processing**
   ```python
   batch_processor = BatchProcessor(config)
   results = batch_processor.process_batch(
       documents,
       num_workers=8,  # Increase workers
       batch_size=10   # Process 10 docs at once
   )
   ```

2. **GPU Acceleration**
   ```python
   config.summary.device = "cuda"  # Use GPU for model inference
   config.chunking.use_gpu = True  # Use GPU for embeddings
   ```

3. **Caching**
   ```python
   # Enable caching for repeated processing
   processor = DocumentProcessor(config, cache_dir="./cache")
   ```

## Best Practices

1. **Organize Input Data**
   ```
   data/
   ├── raw/
   │   ├── oecd_docs/
   │   └── regulatory_docs/
   └── processed/
   ```

2. **Use Appropriate Chunk Sizes**
   - Technical documents: 512-1024 tokens
   - Legal documents: 1024-2048 tokens
   - Summary documents: 256-512 tokens

3. **Monitor Processing**
   ```python
   import logging
   logging.basicConfig(level=logging.INFO)
   
   # Monitor memory usage
   import psutil
   process = psutil.Process()
   print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
   ```

## Examples

See the `examples/` directory for complete working examples:
- `basic_usage.py` - Simple document processing
- `batch_processing.py` - Processing multiple documents
- `custom_config.py` - Using custom configurations
- `streaming_example.py` - Real-time processing