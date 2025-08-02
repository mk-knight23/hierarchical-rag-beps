# Hierarchical RAG System for OECD BEPS Pillar Two

A sophisticated Retrieval-Augmented Generation (RAG) system designed specifically for processing and analyzing OECD BEPS Pillar Two documents using hierarchical document processing and intelligent chunking.

## Overview

This system implements a hierarchical approach to document processing that breaks down complex OECD BEPS documents into semantically meaningful chunks while preserving document structure and metadata. The processed documents are then summarized and prepared for advanced retrieval and analysis.

## Features

- **Document Processing Pipeline**: Complete pipeline for processing PDF documents
- **Intelligent Chunking**: Semantic boundary detection with configurable chunk sizes
- **Document Summarization**: Automatic summary generation using Phi3 model
- **Metadata Preservation**: Comprehensive metadata tracking for all chunks
- **Parallel Processing**: Efficient batch processing with progress tracking
- **Error Handling**: Robust error handling and logging
- **Configurable**: YAML-based configuration system

## Architecture

The system follows a modular architecture with the following components:

```
hierarchical-rag-beps/
├── src/
│   ├── models/
│   │   └── data_structures.py    # Core data classes
│   ├── processing/
│   │   ├── document_loader.py    # PDF document loading
│   │   ├── chunker.py           # Text chunking engine
│   │   ├── summary_generator.py # Document summarization
│   │   └── pipeline.py          # Main processing pipeline
│   ├── config/
│   │   ├── __init__.py
│   │   ├── config_loader.py     # Configuration loading
│   │   └── processing_config.yaml # Default configuration
├── examples/
│   ├── basic_usage.py           # Simple usage examples
│   └── batch_processing.py      # Advanced batch processing
├── data/
│   ├── raw/                     # Input documents
│   └── processed/               # Processed output
└── docs/
    └── architecture.md          # System architecture
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd hierarchical-rag-beps
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install system dependencies for PDF processing:
```bash
# On Ubuntu/Debian
sudo apt-get install poppler-utils

# On macOS
brew install poppler

# On Windows
# Download poppler from https://github.com/oschwartz10612/poppler-windows/releases/
```

## Quick Start

### Basic Usage

```python
from src.processing.pipeline import DocumentProcessor
from src.config.config_loader import load_config

# Load configuration
config = load_config()

# Initialize processor
processor = DocumentProcessor(config)

# Process a single document
result = processor.process_single_document("data/raw/sample.pdf")
print(f"Generated {len(result.chunks)} chunks")
print(f"Summary: {result.summary.summary_text[:200]}...")
```

### Batch Processing

```python
from examples.batch_processing import BatchProcessor

# Initialize batch processor
processor = BatchProcessor()

# Process entire directory
report = processor.process_with_report(
    input_dir="data/raw",
    output_dir="data/processed"
)

print(f"Processed {report['processing_summary']['total_documents']} documents")
```

## Configuration

The system uses YAML configuration files for flexible settings. Edit `src/config/processing_config.yaml`:

```yaml
processing:
  chunk_size: 1000
  chunk_overlap: 200
  batch_size: 10
  max_workers: 4
  
paths:
  input_dir: "data/raw"
  output_dir: "data/processed"
  cache_dir: "data/cache"
  
models:
  summary_model: "microsoft/Phi-3-mini-4k-instruct"
  device: "auto"
  
logging:
  level: "INFO"
  file: "logs/processing.log"
```

## Usage Examples

### 1. Basic Document Processing

```python
from src.processing.pipeline import DocumentProcessor
from src.config.config_loader import load_config

config = load_config()
processor = DocumentProcessor(config)

# Process single document
result = processor.process_single_document("path/to/document.pdf")
print(f"Document: {result.document.metadata['filename']}")
print(f"Chunks: {len(result.chunks)}")
print(f"Summary: {result.summary.summary_text}")
```

### 2. Directory Processing

```python
# Process entire directory
results = processor.process_directory("data/raw")

for doc_id, result in results.items():
    print(f"\n{result.document.metadata['filename']}:")
    print(f"  - {len(result.chunks)} chunks")
    print(f"  - Topics: {', '.join(result.summary.key_topics)}")
```

### 3. Custom Configuration

```python
# Load custom config
custom_config = load_config("my_config.yaml")

# Override specific settings
custom_config['processing']['chunk_size'] = 1500
custom_config['processing']['max_workers'] = 8

processor = DocumentProcessor(custom_config)
```

## Data Structures

### Document
Represents a raw document with metadata:
- `content`: Raw text content
- `metadata`: Document metadata (filename, page count, etc.)

### Chunk
Represents a processed text chunk:
- `chunk_id`: Unique identifier
- `text`: Chunk text content
- `metadata`: Chunk metadata (source, page numbers, etc.)

### Summary
Document summary with:
- `summary_text`: Concise document summary
- `topics`: Extracted topics
- `keywords`: Key terms and phrases

### ProcessedDocument
Complete processed document containing:
- `document`: Original document
- `chunks`: List of chunks
- `summary`: Document summary

## Error Handling

The system includes comprehensive error handling:

- **File Access**: Handles missing or corrupted files
- **PDF Processing**: Manages PDF parsing errors
- **Model Errors**: Handles model loading and inference failures
- **Memory Management**: Prevents memory exhaustion with large documents

## Performance Optimization

- **Parallel Processing**: Utilizes multiple CPU cores
- **Batch Processing**: Efficient batch size configuration
- **Caching**: Optional caching for repeated processing
- **Memory Management**: Streaming processing for large documents

## Testing

Run the test suite:

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

## Logging

Logs are written to:
- Console: Configurable log level
- File: `logs/processing.log`
- Structured format for easy parsing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
- Check the documentation in `docs/`
- Review the examples in `examples/`
- Open an issue on GitHub