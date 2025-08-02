# OECD BEPS PDF Batch Processing Examples

This directory contains scripts for processing the 7 OECD BEPS Pillar Two PDF documents using the hierarchical RAG document processing pipeline.

## Available Scripts

### 1. `batch_process_pdfs.py` - Main Batch Processing Script
**Purpose**: Process all 7 PDF documents in the data/raw directory with comprehensive logging and reporting.

**Usage**:
```bash
# Basic usage - process all PDFs
python examples/batch_process_pdfs.py

# With custom config
python examples/batch_process_pdfs.py --config config/custom_config.yaml
```

**Features**:
- ✅ Processes all 7 PDF files in data/raw
- ✅ Creates detailed processing reports
- ✅ Generates individual document summaries
- ✅ Creates searchable chunk index
- ✅ Provides comprehensive statistics
- ✅ Handles errors gracefully

**Output Structure**:
```
output/pdf_processing_results/
├── batch_processing_summary.json     # Overall processing statistics
├── chunk_index.json                  # Searchable index of all chunks
├── administrative-guidance-global-anti-base-erosion-rules-pillar-two-july-2023/
│   └── document_summary.json         # Individual document summary
├── agreed-administrative-guidance-for-the-pillar-two-globe-rules/
│   └── document_summary.json
├── beps-2-infographic-singapore-mof/
│   └── document_summary.json
├── dme_deloitte-global-minimum-tax-faq/
│   └── document_summary.json
├── outcome-statement-on-the-two-pillar-solution/
│   └── document_summary.json
├── tax-challenges-pillar-two-commentary/
│   └── document_summary.json
└── tax-challenges-pillar-two-examples/
    └── document_summary.json
```

### 2. `check_pdf_files.py` - File Verification Script
**Purpose**: Verify what PDF and text files are available in the data/raw directory.

**Usage**:
```bash
python examples/check_pdf_files.py
```

**Current PDF Files**:
1. **administrative-guidance-global-anti-base-erosion-rules-pillar-two-july-2023.pdf** (1.1 MB)
2. **agreed-administrative-guidance-for-the-pillar-two-globe-rules.pdf** (1.3 MB)
3. **beps-2-infographic (Singapore MOF).pdf** (0.7 MB)
4. **dme_deloitte-global-minimum-tax-faq.pdf** (1.2 MB)
5. **outcome-statement-on-the-two-pillar-solution-to-address-the-tax-challenges-arising-from-the-digitalisation-of-the-economy-july-2023.pdf** (0.2 MB)
6. **tax-challenges-arising-from-the-digitalisation-of-the-economy-global-anti-base-erosion-model-rules-pillar-two-commentary.pdf** (3.2 MB)
7. **tax-challenges-arising-from-the-digitalisation-of-the-economy-global-anti-base-erosion-model-rules-pillar-two-examples.pdf** (2.0 MB)

### 3. `batch_processing.py` - Original Batch Script
**Purpose**: Original batch processing script (kept for reference).

**Usage**:
```bash
python examples/batch_processing.py
```

## Quick Start Guide

### Step 1: Verify PDF Files
```bash
python examples/check_pdf_files.py
```

### Step 2: Run Batch Processing
```bash
python examples/batch_process_pdfs.py
```

### Step 3: Review Results
Check the `output/pdf_processing_results/` directory for:
- Processing summary and statistics
- Individual document summaries
- Searchable chunk index
- Error reports (if any)

## Configuration

The processing pipeline uses the configuration in `config/processing_config.yaml`:

- **Chunk Size**: 1000 tokens
- **Chunk Overlap**: 200 tokens
- **Max Workers**: 2 (conservative for PDF processing)
- **Batch Size**: 3 documents
- **Model**: microsoft/Phi-3-mini-4k-instruct

## Processing Statistics

Expected processing metrics for the 7 PDF documents:
- **Total Processing Time**: ~2-5 minutes (depending on system)
- **Total Chunks**: ~200-400 chunks
- **Total Tokens**: ~50,000-100,000 tokens
- **Success Rate**: >95% (with error handling)

## Error Handling

The batch processor includes comprehensive error handling:
- **File Size Limits**: 100MB max per file
- **Timeout Protection**: 30 seconds per document
- **Encoding Issues**: Automatic fallback to latin-1
- **PDF Parsing**: Graceful handling of corrupted PDFs
- **Memory Management**: Batch processing to prevent OOM

## Troubleshooting

### Common Issues

1. **"No PDF files found"**
   - Ensure PDF files are in `data/raw/` directory
   - Check file extensions are `.pdf`

2. **"PyPDF2 not found"**
   ```bash
   pip install PyPDF2
   ```

3. **Memory issues with large PDFs**
   - Reduce `batch_size` in config
   - Increase `max_file_size_mb` if needed

4. **Processing timeouts**
   - Increase `timeout_seconds` in config
   - Check system resources

### Performance Optimization

For faster processing:
1. Increase `max_workers` in config (if system has more cores)
2. Increase `batch_size` for better throughput
3. Use SSD storage for better I/O performance

## Output Analysis

After processing, you can analyze the results:

```python
import json
from pathlib import Path

# Load processing summary
with open("output/pdf_processing_results/batch_processing_summary.json") as f:
    summary = json.load(f)

# Check processing statistics
print(f"Processed {summary['statistics']['successful']} documents successfully")
print(f"Total chunks: {summary['statistics']['total_chunks_created']}")

# Search for specific topics
with open("output/pdf_processing_results/chunk_index.json") as f:
    chunks = json.load(f)

# Find chunks about "minimum tax"
min_tax_chunks = [c for c in chunks.values() if "minimum tax" in str(c).lower()]
print(f"Found {len(min_tax_chunks)} chunks about minimum tax")
```

## Integration with RAG System

The processed chunks can be directly used with the hierarchical RAG system:

```python
from src.retrieval.hierarchical_retriever import HierarchicalRetriever
from src.storage.vector_store import VectorStore

# Initialize components
vector_store = VectorStore()
retriever = HierarchicalRetriever(vector_store)

# Load processed chunks
with open("output/pdf_processing_results/chunk_index.json") as f:
    chunks = json.load(f)

# Add chunks to vector store
for chunk_id, chunk_info in chunks.items():
    retriever.add_chunk(chunk_id, chunk_info)