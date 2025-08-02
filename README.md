# BEPS Hierarchical RAG Agent System

A sophisticated AI agent system for processing and analyzing Base Erosion and Profit Shifting (BEPS) documents using hierarchical retrieval-augmented generation (RAG) with intelligent routing and web search capabilities.

## 🎯 Overview

This system provides an intelligent agent that can:
- Process BEPS documents (PDF reports, policy papers, regulatory guidance)
- Answer complex queries about BEPS regulations and compliance
- Provide factual, analytical, procedural, and comparative insights
- Integrate RAG retrieval with real-time web search
- Handle batch processing and caching for performance

## 🏗️ Architecture

The system consists of several key components:

### Core Components
- **BEPSAgent**: Main orchestrator class
- **QueryClassifier**: Categorizes queries into types (factual, analytical, procedural, temporal, comparative)
- **ResponseRouter**: Determines optimal response strategy
- **RAGHandler**: Retrieves relevant information from processed documents
- **WebSearchHandler**: Provides real-time information when RAG is insufficient
- **ConfidenceScorer**: Evaluates response quality and reliability

### Response Strategies
- **RAG_RETRIEVAL**: Use processed BEPS documents
- **WEB_SEARCH**: Fetch current information
- **HYBRID**: Combine RAG and web search
- **DIRECT_ANSWER**: Use when query is simple

## 📁 Project Structure

```
hierarchical-rag-beps/
├── data/
│   ├── raw/                    # Original PDF files
│   ├── processed/              # Processed text files
│   └── vector_store/           # FAISS vector database
├── src/
│   └── agent/
│       ├── __init__.py
│       ├── beps_agent.py       # Main agent class
│       ├── query_classifier.py # Query type classification
│       ├── response_router.py  # Strategy selection
│       ├── rag_handler.py      # RAG implementation
│       ├── web_search_handler.py # Web search integration
│       └── confidence_scorer.py # Response scoring
├── tests/
│   ├── test_agent.py          # Comprehensive test suite
│   ├── run_tests.py          # Test runner
│   └── integration_demo.py   # Integration demonstration
├── examples/
│   ├── batch_process_pdfs.py  # PDF processing script
│   ├── batch_processing.py    # Batch query processing
│   └── check_pdf_files.py     # PDF validation
├── config/
│   └── processing_config.yaml # Configuration settings
└── README.md
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Process PDF Documents
```bash
python examples/batch_process_pdfs.py
```

### 3. Run Integration Demo
```bash
python tests/integration_demo.py
```

### 4. Run Tests
```bash
python tests/run_tests.py
```

## 💻 Usage Examples

### Basic Query Processing
```python
import asyncio
from src.agent.beps_agent import BEPSAgent

async def main():
    agent = BEPSAgent()
    
    # Single query
    response = await agent.process_query(
        "What are the key recommendations in BEPS Action 1?"
    )
    print(response['answer'])
    
    # Batch processing
    queries = [
        "Explain BEPS Action 5",
        "What is transfer pricing documentation?",
        "How does BEPS affect developing countries?"
    ]
    results = await agent.process_batch_queries(queries)
    
asyncio.run(main())
```

### Advanced Usage
```python
# With custom configuration
agent = BEPSAgent(
    cache_ttl=3600,  # 1 hour cache
    max_web_results=5,
    confidence_threshold=0.7
)

# Health check
health = await agent.health_check()
print(f"System status: {health['status']}")

# Statistics
stats = agent.get_statistics()
print(f"Total queries processed: {stats['total_queries']}")
```

## 🧪 Testing

The system includes comprehensive testing:

### Unit Tests
- Component-level testing for all agent modules
- Mock external dependencies for reliable testing
- Test edge cases and error handling

### Integration Tests
- End-to-end system testing
- Real document processing
- Performance benchmarking

### Run All Tests
```bash
# Run all tests
python tests/run_tests.py

# Run specific test categories
python tests/run_tests.py --unit-only
python tests/run_tests.py --integration-only
python tests/run_tests.py --performance
```

## 📊 Performance Features

- **Caching**: Intelligent caching for repeated queries
- **Batch Processing**: Efficient handling of multiple queries
- **Timeout Handling**: Graceful degradation for slow operations
- **Statistics Collection**: Performance monitoring and analytics
- **Health Checks**: System status monitoring

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Set custom configuration
export BEPS_CACHE_TTL=3600
export BEPS_MAX_WEB_RESULTS=10
export BEPS_CONFIDENCE_THRESHOLD=0.8
```

### Configuration File
```yaml
# config/processing_config.yaml
processing:
  chunk_size: 1000
  chunk_overlap: 200
  max_workers: 4

agent:
  cache_ttl: 3600
  max_web_results: 5
  confidence_threshold: 0.7
  timeout: 30
```

## 🎯 Query Types

The system handles five types of queries:

1. **Factual**: "What is BEPS Action 1 about?"
2. **Analytical**: "How do BEPS recommendations affect developing countries?"
3. **Procedural**: "What steps should companies take for BEPS compliance?"
4. **Temporal**: "What were the main BEPS developments in 2023?"
5. **Comparative**: "Compare BEPS Action 1 and Action 5 approaches"

## 📈 Performance Metrics

The system tracks:
- Query processing time
- Confidence scores
- Cache hit rates
- Strategy effectiveness
- Error rates

## 🔍 Troubleshooting

### Common Issues

1. **PDF Processing Errors**
   - Check file permissions
   - Verify PDF format compatibility
   - Review processing logs

2. **Low Confidence Scores**
   - Increase document coverage
   - Adjust confidence threshold
   - Check document quality

3. **Slow Response Times**
   - Enable caching
   - Optimize chunk sizes
   - Check network connectivity

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

agent = BEPSAgent(debug=True)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run the test suite
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OECD for BEPS documentation
- OpenAI for language model capabilities
- FAISS for vector similarity search