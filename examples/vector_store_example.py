#!/usr/bin/env python3
"""Example usage of the vector store system."""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vector_store import VectorStore
from src.config import ProcessingConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Main example function."""
    try:
        # Initialize configuration
        config = ProcessingConfig()
        
        # Initialize vector store
        vector_store = VectorStore(config)
        await vector_store.initialize()
        
        # Sample documents
        documents = [
            {
                'content': 'BEPS Action 1 addresses the tax challenges of the digital economy. The digital economy has created new business models that challenge traditional tax rules.',
                'metadata': {
                    'source': 'BEPS Action 1 Report',
                    'action': 1,
                    'topic': 'digital_economy'
                }
            },
            {
                'content': 'BEPS Action 5 focuses on harmful tax practices and substance requirements. It aims to prevent countries from offering preferential tax regimes without real economic activity.',
                'metadata': {
                    'source': 'BEPS Action 5 Report',
                    'action': 5,
                    'topic': 'harmful_tax_practices'
                }
            },
            {
                'content': 'BEPS Action 13 introduces country-by-country reporting requirements for large multinational enterprises. This increases transparency in tax reporting.',
                'metadata': {
                    'source': 'BEPS Action 13 Report',
                    'action': 13,
                    'topic': 'country_by_country_reporting'
                }
            }
        ]
        
        # Add documents to vector store
        logger.info("Adding documents to vector store...")
        doc_ids = await vector_store.add_documents(documents)
        logger.info(f"Added {len(doc_ids)} documents")
        
        # Query the vector store
        logger.info("Querying vector store...")
        results = await vector_store.query("What are the tax challenges in the digital economy?", top_k=2)
        
        print("\nQuery Results:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Score: {result['score']:.3f}")
            print(f"   Content: {result['content'][:100]}...")
            print(f"   Metadata: {result['metadata']}")
        
        # Get vector store stats
        stats = vector_store.get_stats()
        print(f"\nVector Store Stats: {stats}")
        
        # Save vector store to disk
        save_path = "data/vector_store"
        success = await vector_store.save_to_disk(save_path)
        if success:
            logger.info(f"Vector store saved to {save_path}")
        
        # Test loading from disk
        new_vector_store = VectorStore(config)
        await new_vector_store.initialize()
        loaded = await new_vector_store.load_from_disk(save_path)
        if loaded:
            logger.info("Vector store loaded successfully from disk")
            new_stats = new_vector_store.get_stats()
            print(f"Loaded vector store stats: {new_stats}")
        
    except Exception as e:
        logger.error(f"Error in example: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())