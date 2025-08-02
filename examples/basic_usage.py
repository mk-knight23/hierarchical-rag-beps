#!/usr/bin/env python3
"""
Basic usage example for the Hierarchical RAG document processing pipeline.

This example demonstrates how to use the document processing pipeline
to process a single PDF document from the OECD BEPS dataset.
"""

import asyncio
import logging
from pathlib import Path
import sys

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from processing.pipeline import DocumentProcessingPipeline
from config.config_loader import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def process_single_document():
    """Process a single PDF document."""
    
    # Load configuration
    config = load_config()
    
    # Initialize the document processor
    processor = DocumentProcessingPipeline(config)
    
    # Example: Process a single PDF file
    # Replace this with your actual PDF file path
    pdf_path = Path("data/raw/administrative-guidance-global-anti-base-erosion-rules-pillar-two-july-2023.pdf")
    
    if not pdf_path.exists():
        logger.warning(f"Sample PDF not found at {pdf_path}")
        logger.info("Creating a sample text file for demonstration...")
        
        # Create a sample text file for demonstration
        sample_text = """
        OECD BEPS Pillar Two Model Rules
        
        The OECD/G20 Inclusive Framework on BEPS has agreed on a two-pillar solution to address 
        the tax challenges arising from the digitalisation of the economy. Pillar Two consists 
        of the Global Anti-Base Erosion (GloBE) Rules, which ensure that large multinational 
        enterprises (MNEs) pay a minimum level of tax on the income arising in each jurisdiction 
        where they operate.
        
        Key Components:
        1. Income Inclusion Rule (IIR)
        2. Undertaxed Payments Rule (UTPR)
        3. Subject to Tax Rule (STTR)
        
        The GloBE Rules apply to MNEs with annual revenue of at least EUR 750 million.
        """
        
        sample_path = Path("data/sample_oecd_document.txt")
        sample_path.parent.mkdir(exist_ok=True)
        sample_path.write_text(sample_text)
        pdf_path = sample_path
    
    logger.info(f"Processing document: {pdf_path}")
    
    try:
        # Process the document
        # Since process_documents expects a directory, we'll create a temporary directory
        # with our single document
        import shutil
        
        # Create a temporary directory for the single document
        temp_dir = Path("temp_single_doc")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(exist_ok=True)
        
        # Copy the document to the temp directory
        if pdf_path.suffix.lower() == '.pdf':
            dest_path = temp_dir / pdf_path.name
            shutil.copy2(pdf_path, dest_path)
        else:
            dest_path = temp_dir / "sample_document.txt"
            shutil.copy2(pdf_path, dest_path)
        
        # Process the document
        processed_docs = await processor.process_documents(str(temp_dir))
        
        if not processed_docs:
            logger.error("No documents were processed")
            return None
            
        processed_doc = processed_docs[0]
        
        # Clean up temp directory
        shutil.rmtree(temp_dir)
        
        # Display results
        logger.info("=" * 50)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 50)
        
        logger.info(f"Document ID: {processed_doc.document.id}")
        logger.info(f"Title: {processed_doc.document.metadata.get('title', 'Untitled')}")
        logger.info(f"Total chunks: {len(processed_doc.chunks)}")
        logger.info(f"Summary length: {len(processed_doc.summary.summary)} characters")
        
        # Display summary
        logger.info("\nDocument Summary:")
        logger.info("-" * 30)
        logger.info(processed_doc.summary.summary)
        
        # Display first few chunks
        logger.info("\nFirst 3 chunks:")
        for i, chunk in enumerate(processed_doc.chunks[:3]):
            logger.info(f"\nChunk {i+1} (ID: {chunk.id})")
            logger.info(f"Tokens: {chunk.metadata.get('token_count', 'N/A')}")
            logger.info(f"Text preview: {chunk.text[:100]}...")
        
        # Save results
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # Save processed document
        import json
        output_file = output_dir / f"{processed_doc.document.id}_processed.json"
        with open(output_file, 'w') as f:
            json.dump(processed_doc.model_dump(), f, indent=2, default=str)
        
        logger.info(f"\nResults saved to: {output_file}")
        
        return processed_doc
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(process_single_document())