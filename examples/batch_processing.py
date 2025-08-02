#!/usr/bin/env python3
"""
Batch processing example for the Hierarchical RAG document processing pipeline.

This example demonstrates how to process multiple documents in parallel
from a directory containing OECD BEPS documents.
"""

import asyncio
import logging
from pathlib import Path
import sys
from typing import List
import json

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from processing.pipeline import DocumentProcessor
from config.config_loader import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def process_documents_batch(
    input_dir: Path,
    output_dir: Path,
    max_concurrent: int = 3
) -> List:
    """Process multiple documents in parallel."""
    
    # Load configuration
    config = load_config()
    
    # Initialize the document processor
    processor = DocumentProcessor(config)
    
    # Find all PDF files in the input directory
    pdf_files = list(input_dir.glob("*.pdf"))
    txt_files = list(input_dir.glob("*.txt"))
    all_files = pdf_files + txt_files
    
    if not all_files:
        logger.warning(f"No PDF or TXT files found in {input_dir}")
        
        # Create sample files for demonstration
        logger.info("Creating sample files for demonstration...")
        input_dir.mkdir(parents=True, exist_ok=True)
        
        # Sample OECD BEPS documents
        samples = [
            ("oecd_beps_pillar_two_1.txt", """
            OECD BEPS Pillar Two Model Rules - Income Inclusion Rule
            The Income Inclusion Rule (IIR) is a key component of the Pillar Two GloBE rules.
            It operates by imposing a top-up tax on the parent entity in respect of the low-taxed income
            of its subsidiaries. The IIR applies when the effective tax rate in a jurisdiction is below
            the minimum rate of 15%.
            """),
            ("oecd_beps_pillar_two_2.txt", """
            OECD BEPS Pillar Two Model Rules - Undertaxed Payments Rule
            The Undertaxed Payments Rule (UTPR) serves as a backstop to the Income Inclusion Rule.
            It denies deductions or requires equivalent adjustments for payments that are subject to
            low levels of taxation. The UTPR ensures that the minimum tax is collected even when
            the IIR does not apply.
            """),
            ("oecd_beps_pillar_two_3.txt", """
            OECD BEPS Pillar Two Model Rules - Subject to Tax Rule
            The Subject to Tax Rule (STTR) is a treaty-based rule that allows source countries to
            impose limited source taxation on certain related party payments that are subject to
            low nominal rates of taxation. The STTR complements the GloBE rules by addressing
            specific base erosion risks.
            """)
        ]
        
        for filename, content in samples:
            file_path = input_dir / filename
            file_path.write_text(content.strip())
            all_files.append(file_path)
    
    logger.info(f"Found {len(all_files)} documents to process")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process documents with concurrency limit
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_single_with_semaphore(file_path: Path):
        async with semaphore:
            try:
                logger.info(f"Processing: {file_path.name}")
                result = await processor.process_document(file_path)
                
                # Save individual result
                output_file = output_dir / f"{result.document_id}.json"
                with open(output_file, 'w') as f:
                    json.dump(result.to_dict(), f, indent=2, default=str)
                
                logger.info(f"Completed: {file_path.name} -> {output_file.name}")
                return result
                
            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {str(e)}")
                return None
    
    # Process all documents
    results = await asyncio.gather(*[
        process_single_with_semaphore(file_path)
        for file_path in all_files
    ])
    
    # Filter out failed processes
    successful_results = [r for r in results if r is not None]
    
    # Generate batch summary
    summary = {
        "total_documents": len(all_files),
        "successful": len(successful_results),
        "failed": len(all_files) - len(successful_results),
        "total_chunks": sum(len(r.chunks) for r in successful_results),
        "processing_time": sum(r.metadata.get("processing_time", 0) for r in successful_results),
        "documents": [
            {
                "id": r.document_id,
                "title": r.metadata.get("title", "Unknown"),
                "chunks": len(r.chunks),
                "file": str(r.metadata.get("source_file", "Unknown"))
            }
            for r in successful_results
        ]
    }
    
    # Save batch summary
    summary_file = output_dir / "batch_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("=" * 50)
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Total documents: {summary['total_documents']}")
    logger.info(f"Successful: {summary['successful']}")
    logger.info(f"Failed: {summary['failed']}")
    logger.info(f"Total chunks created: {summary['total_chunks']}")
    logger.info(f"Summary saved to: {summary_file}")
    
    return successful_results


async def main():
    """Main batch processing function."""
    
    # Define input and output directories
    input_dir = Path("data/documents")
    output_dir = Path("output/batch_results")
    
    # Process the batch
    results = await process_documents_batch(
        input_dir=input_dir,
        output_dir=output_dir,
        max_concurrent=3
    )
    
    return results


if __name__ == "__main__":
    asyncio.run(main())