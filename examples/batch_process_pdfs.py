#!/usr/bin/env python3
"""
Batch processing script for the 7 OECD BEPS PDF documents in data/raw.

This script processes all PDF files found in the data/raw directory using
the hierarchical RAG document processing pipeline.
"""

import asyncio
import logging
from pathlib import Path
import sys
from typing import List, Dict
import json
import time
from datetime import datetime

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


class PDFBatchProcessor:
    """Enhanced batch processor for OECD BEPS PDF documents"""
    
    def __init__(self, config_path: str = None):
        self.config = load_config(config_path)
        self.pipeline = DocumentProcessingPipeline(self.config)
        
    async def process_pdf_batch(self, 
                              input_dir: Path = None,
                              output_dir: Path = None,
                              max_concurrent: int = 3) -> List[Dict]:
        """Process all PDF files in the specified directory"""
        
        # Set default directories
        input_dir = input_dir or Path("data/raw")
        output_dir = output_dir or Path("output/pdf_processing_results")
        
        # Find all PDF files
        pdf_files = list(input_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.error(f"No PDF files found in {input_dir}")
            return []
        
        logger.info(f"Found {len(pdf_files)} PDF files to process:")
        for pdf_file in pdf_files:
            logger.info(f"  - {pdf_file.name} ({pdf_file.stat().st_size / 1024 / 1024:.1f} MB)")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Record start time
        start_time = time.time()
        
        # Process all documents
        logger.info("Starting batch PDF processing...")
        results = await self.pipeline.process_documents(str(input_dir))
        
        # Calculate processing statistics
        processing_time = time.time() - start_time
        
        # Filter successful results
        successful_results = [r for r in results if r.document.error is None]
        failed_results = [r for r in results if r.document.error is not None]
        
        # Generate comprehensive summary
        summary = self._generate_processing_summary(
            results, successful_results, failed_results, processing_time
        )
        
        # Save detailed results
        await self._save_detailed_results(successful_results, output_dir, summary)
        
        # Print final report
        self._print_final_report(summary, output_dir)
        
        return successful_results
    
    def _generate_processing_summary(self, all_results, successful, failed, processing_time):
        """Generate comprehensive processing summary"""
        
        total_chunks = sum(len(r.chunks) for r in successful)
        total_tokens = sum(sum(c.token_count for c in r.chunks) for r in successful)
        
        summary = {
            "processing_info": {
                "processed_at": datetime.utcnow().isoformat(),
                "processing_time_seconds": round(processing_time, 2),
                "input_directory": "data/raw",
                "output_directory": str(Path("output/pdf_processing_results").absolute()),
                "config_used": self.config
            },
            "statistics": {
                "total_documents": len(all_results),
                "successful": len(successful),
                "failed": len(failed),
                "success_rate": round(len(successful) / len(all_results) * 100, 1) if all_results else 0,
                "total_chunks_created": total_chunks,
                "total_tokens_processed": total_tokens,
                "avg_chunks_per_doc": round(total_chunks / len(successful), 1) if successful else 0,
                "avg_processing_time_per_doc": round(processing_time / len(successful), 2) if successful else 0
            },
            "documents": [],
            "failed_documents": []
        }
        
        # Add successful documents
        for result in successful:
            doc_info = {
                "id": result.document.id,
                "title": result.document.metadata.get("title", "Unknown"),
                "file_name": Path(result.document.file_path).name,
                "file_size_bytes": Path(result.document.file_path).stat().st_size,
                "chunks": len(result.chunks),
                "tokens": sum(c.token_count for c in result.chunks),
                "topics": result.summary.key_topics,
                "keywords": result.summary.keywords[:10],  # Top 10 keywords
                "processing_metadata": result.processing_metadata
            }
            summary["documents"].append(doc_info)
        
        # Add failed documents
        for result in failed:
            fail_info = {
                "id": result.document.id,
                "file_name": Path(result.document.file_path).name,
                "error": result.document.error
            }
            summary["failed_documents"].append(fail_info)
        
        return summary
    
    async def _save_detailed_results(self, successful_results, output_dir, summary):
        """Save detailed processing results"""
        
        # Save individual document results
        for result in successful_results:
            doc_dir = output_dir / result.document.id
            doc_dir.mkdir(exist_ok=True)
            
            # Save document summary
            doc_summary = {
                "document": {
                    "id": result.document.id,
                    "file_path": result.document.file_path,
                    "title": result.document.metadata.get("title", "Unknown"),
                    "author": result.document.metadata.get("author", ""),
                    "subject": result.document.metadata.get("subject", ""),
                    "metadata": result.document.metadata
                },
                "summary": {
                    "summary": result.summary.summary,
                    "key_topics": result.summary.key_topics,
                    "keywords": result.summary.keywords,
                    "summary_length": len(result.summary.summary)
                },
                "chunks": [
                    {
                        "id": chunk.id,
                        "start_page": chunk.start_page,
                        "end_page": chunk.end_page,
                        "token_count": chunk.token_count,
                        "section_header": chunk.section_header,
                        "content_preview": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
                    }
                    for chunk in result.chunks
                ],
                "processing_metadata": result.processing_metadata
            }
            
            with open(doc_dir / "document_summary.json", "w") as f:
                json.dump(doc_summary, f, indent=2)
        
        # Save batch summary
        with open(output_dir / "batch_processing_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Save chunk index for all documents
        chunk_index = {}
        for result in successful_results:
            for i, chunk in enumerate(result.chunks):
                chunk_index[chunk.id] = {
                    "document_id": result.document.id,
                    "document_title": result.document.metadata.get("title", "Unknown"),
                    "chunk_index": i,
                    "file_name": Path(result.document.file_path).name,
                    "page_numbers": [chunk.start_page, chunk.end_page],
                    "section_header": chunk.section_header,
                    "token_count": chunk.token_count
                }
        
        with open(output_dir / "chunk_index.json", "w") as f:
            json.dump(chunk_index, f, indent=2)
        
        logger.info(f"Results saved to: {output_dir}")
    
    def _print_final_report(self, summary, output_dir):
        """Print final processing report"""
        
        print("\n" + "="*80)
        print("OECD BEPS PDF BATCH PROCESSING COMPLETE")
        print("="*80)
        
        stats = summary["statistics"]
        info = summary["processing_info"]
        
        print(f"Processing Time: {info['processing_time_seconds']} seconds")
        print(f"Total Documents: {stats['total_documents']}")
        print(f"Successful: {stats['successful']} ({stats['success_rate']}%)")
        print(f"Failed: {stats['failed']}")
        print(f"Total Chunks: {stats['total_chunks_created']}")
        print(f"Total Tokens: {stats['total_tokens_processed']}")
        print(f"Average Chunks per Document: {stats['avg_chunks_per_doc']}")
        
        if summary["documents"]:
            print("\nSuccessfully Processed Documents:")
            for doc in summary["documents"]:
                print(f"  ✓ {doc['file_name']} - {doc['chunks']} chunks, {doc['tokens']} tokens")
        
        if summary["failed_documents"]:
            print("\nFailed Documents:")
            for doc in summary["failed_documents"]:
                print(f"  ✗ {doc['file_name']} - {doc['error']}")
        
        print(f"\nResults saved to: {output_dir}")
        print("="*80)


async def main():
    """Main execution function"""
    
    # Initialize processor
    processor = PDFBatchProcessor()
    
    # Process the PDF batch
    results = await processor.process_pdf_batch(
        input_dir=Path("data/raw"),
        output_dir=Path("output/pdf_processing_results"),
        max_concurrent=2  # Conservative for PDF processing
    )
    
    return results


if __name__ == "__main__":
    asyncio.run(main())