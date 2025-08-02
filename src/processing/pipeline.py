import asyncio
import logging
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
from datetime import datetime
import hashlib

from processing.document_loader import DocumentLoader
from processing.chunker import TextChunker
from processing.summary_generator import SummaryGenerator
from models.data_structures import Document, Chunk, Summary, ProcessedDocument

logger = logging.getLogger(__name__)

class DocumentProcessingPipeline:
    """Main pipeline for processing OECD BEPS documents"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.document_loader = DocumentLoader(config)
        self.chunker = TextChunker(config)
        self.summary_generator = SummaryGenerator(config)
        
        # Setup directories
        self.input_dir = Path(config.get('paths', {}).get('input_dir', 'data/raw'))
        self.output_dir = Path(config.get('paths', {}).get('output_dir', 'data/processed'))
        self.temp_dir = Path(config.get('paths', {}).get('temp_dir', 'data/temp'))
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Processing settings
        self.batch_size = config.get('processing', {}).get('batch_size', 5)
        self.max_workers = config.get('processing', {}).get('max_workers', 4)
        
    async def process_documents(self, input_path: Optional[str] = None) -> List[ProcessedDocument]:
        """Process all documents in the input directory"""
        input_path = input_path or self.input_dir
        
        logger.info(f"Starting document processing from {input_path}")
        
        # Load documents
        documents = self.document_loader.load_documents(str(input_path))
        logger.info(f"Loaded {len(documents)} documents")
        
        if not documents:
            logger.warning("No documents found to process")
            return []
        
        # Process documents in batches
        processed_documents = []
        total_docs = len(documents)
        
        for i in range(0, total_docs, self.batch_size):
            batch = documents[i:i+self.batch_size]
            logger.info(f"Processing batch {i//self.batch_size + 1}/{(total_docs-1)//self.batch_size + 1}")
            
            batch_results = await self._process_batch(batch)
            processed_documents.extend(batch_results)
            
            # Save intermediate results
            self._save_batch_results(batch_results, i//self.batch_size)
        
        # Save final results
        self._save_final_results(processed_documents)
        
        logger.info(f"Completed processing {len(processed_documents)} documents")
        return processed_documents
    
    async def _process_batch(self, documents: List[Document]) -> List[ProcessedDocument]:
        """Process a batch of documents"""
        tasks = [self._process_single_document(doc) for doc in documents]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out failed results
        processed = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Document processing failed: {str(result)}")
            else:
                processed.append(result)
        
        return processed
    
    async def _process_single_document(self, document: Document) -> ProcessedDocument:
        """Process a single document through the pipeline"""
        try:
            logger.info(f"Processing document: {document.id}")
            
            # Generate document summary
            summary = await self.summary_generator.generate_document_summary(document)
            
            # Create chunks
            chunks = self.chunker.chunk_document(document)
            
            # Validate chunks
            valid_chunks = self.chunker.validate_chunks(chunks)
            
            # Generate chunk summaries (optional, for future use)
            # chunk_summaries = await self.summary_generator.generate_chunk_summaries(valid_chunks)
            
            # Create processed document
            processed_doc = ProcessedDocument(
                document=document,
                chunks=valid_chunks,
                summary=summary,
                processing_metadata={
                    'processed_at': datetime.utcnow().isoformat(),
                    'num_chunks': len(valid_chunks),
                    'total_chunks': len(chunks),
                    'validation_passed': len(valid_chunks) == len(chunks),
                    'processing_time': None  # Will be filled by timing decorator
                }
            )
            
            # Save individual document results
            self._save_document_results(processed_doc)
            
            return processed_doc
            
        except Exception as e:
            logger.error(f"Failed to process document {document.id}: {str(e)}")
            raise
    
    def _save_document_results(self, processed_doc: ProcessedDocument):
        """Save results for a single document"""
        doc_dir = self.output_dir / processed_doc.document.id
        doc_dir.mkdir(exist_ok=True)
        
        # Save document info
        doc_info = {
            'document': {
                'id': processed_doc.document.id,
                'file_path': processed_doc.document.file_path,
                'title': processed_doc.document.metadata.get('title', 'Untitled'),
                'metadata': processed_doc.document.metadata
            },
            'summary': {
                'summary': processed_doc.summary.summary,
                'key_topics': processed_doc.summary.key_topics,
                'keywords': processed_doc.summary.keywords
            },
            'processing_metadata': processed_doc.processing_metadata
        }
        
        with open(doc_dir / 'document_info.json', 'w') as f:
            json.dump(doc_info, f, indent=2)
        
        # Save chunks
        chunks_data = []
        for chunk in processed_doc.chunks:
            chunk_data = {
                'id': chunk.id,
                'content': chunk.text,
                'start_page': chunk.start_page,
                'end_page': chunk.end_page,
                'token_count': chunk.token_count,
                'metadata': chunk.metadata
            }
            chunks_data.append(chunk_data)
        
        with open(doc_dir / 'chunks.json', 'w') as f:
            json.dump(chunks_data, f, indent=2)
        
        # Save raw content
        with open(doc_dir / 'raw_content.txt', 'w', encoding='utf-8') as f:
            f.write(processed_doc.document.content)
    
    def _save_batch_results(self, batch: List[ProcessedDocument], batch_index: int):
        """Save intermediate batch results"""
        batch_file = self.temp_dir / f'batch_{batch_index:03d}.json'
        
        batch_data = []
        for doc in batch:
            batch_data.append({
                'document_id': doc.document.id,
                'num_chunks': len(doc.chunks),
                'summary_length': len(doc.summary.summary),
                'topics': doc.summary.key_topics,
                'keywords': doc.summary.keywords
            })
        
        with open(batch_file, 'w') as f:
            json.dump(batch_data, f, indent=2)
    
    def _save_final_results(self, processed_documents: List[ProcessedDocument]):
        """Save final processing results"""
        # Create index file
        index_data = {
            'processing_info': {
                'processed_at': datetime.utcnow().isoformat(),
                'total_documents': len(processed_documents),
                'total_chunks': sum(len(doc.chunks) for doc in processed_documents),
                'config': self.config
            },
            'documents': []
        }
        
        for doc in processed_documents:
            doc_info = {
                'id': doc.document.id,
                'title': doc.document.metadata.get('title', 'Untitled'),
                'file_path': doc.document.file_path,
                'num_chunks': len(doc.chunks),
                'topics': doc.summary.key_topics,
                'keywords': doc.summary.keywords,
                'processing_metadata': doc.processing_metadata
            }
            index_data['documents'].append(doc_info)
        
        with open(self.output_dir / 'index.json', 'w') as f:
            json.dump(index_data, f, indent=2)
        
        # Create chunk index for easy lookup
        chunk_index = {}
        for doc in processed_documents:
            for i, chunk in enumerate(doc.chunks):
                chunk_index[chunk.id] = {
                    'document_id': doc.document.id,
                    'document_title': doc.document.metadata.get('title', 'Untitled'),
                    'chunk_index': i,
                    'page_numbers': [chunk.start_page, chunk.end_page],
                    'section_headers': chunk.section_header
                }
        
        with open(self.output_dir / 'chunk_index.json', 'w') as f:
            json.dump(chunk_index, f, indent=2)
    
    def get_processing_stats(self) -> Dict:
        """Get processing statistics"""
        if not (self.output_dir / 'index.json').exists():
            return {}
        
        with open(self.output_dir / 'index.json', 'r') as f:
            index_data = json.load(f)
        
        return index_data.get('processing_info', {})
    
    def validate_processing(self) -> Dict[str, any]:
        """Validate the processing results"""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check if required files exist
        required_files = ['index.json', 'chunk_index.json']
        for file_name in required_files:
            file_path = self.output_dir / file_name
            if not file_path.exists():
                validation_results['valid'] = False
                validation_results['errors'].append(f"Missing required file: {file_name}")
        
        # Validate document directories
        if (self.output_dir / 'index.json').exists():
            with open(self.output_dir / 'index.json', 'r') as f:
                index_data = json.load(f)
            
            for doc in index_data.get('documents', []):
                doc_dir = self.output_dir / doc['id']
                if not doc_dir.exists():
                    validation_results['warnings'].append(f"Missing directory for document: {doc['id']}")
                else:
                    # Check required files in document directory
                    required_doc_files = ['document_info.json', 'chunks.json', 'raw_content.txt']
                    for file_name in required_doc_files:
                        if not (doc_dir / file_name).exists():
                            validation_results['warnings'].append(
                                f"Missing file {file_name} in document {doc['id']}"
                            )
        
        return validation_results
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        temp_files = list(self.temp_dir.glob('batch_*.json'))
        for temp_file in temp_files:
            temp_file.unlink()
        logger.info(f"Cleaned up {len(temp_files)} temporary files")

# Convenience function for running the pipeline
async def run_pipeline(config_path: str = None, input_path: str = None) -> List[ProcessedDocument]:
    """Run the complete processing pipeline"""
    from config.config_loader import load_config
    
    if config_path:
        config = load_config(config_path)
    else:
        config = load_config()
    
    pipeline = DocumentProcessingPipeline(config)
    results = await pipeline.process_documents(input_path)
    
    # Cleanup
    pipeline.cleanup_temp_files()
    
    return results

# CLI entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process OECD BEPS documents")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--input", help="Input directory path")
    parser.add_argument("--validate", action="store_true", help="Validate processing results")
    
    args = parser.parse_args()
    
    if args.validate:
        config = load_config(args.config) if args.config else load_config()
        pipeline = DocumentProcessingPipeline(config)
        validation = pipeline.validate_processing()
        print(json.dumps(validation, indent=2))
    else:
        results = asyncio.run(run_pipeline(args.config, args.input))
        print(f"Processed {len(results)} documents successfully")