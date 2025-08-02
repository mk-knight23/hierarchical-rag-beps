"""Document processing utilities for the vector store."""

import logging
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import asyncio

from config import ProcessingConfig

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Document processing and chunking utilities."""
    
    def __init__(self, config: ProcessingConfig):
        """Initialize the document processor."""
        self.config = config
        self.chunk_size = config.processing.chunk_size
        self.chunk_overlap = config.processing.chunk_overlap
        self.min_chunk_size = config.chunking.min_chunk_size
        self.max_chunk_size = config.chunking.max_chunk_size
        
    async def process_document(self, content: str) -> str:
        """
        Process a document for embedding.
        
        Args:
            content: Raw document content
            
        Returns:
            Processed content ready for embedding
        """
        # Clean the content
        content = self._clean_text(content)
        
        # Apply any additional processing
        content = self._normalize_text(content)
        
        return content
    
    async def chunk_document(self, content: str) -> List[Dict[str, Any]]:
        """
        Split a document into chunks for processing.
        
        Args:
            content: Document content
            
        Returns:
            List of chunks with metadata
        """
        # Clean and normalize content
        content = await self.process_document(content)
        
        # Split into chunks
        chunks = self._split_into_chunks(content)
        
        # Create chunk metadata
        chunk_data = []
        for i, chunk in enumerate(chunks):
            chunk_data.append({
                'content': chunk,
                'metadata': {
                    'chunk_index': i,
                    'chunk_size': len(chunk),
                    'total_chunks': len(chunks)
                }
            })
        
        return chunk_data
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might interfere with processing
        text = re.sub(r'[^\w\s.,!?;:()\-\[\]{}"\']', '', text)
        
        # Remove extra newlines
        text = re.sub(r'\n+', '\n', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _normalize_text(self, text: str) -> str:
        """Apply text normalization."""
        # Convert to lowercase if needed
        # text = text.lower()
        
        # Remove very short lines
        lines = text.split('\n')
        lines = [line for line in lines if len(line.strip()) > 10]
        text = '\n'.join(lines)
        
        return text
    
    def _split_into_chunks(self, text: str) -> List[str]:
        """Split text into chunks based on configuration."""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Find end position
            end = start + self.chunk_size
            
            # If we're at the end, just take the rest
            if end >= len(text):
                chunk = text[start:]
                if len(chunk) >= self.min_chunk_size:
                    chunks.append(chunk)
                break
            
            # Try to find a good break point
            if self.config.chunking.preserve_paragraphs:
                # Look for paragraph breaks
                paragraph_end = text.rfind('\n\n', start, end)
                if paragraph_end != -1 and paragraph_end > start + self.min_chunk_size:
                    end = paragraph_end + 2
            
            elif self.config.chunking.preserve_sentences:
                # Look for sentence breaks
                sentence_end = text.rfind('. ', start, end)
                if sentence_end != -1 and sentence_end > start + self.min_chunk_size:
                    end = sentence_end + 1
            
            # Ensure we don't create chunks that are too small
            if end - start < self.min_chunk_size:
                end = min(start + self.chunk_size, len(text))
            
            chunk = text[start:end].strip()
            if len(chunk) >= self.min_chunk_size:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            
            # Ensure we make progress
            if start >= len(text) - self.min_chunk_size:
                break
        
        return chunks
    
    async def extract_metadata(self, content: str, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract metadata from document content.
        
        Args:
            content: Document content
            file_path: Original file path
            
        Returns:
            Dictionary with extracted metadata
        """
        metadata = {
            'word_count': len(content.split()),
            'char_count': len(content),
            'line_count': len(content.split('\n')),
            'processed': True
        }
        
        if file_path:
            path = Path(file_path)
            metadata.update({
                'filename': path.name,
                'file_extension': path.suffix,
                'file_size': path.stat().st_size if path.exists() else 0
            })
        
        # Extract potential keywords
        words = content.lower().split()
        word_freq = {}
        for word in words:
            word = re.sub(r'[^\w]', '', word)
            if len(word) > 3 and word not in {'this', 'that', 'with', 'have', 'from'}:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top keywords
        top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        metadata['keywords'] = [word for word, freq in top_keywords]
        
        return metadata
    
    async def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a file and return chunks with metadata.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with processed content and metadata
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read file content
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(path, 'r', encoding='latin-1') as f:
                content = f.read()
        
        # Extract metadata
        metadata = await self.extract_metadata(content, file_path)
        
        # Create chunks
        chunks = await self.chunk_document(content)
        
        return {
            'content': content,
            'metadata': metadata,
            'chunks': chunks
        }
    
    def validate_chunk_size(self, chunks: List[str]) -> bool:
        """Validate that chunks meet size requirements."""
        for chunk in chunks:
            if len(chunk) < self.min_chunk_size or len(chunk) > self.max_chunk_size:
                return False
        return True
    
    def get_chunk_stats(self, chunks: List[str]) -> Dict[str, Any]:
        """Get statistics about chunks."""
        if not chunks:
            return {
                'total_chunks': 0,
                'avg_chunk_size': 0,
                'min_chunk_size': 0,
                'max_chunk_size': 0
            }
        
        sizes = [len(chunk) for chunk in chunks]
        return {
            'total_chunks': len(chunks),
            'avg_chunk_size': sum(sizes) / len(sizes),
            'min_chunk_size': min(sizes),
            'max_chunk_size': max(sizes)
        }