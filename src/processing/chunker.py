import re
import logging
from typing import List, Dict, Optional
from models.data_structures import Chunk, Document

logger = logging.getLogger(__name__)

class TextChunker:
    """Intelligent text chunking with semantic boundary detection"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.chunk_size = config.get('chunking', {}).get('chunk_size', 1000)
        self.overlap = config.get('chunking', {}).get('overlap', 200)
        self.min_chunk_size = config.get('chunking', {}).get('min_chunk_size', 100)
        self.max_chunk_size = config.get('chunking', {}).get('max_chunk_size', 1500)
        
    def chunk_document(self, document: Document) -> List[Chunk]:
        """Split document into chunks with metadata preservation"""
        if not document.content:
            logger.warning(f"Empty document: {document.id}")
            return []
            
        # Tokenize content into sentences
        sentences = self._split_into_sentences(document.content)
        
        # Group sentences into chunks respecting boundaries
        chunks = self._create_chunks(sentences, document)
        
        logger.info(f"Created {len(chunks)} chunks from document {document.id}")
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex"""
        # Split on sentence boundaries while preserving punctuation
        sentence_endings = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_endings, text.strip())
        
        # Clean up sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _create_chunks(self, sentences: List[str], document: Document) -> List[Chunk]:
        """Create chunks from sentences with overlap"""
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # Check if adding this sentence would exceed chunk size
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Create chunk from current sentences
                chunk_text = ' '.join(current_chunk)
                chunk = self._create_chunk(
                    chunk_text, 
                    document, 
                    chunk_index,
                    len(chunks)
                )
                chunks.append(chunk)
                
                # Prepare overlap for next chunk
                overlap_sentences = self._get_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences
                current_length = sum(len(s) for s in current_chunk)
                chunk_index += 1
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Create final chunk if there's remaining content
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk = self._create_chunk(
                chunk_text, 
                document, 
                chunk_index,
                len(chunks)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """Get sentences for overlap based on token count"""
        overlap_sentences = []
        overlap_length = 0
        
        # Work backwards through sentences
        for sentence in reversed(sentences):
            sentence_length = len(sentence)
            if overlap_length + sentence_length <= self.overlap:
                overlap_sentences.insert(0, sentence)
                overlap_length += sentence_length
            else:
                break
        
        return overlap_sentences
    
    def _create_chunk(self, text: str, document: Document, chunk_index: int,
                     global_chunk_index: int) -> Chunk:
        """Create a chunk with metadata"""
        # Extract page numbers from chunk text
        page_numbers = self._extract_page_numbers(text)
        
        # Extract section headers from context
        section_headers = self._extract_section_headers(text, document.content)
        
        # Generate chunk ID
        chunk_id = f"{document.id}_chunk_{chunk_index}"
        
        # Calculate position in document
        start_pos = document.content.find(text)
        end_pos = start_pos + len(text) if start_pos != -1 else 0
        
        return Chunk(
            id=chunk_id,
            document_id=document.id,
            text=text,
            start_page=min(page_numbers) if page_numbers else 1,
            end_page=max(page_numbers) if page_numbers else 1,
            token_count=len(text.split()),
            metadata={
                'source_file': str(document.file_path) if document.file_path else '',
                'chunk_size': str(len(text)),
                'word_count': str(len(text.split())),
                'sentence_count': str(len(self._split_into_sentences(text))),
                'chunk_index': str(chunk_index),
                'global_chunk_index': str(global_chunk_index),
                'start_position': str(start_pos),
                'end_position': str(end_pos),
                'page_numbers': ','.join(map(str, page_numbers)),
                'section_headers': '|'.join(section_headers)
            }
        )
    
    def _extract_page_numbers(self, text: str) -> List[int]:
        """Extract page numbers from chunk text"""
        # Look for [Page X] markers
        page_pattern = r'\[Page (\d+)\]'
        matches = re.findall(page_pattern, text)
        return [int(match) for match in matches]
    
    def _extract_section_headers(self, chunk_text: str, full_text: str) -> List[str]:
        """Extract section headers from document context"""
        # Simple heuristic: look for lines that are all caps or end with colon
        headers = []
        
        # Find the position of this chunk in the full text
        chunk_start = full_text.find(chunk_text)
        if chunk_start == -1:
            return headers
            
        # Look backwards for section headers
        search_start = max(0, chunk_start - 1000)
        search_text = full_text[search_start:chunk_start]
        
        # Common patterns for headers
        lines = search_text.split('\n')
        for line in lines[-10:]:  # Check last 10 lines
            line = line.strip()
            if line and (line.isupper() or line.endswith(':') or 
                        (len(line) < 100 and line.count(' ') < 5)):
                headers.append(line)
        
        return headers[-3:]  # Return last 3 headers
    
    def validate_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Validate and filter chunks"""
        valid_chunks = []
        
        for chunk in chunks:
            # Check minimum size
            if len(chunk.text) < self.min_chunk_size:
                logger.warning(f"Chunk too small: {chunk.id} ({len(chunk.text)} chars)")
                continue
                
            # Check maximum size
            if len(chunk.text) > self.max_chunk_size:
                logger.warning(f"Chunk too large: {chunk.id} ({len(chunk.text)} chars)")
                # Split oversized chunks
                sub_chunks = self._split_oversized_chunk(chunk)
                valid_chunks.extend(sub_chunks)
            else:
                valid_chunks.append(chunk)
        
        return valid_chunks
    
    def _split_oversized_chunk(self, chunk: Chunk) -> List[Chunk]:
        """Split an oversized chunk into smaller pieces"""
        content = chunk.text
        chunk_size = len(content)
        
        # Calculate number of sub-chunks needed
        num_sub_chunks = (chunk_size // self.max_chunk_size) + 1
        
        sub_chunks = []
        chunk_length = len(content)
        sub_chunk_size = chunk_length // num_sub_chunks
        
        for i in range(num_sub_chunks):
            start = i * sub_chunk_size
            end = min((i + 1) * sub_chunk_size + self.overlap, chunk_length)
            
            if start < end:
                sub_content = content[start:end]
                
                # Extract page numbers for sub-chunk
                page_numbers = self._extract_page_numbers(sub_content)
                
                sub_chunk = Chunk(
                    id=f"{chunk.id}_sub_{i}",
                    document_id=chunk.document_id,
                    text=sub_content,
                    start_page=min(page_numbers) if page_numbers else 1,
                    end_page=max(page_numbers) if page_numbers else 1,
                    token_count=len(sub_content.split()),
                    metadata={
                        'source_file': chunk.metadata.get('source_file', ''),
                        'chunk_size': str(len(sub_content)),
                        'word_count': str(len(sub_content.split())),
                        'sentence_count': str(len(self._split_into_sentences(sub_content))),
                        'sub_chunk_index': str(i),
                        'parent_chunk_id': chunk.id,
                        'chunk_index': str(chunk.metadata.get('chunk_index', '0')),
                        'global_chunk_index': str(chunk.metadata.get('global_chunk_index', '0')),
                        'start_position': str(start),
                        'end_position': str(end)
                    }
                )
                sub_chunks.append(sub_chunk)
        
        return sub_chunks