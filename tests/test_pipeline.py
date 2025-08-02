#!/usr/bin/env python3
"""
Comprehensive tests for the document processing pipeline.
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
import sys

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from processing.pipeline import DocumentProcessor
from processing.document_loader import DocumentLoader
from processing.chunker import TextChunker
from processing.summary_generator import SummaryGenerator
from config.config_loader import load_config
from models.data_structures import Document, Chunk, Summary, ProcessedDocument


class TestDataStructures:
    """Test the core data structures."""
    
    def test_document_creation(self):
        """Test Document creation and validation."""
        doc = Document(
            document_id="test_doc_123",
            content="This is a test document content.",
            metadata={
                "title": "Test Document",
                "source_file": "/path/to/test.pdf",
                "page_count": 5
            }
        )
        
        assert doc.document_id == "test_doc_123"
        assert doc.content == "This is a test document content."
        assert doc.metadata["title"] == "Test Document"
    
    def test_chunk_creation(self):
        """Test Chunk creation and validation."""
        chunk = Chunk(
            chunk_id="chunk_1",
            document_id="test_doc_123",
            text="This is a test chunk.",
            metadata={
                "start_index": 0,
                "end_index": 21,
                "token_count": 5
            }
        )
        
        assert chunk.chunk_id == "chunk_1"
        assert chunk.document_id == "test_doc_123"
        assert chunk.text == "This is a test chunk."
    
    def test_summary_creation(self):
        """Test Summary creation and validation."""
        summary = Summary(
            document_id="test_doc_123",
            text="This is a test summary.",
            metadata={
                "keywords": ["test", "summary"],
                "topics": ["testing", "validation"]
            }
        )
        
        assert summary.document_id == "test_doc_123"
        assert summary.text == "This is a test summary."
        assert "test" in summary.metadata["keywords"]
    
    def test_processed_document(self):
        """Test ProcessedDocument creation."""
        chunks = [
            Chunk(
                chunk_id="chunk_1",
                document_id="test_doc_123",
                text="Chunk 1 text",
                metadata={"token_count": 2}
            ),
            Chunk(
                chunk_id="chunk_2",
                document_id="test_doc_123",
                text="Chunk 2 text",
                metadata={"token_count": 2}
            )
        ]
        
        summary = Summary(
            document_id="test_doc_123",
            text="Test summary",
            metadata={"keywords": ["test"]}
        )
        
        processed_doc = ProcessedDocument(
            document_id="test_doc_123",
            chunks=chunks,
            summary=summary,
            metadata={"processing_time": 1.5}
        )
        
        assert processed_doc.document_id == "test_doc_123"
        assert len(processed_doc.chunks) == 2
        assert processed_doc.summary.text == "Test summary"


class TestDocumentLoader:
    """Test the document loader."""
    
    def test_load_text_file(self):
        """Test loading a text file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test text file content.")
            temp_path = Path(f.name)
        
        try:
            loader = DocumentLoader()
            doc = loader.load_document(temp_path)
            
            assert doc.content == "This is a test text file content."
            assert doc.metadata["source_file"] == str(temp_path)
            assert doc.metadata["file_type"] == "txt"
        finally:
            temp_path.unlink()
    
    def test_load_nonexistent_file(self):
        """Test loading a non-existent file."""
        loader = DocumentLoader()
        
        with pytest.raises(FileNotFoundError):
            loader.load_document(Path("/nonexistent/file.txt"))
    
    def test_unsupported_file_type(self):
        """Test loading an unsupported file type."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.doc', delete=False) as f:
            f.write("Test content")
            temp_path = Path(f.name)
        
        try:
            loader = DocumentLoader()
            
            with pytest.raises(ValueError, match="Unsupported file type"):
                loader.load_document(temp_path)
        finally:
            temp_path.unlink()


class TestTextChunker:
    """Test the text chunker."""
    
    def test_basic_chunking(self):
        """Test basic text chunking."""
        chunker = TextChunker(
            chunk_size=50,
            chunk_overlap=10
        )
        
        text = "This is a longer text that should be chunked into smaller pieces. " * 10
        chunks = chunker.chunk_text(text, "test_doc")
        
        assert len(chunks) > 1
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert all(chunk.document_id == "test_doc" for chunk in chunks)
    
    def test_chunk_metadata(self):
        """Test chunk metadata generation."""
        chunker = TextChunker(
            chunk_size=30,
            chunk_overlap=5
        )
        
        text = "This is sentence one. This is sentence two. This is sentence three."
        chunks = chunker.chunk_text(text, "test_doc")
        
        assert len(chunks) >= 1
        for chunk in chunks:
            assert "start_index" in chunk.metadata
            assert "end_index" in chunk.metadata
            assert "token_count" in chunk.metadata
    
    def test_semantic_boundaries(self):
        """Test that semantic boundaries are respected."""
        chunker = TextChunker(
            chunk_size=100,
            chunk_overlap=20
        )
        
        # Text with clear sentence boundaries
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = chunker.chunk_text(text, "test_doc")
        
        # Ensure chunks don't break in the middle of sentences
        for chunk in chunks:
            # Each chunk should end with a sentence boundary or be the last chunk
            text = chunk.text.strip()
            assert text.endswith('.') or chunk == chunks[-1]


class TestSummaryGenerator:
    """Test the summary generator."""
    
    @pytest.mark.asyncio
    async def test_generate_summary(self):
        """Test summary generation."""
        generator = SummaryGenerator()
        
        text = """
        The OECD BEPS Pillar Two Model Rules represent a significant development in international
        taxation. These rules ensure that large multinational enterprises pay a minimum level
        of tax on their global income. The key components include the Income Inclusion Rule,
        Undertaxed Payments Rule, and Subject to Tax Rule.
        """
        
        summary = await generator.generate_summary(text, "test_doc")
        
        assert isinstance(summary, Summary)
        assert summary.document_id == "test_doc"
        assert len(summary.text) > 0
        assert "keywords" in summary.metadata
        assert "topics" in summary.metadata
    
    def test_extract_keywords(self):
        """Test keyword extraction."""
        generator = SummaryGenerator()
        
        text = "The OECD BEPS Pillar Two rules address tax challenges in international taxation."
        keywords = generator._extract_keywords(text)
        
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        assert all(isinstance(kw, str) for kw in keywords)


class TestDocumentProcessor:
    """Test the complete document processor."""
    
    @pytest.mark.asyncio
    async def test_process_text_document(self):
        """Test processing a text document."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("""
            OECD BEPS Pillar Two Model Rules
            
            The Global Anti-Base Erosion (GloBE) Rules ensure that large multinational
            enterprises pay a minimum level of tax on their global income. These rules
            apply to companies with annual revenue of at least EUR 750 million.
            
            Key Components:
            1. Income Inclusion Rule (IIR)
            2. Undertaxed Payments Rule (UTPR)
            3. Subject to Tax Rule (STTR)
            """)
            temp_path = Path(f.name)
        
        try:
            config = load_config()
            processor = DocumentProcessor(config)
            
            processed_doc = await processor.process_document(temp_path)
            
            assert isinstance(processed_doc, ProcessedDocument)
            assert len(processed_doc.chunks) > 0
            assert isinstance(processed_doc.summary, Summary)
            assert processed_doc.metadata["source_file"] == str(temp_path)
            
        finally:
            temp_path.unlink()
    
    @pytest.mark.asyncio
    async def test_empty_document(self):
        """Test processing an empty document."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("")
            temp_path = Path(f.name)
        
        try:
            config = load_config()
            processor = DocumentProcessor(config)
            
            with pytest.raises(ValueError, match="Empty document"):
                await processor.process_document(temp_path)
                
        finally:
            temp_path.unlink()


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_processing(self):
        """Test complete end-to-end processing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test document
            test_file = temp_path / "test_document.txt"
            test_file.write_text("""
            OECD BEPS Pillar Two Implementation
                
            The implementation of Pillar Two requires careful consideration of:
            1. Domestic minimum tax rules
            2. Safe harbor provisions
            3. Administrative procedures
            4. Coordination with existing tax treaties
                
            Member countries must ensure that their domestic legislation is
            consistent with the model rules provided by the OECD.
            """)
            
            # Process document
            config = load_config()
            processor = DocumentProcessor(config)
            
            processed_doc = await processor.process_document(test_file)
            
            # Verify results
            assert processed_doc.document_id is not None
            assert len(processed_doc.chunks) > 0
            assert processed_doc.summary.text is not None
            assert len(processed_doc.summary.metadata["keywords"]) > 0
            
            # Test serialization
            serialized = processed_doc.to_dict()
            assert "document_id" in serialized
            assert "chunks" in serialized
            assert "summary" in serialized
            
            # Test deserialization
            deserialized = ProcessedDocument.from_dict(serialized)
            assert deserialized.document_id == processed_doc.document_id
            assert len(deserialized.chunks) == len(processed_doc.chunks)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])