#!/usr/bin/env python3
"""Tests for the vector store system."""

import asyncio
import pytest
import tempfile
import shutil
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vector_store import VectorStore, EmbeddingModel, DocumentProcessor
from src.config import ProcessingConfig


class TestVectorStore:
    """Test cases for VectorStore."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ProcessingConfig()
    
    @pytest.fixture
    async def vector_store(self, config):
        """Create and initialize vector store."""
        vs = VectorStore(config)
        await vs.initialize()
        return vs
    
    @pytest.mark.asyncio
    async def test_vector_store_initialization(self, config):
        """Test vector store initialization."""
        vs = VectorStore(config)
        await vs.initialize()
        assert vs.is_initialized()
    
    @pytest.mark.asyncio
    async def test_add_single_document(self, vector_store):
        """Test adding a single document."""
        doc = {
            'content': 'Test document content',
            'metadata': {'source': 'test'}
        }
        
        doc_id = await vector_store.add_document(doc)
        assert doc_id is not None
        assert isinstance(doc_id, str)
    
    @pytest.mark.asyncio
    async def test_add_multiple_documents(self, vector_store):
        """Test adding multiple documents."""
        docs = [
            {'content': 'Document 1', 'metadata': {'id': 1}},
            {'content': 'Document 2', 'metadata': {'id': 2}},
            {'content': 'Document 3', 'metadata': {'id': 3}}
        ]
        
        doc_ids = await vector_store.add_documents(docs)
        assert len(doc_ids) == 3
        assert all(isinstance(doc_id, str) for doc_id in doc_ids)
    
    @pytest.mark.asyncio
    async def test_query_vector_store(self, vector_store):
        """Test querying the vector store."""
        docs = [
            {'content': 'BEPS Action 1 digital economy challenges', 'metadata': {'action': 1}},
            {'content': 'BEPS Action 5 harmful tax practices', 'metadata': {'action': 5}},
            {'content': 'BEPS Action 13 country by country reporting', 'metadata': {'action': 13}}
        ]
        
        await vector_store.add_documents(docs)
        
        results = await vector_store.query("digital economy", top_k=2)
        assert len(results) <= 2
        assert all('score' in result for result in results)
        assert all('content' in result for result in results)
        assert all('metadata' in result for result in results)
    
    @pytest.mark.asyncio
    async def test_empty_query(self, vector_store):
        """Test querying with empty query."""
        results = await vector_store.query("")
        assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_save_load_vector_store(self, vector_store):
        """Test saving and loading vector store."""
        docs = [
            {'content': 'Test document 1', 'metadata': {'id': 1}},
            {'content': 'Test document 2', 'metadata': {'id': 2}}
        ]
        
        await vector_store.add_documents(docs)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_vector_store"
            
            # Save
            success = await vector_store.save_to_disk(str(save_path))
            assert success
            
            # Create new vector store and load
            new_vs = VectorStore(vector_store.config)
            await new_vs.initialize()
            loaded = await new_vs.load_from_disk(str(save_path))
            assert loaded
            
            # Verify data
            stats = new_vs.get_stats()
            assert stats['total_documents'] >= 2
    
    @pytest.mark.asyncio
    async def test_get_stats(self, vector_store):
        """Test getting vector store statistics."""
        stats = vector_store.get_stats()
        assert isinstance(stats, dict)
        assert 'total_documents' in stats
        assert 'total_chunks' in stats
        assert 'embedding_dimension' in stats


class TestEmbeddingModel:
    """Test cases for EmbeddingModel."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ProcessingConfig()
    
    @pytest.fixture
    async def embedding_model(self, config):
        """Create and initialize embedding model."""
        em = EmbeddingModel(config)
        await em.initialize()
        return em
    
    @pytest.mark.asyncio
    async def test_embedding_model_initialization(self, config):
        """Test embedding model initialization."""
        em = EmbeddingModel(config)
        await em.initialize()
        assert em.is_initialized()
    
    @pytest.mark.asyncio
    async def test_embed_text(self, embedding_model):
        """Test embedding single text."""
        text = "Test embedding text"
        embedding = await embedding_model.embed_text(text)
        
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == embedding_model.get_embedding_dimension()
    
    @pytest.mark.asyncio
    async def test_embed_batch(self, embedding_model):
        """Test embedding batch of texts."""
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = await embedding_model.embed_batch(texts)
        
        assert len(embeddings) == 3
        assert all(isinstance(emb, np.ndarray) for emb in embeddings)
        assert all(len(emb) == embedding_model.get_embedding_dimension() for emb in embeddings)
    
    @pytest.mark.asyncio
    async def test_similarity_calculation(self, embedding_model):
        """Test similarity calculation."""
        text1 = "The digital economy creates tax challenges"
        text2 = "Digital business models affect taxation"
        text3 = "Completely unrelated content"
        
        emb1 = await embedding_model.embed_text(text1)
        emb2 = await embedding_model.embed_text(text2)
        emb3 = await embedding_model.embed_text(text3)
        
        sim12 = await embedding_model.similarity(emb1, emb2)
        sim13 = await embedding_model.similarity(emb1, emb3)
        
        assert -1 <= sim12 <= 1
        assert -1 <= sim13 <= 1
        assert sim12 > sim13  # Similar texts should have higher similarity
    
    @pytest.mark.asyncio
    async def test_find_similar(self, embedding_model):
        """Test finding similar embeddings."""
        texts = ["Text A", "Text B", "Text C", "Text D"]
        embeddings = await embedding_model.embed_batch(texts)
        
        query_embedding = embeddings[0]
        similar = await embedding_model.find_similar(
            query_embedding, 
            embeddings[1:], 
            top_k=2
        )
        
        assert len(similar) <= 2
        assert all(isinstance(item, tuple) for item in similar)
        assert all(len(item) == 2 for item in similar)


class TestDocumentProcessor:
    """Test cases for DocumentProcessor."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ProcessingConfig()
    
    @pytest.fixture
    def document_processor(self, config):
        """Create document processor."""
        return DocumentProcessor(config)
    
    @pytest.mark.asyncio
    async def test_process_document(self, document_processor):
        """Test document processing."""
        content = "  This is   a test   document with   extra spaces.  \n\nNew line.  "
        processed = await document_processor.process_document(content)
        
        assert processed != content
        assert "  " not in processed.strip()
        assert processed.strip() == processed
    
    @pytest.mark.asyncio
    async def test_chunk_document(self, document_processor):
        """Test document chunking."""
        content = " ".join(["word"] * 1000)  # Long content
        chunks = await document_processor.chunk_document(content)
        
        assert len(chunks) > 1
        assert all('content' in chunk for chunk in chunks)
        assert all('metadata' in chunk for chunk in chunks)
        assert all('chunk_index' in chunk['metadata'] for chunk in chunks)
    
    @pytest.mark.asyncio
    async def test_extract_metadata(self, document_processor):
        """Test metadata extraction."""
        content = "This is a test document with several words and lines.\nSecond line here.\nThird line."
        metadata = await document_processor.extract_metadata(content, "test.txt")
        
        assert isinstance(metadata, dict)
        assert 'word_count' in metadata
        assert 'char_count' in metadata
        assert 'line_count' in metadata
        assert metadata['word_count'] > 0
        assert metadata['char_count'] > 0
        assert metadata['line_count'] > 0
    
    def test_validate_chunk_size(self, document_processor):
        """Test chunk size validation."""
        chunks = ["a" * 50, "b" * 100, "c" * 150]
        assert document_processor.validate_chunk_size(chunks)
        
        chunks = ["a" * 10, "b" * 1000]  # Too small and too large
        assert not document_processor.validate_chunk_size(chunks)
    
    def test_get_chunk_stats(self, document_processor):
        """Test chunk statistics."""
        chunks = ["a" * 50, "b" * 100, "c" * 150]
        stats = document_processor.get_chunk_stats(chunks)
        
        assert isinstance(stats, dict)
        assert stats['total_chunks'] == 3
        assert stats['min_chunk_size'] == 50
        assert stats['max_chunk_size'] == 150
        assert stats['avg_chunk_size'] == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])