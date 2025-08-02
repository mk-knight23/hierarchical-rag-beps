"""Vector store implementation for document storage and retrieval."""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path
import json
import time

from .embeddings import EmbeddingModel
from .document_processor import DocumentProcessor
from config import ProcessingConfig

logger = logging.getLogger(__name__)


class VectorStore:
    """Vector store for storing and retrieving document embeddings."""
    
    def __init__(self, config: ProcessingConfig):
        """Initialize the vector store."""
        self.config = config
        self.embedding_model = None
        self.document_processor = None
        self._documents = []
        self._embeddings = []
        self._metadata = []
        self._initialized = False
        
    async def initialize(self):
        """Initialize the vector store components."""
        try:
            self.embedding_model = EmbeddingModel(self.config)
            self.document_processor = DocumentProcessor(self.config)
            
            # Load existing documents if any
            await self._load_existing_documents()
            
            self._initialized = True
            logger.info("Vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            raise
    
    async def add_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 10
    ) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents with 'content' and 'metadata'
            batch_size: Batch size for processing
            
        Returns:
            List of document IDs
        """
        if not self._initialized:
            raise RuntimeError("Vector store not initialized")
        
        document_ids = []
        
        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_ids = await self._process_batch(batch)
            document_ids.extend(batch_ids)
        
        logger.info(f"Added {len(document_ids)} documents to vector store")
        return document_ids
    
    async def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            threshold: Similarity threshold
            filters: Optional filters for metadata
            
        Returns:
            List of search results with scores
        """
        if not self._initialized:
            raise RuntimeError("Vector store not initialized")
        
        if not self._documents:
            return []
        
        # Generate query embedding
        query_embedding = await self.embedding_model.embed_text(query)
        
        # Calculate similarities
        similarities = []
        for i, doc_embedding in enumerate(self._embeddings):
            # Check filters
            if filters and not self._apply_filters(self._metadata[i], filters):
                continue
            
            similarity = self._calculate_similarity(query_embedding, doc_embedding)
            if similarity >= threshold:
                similarities.append((i, similarity))
        
        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = similarities[:top_k]
        
        results = []
        for idx, score in top_results:
            doc = self._documents[idx]
            metadata = self._metadata[idx]
            
            results.append({
                'content': doc,
                'metadata': metadata,
                'score': score,
                'document_id': metadata.get('document_id', str(idx))
            })
        
        return results
    
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """
        Delete documents by IDs.
        
        Args:
            document_ids: List of document IDs to delete
            
        Returns:
            True if successful
        """
        if not self._initialized:
            raise RuntimeError("Vector store not initialized")
        
        indices_to_remove = []
        for i, metadata in enumerate(self._metadata):
            if metadata.get('document_id') in document_ids:
                indices_to_remove.append(i)
        
        # Remove in reverse order to maintain indices
        for idx in sorted(indices_to_remove, reverse=True):
            del self._documents[idx]
            del self._embeddings[idx]
            del self._metadata[idx]
        
        logger.info(f"Deleted {len(indices_to_remove)} documents")
        return True
    
    async def get_document_count(self) -> int:
        """Get total number of documents."""
        return len(self._documents)
    
    async def get_document_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID."""
        for i, metadata in enumerate(self._metadata):
            if metadata.get('document_id') == document_id:
                return {
                    'content': self._documents[i],
                    'metadata': metadata
                }
        return None
    
    async def update_document(
        self,
        document_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update document content and metadata."""
        for i, meta in enumerate(self._metadata):
            if meta.get('document_id') == document_id:
                # Update content and re-embed
                embedding = await self.embedding_model.embed_text(content)
                
                self._documents[i] = content
                self._embeddings[i] = embedding
                
                if metadata:
                    self._metadata[i].update(metadata)
                
                logger.info(f"Updated document {document_id}")
                return True
        
        return False
    
    async def _process_batch(self, batch: List[Dict[str, Any]]) -> List[str]:
        """Process a batch of documents."""
        document_ids = []
        
        for doc in batch:
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})
            
            # Generate document ID if not provided
            if 'document_id' not in metadata:
                metadata['document_id'] = f"doc_{int(time.time() * 1000)}_{len(self._documents)}"
            
            # Process document
            processed_content = await self.document_processor.process_document(content)
            
            # Generate embedding
            embedding = await self.embedding_model.embed_text(processed_content)
            
            # Store document
            self._documents.append(processed_content)
            self._embeddings.append(embedding)
            self._metadata.append(metadata)
            
            document_ids.append(metadata['document_id'])
        
        return document_ids
    
    async def _load_existing_documents(self):
        """Load existing documents from storage."""
        # In a real implementation, this would load from persistent storage
        # For now, we'll just initialize empty storage
        pass
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        # Ensure embeddings are 1D
        embedding1 = embedding1.flatten()
        embedding2 = embedding2.flatten()
        
        # Calculate cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def _apply_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Apply metadata filters."""
        for key, value in filters.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True
    
    async def save_to_disk(self, directory: str) -> bool:
        """Save vector store to disk."""
        try:
            save_dir = Path(directory)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save documents
            with open(save_dir / 'documents.json', 'w') as f:
                json.dump(self._documents, f, indent=2)
            
            # Save embeddings
            np.save(save_dir / 'embeddings.npy', np.array(self._embeddings))
            
            # Save metadata
            with open(save_dir / 'metadata.json', 'w') as f:
                json.dump(self._metadata, f, indent=2)
            
            logger.info(f"Vector store saved to {directory}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save vector store: {str(e)}")
            return False
    
    async def load_from_disk(self, directory: str) -> bool:
        """Load vector store from disk."""
        try:
            load_dir = Path(directory)
            
            if not load_dir.exists():
                logger.warning(f"Directory {directory} does not exist")
                return False
            
            # Load documents
            with open(load_dir / 'documents.json', 'r') as f:
                self._documents = json.load(f)
            
            # Load embeddings
            embeddings_array = np.load(load_dir / 'embeddings.npy')
            self._embeddings = [emb for emb in embeddings_array]
            
            # Load metadata
            with open(load_dir / 'metadata.json', 'r') as f:
                self._metadata = json.load(f)
            
            logger.info(f"Vector store loaded from {directory}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load vector store: {str(e)}")
            return False
    
    async def clear(self):
        """Clear all documents from the vector store."""
        self._documents.clear()
        self._embeddings.clear()
        self._metadata.clear()
        logger.info("Vector store cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return {
            'total_documents': len(self._documents),
            'embedding_dimension': len(self._embeddings[0]) if self._embeddings else 0,
            'initialized': self._initialized
        }