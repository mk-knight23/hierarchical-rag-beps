"""Hierarchical RAG implementation for BEPS document retrieval."""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np

from ..vector_store.vector_store import VectorStore
from ..config.config import ProcessingConfig
from ..processing.document_loader import DocumentLoader
from ..processing.chunker import TextChunker

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from hierarchical retrieval."""
    content: str
    score: float
    source: str
    chunk_id: str
    metadata: Dict[str, Any]


class HierarchicalRAG:
    """Hierarchical Retrieval-Augmented Generation for BEPS documents."""
    
    def __init__(self, config: ProcessingConfig, vector_store: VectorStore):
        """Initialize HierarchicalRAG.
        
        Args:
            config: Processing configuration
            vector_store: Vector store instance
        """
        self.config = config
        self.vector_store = vector_store
        self.document_loader = DocumentLoader(config)
        self.chunker = Chunker(config)
        
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        confidence_threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Retrieve relevant documents using hierarchical approach.
        
        Args:
            query: User query
            top_k: Number of top results to return
            confidence_threshold: Minimum confidence score
            filters: Optional filters for retrieval
            
        Returns:
            List of retrieval results
        """
        try:
            # Step 1: Semantic search
            semantic_results = await self._semantic_search(
                query, top_k * 2, filters
            )
            
            # Step 2: Rerank results
            reranked_results = await self._rerank_results(
                query, semantic_results
            )
            
            # Step 3: Apply confidence threshold
            filtered_results = [
                result for result in reranked_results
                if result.score >= confidence_threshold
            ]
            
            # Step 4: Return top_k results
            return filtered_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in hierarchical retrieval: {e}")
            return []
    
    async def _semantic_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Perform semantic search using vector store.
        
        Args:
            query: User query
            top_k: Number of results to retrieve
            filters: Optional filters
            
        Returns:
            List of semantic search results
        """
        try:
            # Get query embedding
            query_embedding = await self.vector_store.get_query_embedding(query)
            
            # Search vector store
            search_results = await self.vector_store.search(
                query_embedding,
                top_k=top_k,
                filters=filters
            )
            
            # Convert to retrieval results
            results = []
            for result in search_results:
                retrieval_result = RetrievalResult(
                    content=result.content,
                    score=result.score,
                    source=result.metadata.get("source", "unknown"),
                    chunk_id=result.metadata.get("chunk_id", ""),
                    metadata=result.metadata
                )
                results.append(retrieval_result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    async def _rerank_results(
        self,
        query: str,
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Rerank results using cross-encoder or other reranking strategy.
        
        Args:
            query: Original query
            results: Results to rerank
            
        Returns:
            Reranked results
        """
        try:
            # Simple reranking based on query-document similarity
            # In production, this could use a cross-encoder model
            
            # Calculate enhanced scores
            for result in results:
                # Boost score based on exact matches
                query_terms = query.lower().split()
                content_lower = result.content.lower()
                
                exact_match_bonus = 0.0
                for term in query_terms:
                    if term in content_lower:
                        exact_match_bonus += 0.1
                
                # Apply bonus
                result.score = min(1.0, result.score + exact_match_bonus)
            
            # Sort by new scores
            results.sort(key=lambda x: x.score, reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in reranking: {e}")
            return results
    
    async def add_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """Add documents to the knowledge base.
        
        Args:
            documents: List of document paths or content
            metadata: Optional metadata for each document
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if metadata is None:
                metadata = [{}] * len(documents)
            
            for doc_path, meta in zip(documents, metadata):
                # Load document
                content = await self.document_loader.load_document(doc_path)
                if not content:
                    continue
                
                # Chunk document
                chunks = await self.chunker.chunk_document(content)
                
                # Add to vector store
                for i, chunk in enumerate(chunks):
                    chunk_metadata = {
                        **meta,
                        "source": doc_path,
                        "chunk_id": f"{doc_path}_chunk_{i}",
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }
                    
                    await self.vector_store.add_document(
                        content=chunk,
                        metadata=chunk_metadata
                    )
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False
    
    async def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about the document collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            stats = await self.vector_store.get_collection_stats()
            return {
                "total_documents": stats.get("total_documents", 0),
                "total_chunks": stats.get("total_chunks", 0),
                "average_chunk_size": stats.get("average_chunk_size", 0),
                "collection_size": stats.get("collection_size", 0)
            }
        except Exception as e:
            logger.error(f"Error getting document stats: {e}")
            return {}
    
    async def delete_documents(
        self,
        source_filter: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Delete documents from the knowledge base.
        
        Args:
            source_filter: Filter by source document
            metadata_filter: Filter by metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            await self.vector_store.delete_documents(
                source_filter=source_filter,
                metadata_filter=metadata_filter
            )
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False