"""
RAG-based Response Handler for Decision-Making Agent

This module provides intelligent retrieval and generation capabilities
using the hierarchical RAG system for domain-specific queries.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import asyncio
from pathlib import Path
import json

from ..hierarchical_rag import HierarchicalRAG
from ..vector_store import VectorStore
from ..vector_store.embeddings import EmbeddingModel
from ..config import ProcessingConfig

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Data class for RAG response."""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    query_relevance: float
    retrieval_time: float
    chunks_used: int
    metadata: Dict[str, Any]


@dataclass
class RetrievalContext:
    """Data class for retrieval context."""
    query: str
    top_k: int
    filters: Dict[str, Any]
    rerank: bool
    max_tokens: int


class RAGHandler:
    """
    RAG-based response handler using hierarchical retrieval system.
    """
    
    def __init__(self, config: ProcessingConfig, vector_store: VectorStore):
        """
        Initialize RAG handler.
        
        Args:
            config: Processing configuration
            vector_store: Vector store instance
        """
        self.config = config
        self.vector_store = vector_store
        self.hierarchical_rag = None
        self.embedding_model = None
        
    async def initialize(self):
        """Initialize the RAG system components."""
        try:
            self.embedding_model = EmbeddingModel(self.config.embeddings)
            self.hierarchical_rag = HierarchicalRAG(
                config=self.config,
                vector_store=self.vector_store,
                embedding_model=self.embedding_model
            )
            logger.info("RAG handler initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG handler: {str(e)}")
            raise
    
    async def generate_response(
        self,
        query: str,
        context: Optional[RetrievalContext] = None
    ) -> RAGResponse:
        """
        Generate response using RAG retrieval and generation.
        
        Args:
            query: User query
            context: Retrieval context parameters
            
        Returns:
            RAGResponse with generated answer and metadata
        """
        if not self.hierarchical_rag:
            await self.initialize()
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Set default context
            if context is None:
                context = RetrievalContext(
                    query=query,
                    top_k=5,
                    filters={},
                    rerank=True,
                    max_tokens=1000
                )
            
            # Perform hierarchical retrieval
            retrieval_results = await self._perform_retrieval(context)
            
            if not retrieval_results:
                return RAGResponse(
                    answer="I couldn't find relevant information in the available documents.",
                    sources=[],
                    confidence=0.0,
                    query_relevance=0.0,
                    retrieval_time=asyncio.get_event_loop().time() - start_time,
                    chunks_used=0,
                    metadata={"error": "no_relevant_chunks"}
                )
            
            # Generate answer from retrieved chunks
            answer, sources, confidence = await self._generate_answer(
                query, retrieval_results
            )
            
            retrieval_time = asyncio.get_event_loop().time() - start_time
            
            return RAGResponse(
                answer=answer,
                sources=sources,
                confidence=confidence,
                query_relevance=self._calculate_query_relevance(query, retrieval_results),
                retrieval_time=retrieval_time,
                chunks_used=len(retrieval_results),
                metadata={
                    "retrieval_method": "hierarchical",
                    "top_k": context.top_k,
                    "rerank": context.rerank
                }
            )
            
        except Exception as e:
            logger.error(f"RAG response generation failed: {str(e)}")
            return RAGResponse(
                answer="I encountered an error while processing your query.",
                sources=[],
                confidence=0.0,
                query_relevance=0.0,
                retrieval_time=asyncio.get_event_loop().time() - start_time,
                chunks_used=0,
                metadata={"error": str(e)}
            )
    
    async def _perform_retrieval(
        self,
        context: RetrievalContext
    ) -> List[Dict[str, Any]]:
        """Perform hierarchical retrieval."""
        try:
            # Use hierarchical RAG for retrieval
            results = await self.hierarchical_rag.retrieve(
                query=context.query,
                top_k=context.top_k,
                filters=context.filters,
                rerank=context.rerank
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Retrieval failed: {str(e)}")
            return []
    
    async def _generate_answer(
        self,
        query: str,
        retrieval_results: List[Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, Any]], float]:
        """Generate answer from retrieved chunks."""
        if not retrieval_results:
            return "No relevant information found.", [], 0.0
        
        # Extract relevant chunks
        chunks = []
        sources = []
        
        for result in retrieval_results:
            chunk_text = result.get('text', '')
            metadata = result.get('metadata', {})
            score = result.get('score', 0.0)
            
            chunks.append({
                'text': chunk_text,
                'score': score,
                'metadata': metadata
            })
            
            # Build source information
            source_info = {
                'document': metadata.get('source_file', 'Unknown'),
                'page': metadata.get('page', 0),
                'section': metadata.get('section', ''),
                'score': score,
                'type': metadata.get('type', 'text')
            }
            sources.append(source_info)
        
        # Generate answer using context from chunks
        answer = await self._create_answer_from_chunks(query, chunks)
        
        # Calculate confidence based on chunk scores and relevance
        confidence = self._calculate_confidence(chunks)
        
        return answer, sources, confidence
    
    async def _create_answer_from_chunks(
        self,
        query: str,
        chunks: List[Dict[str, Any]]
    ) -> str:
        """Create a coherent answer from retrieved chunks."""
        if not chunks:
            return "I couldn't find relevant information to answer your question."
        
        # Sort chunks by relevance score
        chunks.sort(key=lambda x: x['score'], reverse=True)
        
        # Build context from top chunks
        context_parts = []
        used_chunks = 0
        
        for chunk in chunks[:3]:  # Use top 3 chunks
            text = chunk['text'].strip()
            if text and len(text) > 50:  # Filter out very short chunks
                context_parts.append(text)
                used_chunks += 1
        
        if not context_parts:
            return "The retrieved information doesn't contain sufficient detail to answer your question."
        
        # Create answer based on query type
        answer = self._format_answer(query, context_parts)
        
        return answer
    
    def _format_answer(self, query: str, context_parts: List[str]) -> str:
        """Format the final answer from context parts."""
        # Simple formatting - in production, use LLM for better generation
        answer = f"Based on the available information:\n\n"
        
        # Add context parts with some basic formatting
        for i, part in enumerate(context_parts, 1):
            # Clean up the text
            clean_part = part.replace('\n', ' ').strip()
            if len(clean_part) > 200:
                clean_part = clean_part[:200] + "..."
            
            answer += f"{clean_part}\n\n"
        
        # Add source reference
        answer += "This information is derived from the processed OECD BEPS documents."
        
        return answer
    
    def _calculate_confidence(self, chunks: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on chunk relevance."""
        if not chunks:
            return 0.0
        
        # Average score of top chunks
        top_scores = [chunk['score'] for chunk in chunks[:3]]
        avg_score = sum(top_scores) / len(top_scores)
        
        # Adjust based on number of relevant chunks
        relevant_chunks = len([c for c in chunks if c['score'] > 0.7])
        coverage_bonus = min(relevant_chunks * 0.1, 0.3)
        
        confidence = min(avg_score + coverage_bonus, 1.0)
        return confidence
    
    def _calculate_query_relevance(
        self,
        query: str,
        retrieval_results: List[Dict[str, Any]]
    ) -> float:
        """Calculate how relevant the retrieved results are to the query."""
        if not retrieval_results:
            return 0.0
        
        # Simple relevance calculation based on scores
        scores = [result.get('score', 0.0) for result in retrieval_results]
        avg_score = sum(scores) / len(scores)
        
        return avg_score
    
    async def get_document_summary(
        self,
        document_path: str,
        max_chunks: int = 5
    ) -> Dict[str, Any]:
        """
        Get summary of a specific document.
        
        Args:
            document_path: Path to the document
            max_chunks: Maximum number of chunks to include
            
        Returns:
            Document summary information
        """
        try:
            # Query for chunks from specific document
            filters = {"source_file": document_path}
            
            results = await self.hierarchical_rag.retrieve(
                query="summary overview key points",
                top_k=max_chunks,
                filters=filters
            )
            
            if not results:
                return {"error": "Document not found"}
            
            # Extract key information
            key_points = []
            for result in results:
                text = result.get('text', '')
                if text and len(text) > 100:
                    key_points.append(text[:150] + "...")
            
            return {
                "document": document_path,
                "key_points": key_points,
                "total_chunks": len(results),
                "summary": "Document summary retrieved successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to get document summary: {str(e)}")
            return {"error": str(e)}
    
    async def search_similar_documents(
        self,
        query: str,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Find similar documents based on query.
        
        Args:
            query: Search query
            threshold: Similarity threshold
            
        Returns:
            List of similar documents with scores
        """
        try:
            results = await self.hierarchical_rag.retrieve(
                query=query,
                top_k=10,
                rerank=True
            )
            
            # Group by document
            documents = {}
            for result in results:
                source = result.get('metadata', {}).get('source_file', 'Unknown')
                score = result.get('score', 0.0)
                
                if score >= threshold:
                    if source not in documents:
                        documents[source] = {
                            'document': source,
                            'max_score': score,
                            'chunks': 0
                        }
                    documents[source]['chunks'] += 1
                    documents[source]['max_score'] = max(
                        documents[source]['max_score'], score
                    )
            
            return sorted(
                documents.values(),
                key=lambda x: x['max_score'],
                reverse=True
            )
            
        except Exception as e:
            logger.error(f"Failed to search similar documents: {str(e)}")
            return []