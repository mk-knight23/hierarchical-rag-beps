"""
Response Router for Decision-Making Agent

This module implements intelligent routing logic to determine the best
response strategy based on query classification and confidence scores.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import asyncio

from .query_classifier import QueryClassifier, QueryClassification, ResponseStrategy
from .web_search_handler import WebSearchHandler
from .rag_handler import RAGHandler, RAGResponse
from .confidence_scorer import ConfidenceScorer, ConfidenceScore
from ..config import ProcessingConfig
from ..vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    """Data class for routing decision."""
    strategy: ResponseStrategy
    confidence_threshold: float
    fallback_strategies: List[ResponseStrategy]
    reasoning: str
    metadata: Dict[str, Any]


@dataclass
class AgentResponse:
    """Data class for final agent response."""
    answer: str
    strategy: str
    confidence_score: ConfidenceScore
    sources: List[Dict[str, Any]]
    routing_decision: RoutingDecision
    processing_time: float
    metadata: Dict[str, Any]


class ResponseRouter:
    """
    Intelligent response routing system for optimal strategy selection.
    """
    
    def __init__(
        self,
        config: ProcessingConfig,
        vector_store: VectorStore,
        web_search_handler: Optional[WebSearchHandler] = None
    ):
        """
        Initialize response router.
        
        Args:
            config: Processing configuration
            vector_store: Vector store instance
            web_search_handler: Web search handler instance
        """
        self.config = config
        self.vector_store = vector_store
        
        # Initialize components
        self.query_classifier = QueryClassifier()
        self.confidence_scorer = ConfidenceScorer()
        self.rag_handler = RAGHandler(config, vector_store)
        self.web_search_handler = web_search_handler
        
        # Routing thresholds
        self.routing_config = {
            'min_confidence_threshold': 0.6,
            'max_fallback_attempts': 2,
            'hybrid_threshold': 0.75,
            'web_search_threshold': 0.5
        }
        
        # Update from config
        if hasattr(config, 'agent_config'):
            self.routing_config.update(config.agent_config.get('routing', {}))
    
    async def initialize(self):
        """Initialize all router components."""
        try:
            await self.rag_handler.initialize()
            logger.info("Response router initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize response router: {str(e)}")
            raise
    
    async def route_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Route query to appropriate response strategy.
        
        Args:
            query: User query
            context: Additional context for routing
            
        Returns:
            AgentResponse with final answer and metadata
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Step 1: Classify query
            query_classification = await self.query_classifier.classify(query)
            logger.info(f"Query classified as: {query_classification}")
            
            # Step 2: Make routing decision
            routing_decision = self._make_routing_decision(
                query_classification,
                context
            )
            logger.info(f"Routing decision: {routing_decision}")
            
            # Step 3: Execute strategy
            response_data = await self._execute_strategy(
                query,
                routing_decision,
                context
            )
            
            # Step 4: Calculate confidence
            confidence_score = self.confidence_scorer.calculate_confidence(
                query_classification,
                response_data,
                routing_decision.strategy
            )
            
            # Step 5: Handle fallback if needed
            if confidence_score.overall_score < routing_decision.confidence_threshold:
                response_data = await self._handle_fallback(
                    query,
                    routing_decision,
                    confidence_score,
                    context
                )
                
                # Recalculate confidence after fallback
                confidence_score = self.confidence_scorer.calculate_confidence(
                    query_classification,
                    response_data,
                    routing_decision.strategy
                )
            
            # Step 6: Format final response
            processing_time = asyncio.get_event_loop().time() - start_time
            
            return AgentResponse(
                answer=response_data.get('answer', ''),
                strategy=routing_decision.strategy.value,
                confidence_score=confidence_score,
                sources=response_data.get('sources', []),
                routing_decision=routing_decision,
                processing_time=processing_time,
                metadata={
                    'query_classification': query_classification.to_dict(),
                    'fallback_used': confidence_score.overall_score < routing_decision.confidence_threshold
                }
            )
            
        except Exception as e:
            logger.error(f"Error routing query: {str(e)}")
            return await self._handle_error(query, str(e), start_time)
    
    def _make_routing_decision(
        self,
        query_classification: QueryClassification,
        context: Optional[Dict[str, Any]] = None
    ) -> RoutingDecision:
        """
        Make intelligent routing decision based on query classification.
        
        Args:
            query_classification: Query classification results
            context: Additional context
            
        Returns:
            RoutingDecision with strategy and parameters
        """
        query_type = query_classification.query_type.value
        confidence = query_classification.confidence
        
        # Strategy mapping based on query type
        strategy_mapping = {
            'factual': ResponseStrategy.RAG_RETRIEVAL,
            'definition': ResponseStrategy.RAG_RETRIEVAL,
            'analytical': ResponseStrategy.HYBRID,
            'procedural': ResponseStrategy.RAG_RETRIEVAL,
            'comparative': ResponseStrategy.HYBRID,
            'temporal': ResponseStrategy.WEB_SEARCH,
            'opinion': ResponseStrategy.WEB_SEARCH,
            'general': ResponseStrategy.RAG_RETRIEVAL
        }
        
        # Determine primary strategy
        primary_strategy = strategy_mapping.get(
            query_type,
            ResponseStrategy.RAG_RETRIEVAL
        )
        
        # Adjust based on confidence
        if confidence < 0.5:
            # Low confidence queries use hybrid approach
            primary_strategy = ResponseStrategy.HYBRID
        
        # Determine fallback strategies
        fallback_strategies = self._get_fallback_strategies(primary_strategy)
        
        # Set confidence threshold
        confidence_threshold = self._get_confidence_threshold(
            primary_strategy,
            query_classification
        )
        
        # Generate reasoning
        reasoning = self._generate_routing_reasoning(
            primary_strategy,
            query_classification,
            confidence_threshold
        )
        
        return RoutingDecision(
            strategy=primary_strategy,
            confidence_threshold=confidence_threshold,
            fallback_strategies=fallback_strategies,
            reasoning=reasoning,
            metadata={
                'query_type': query_type,
                'confidence': confidence,
                'domain_specific': query_classification.domain_specific
            }
        )
    
    def _get_fallback_strategies(
        self,
        primary_strategy: ResponseStrategy
    ) -> List[ResponseStrategy]:
        """Get fallback strategies for a given primary strategy."""
        fallback_map = {
            ResponseStrategy.RAG_RETRIEVAL: [
                ResponseStrategy.WEB_SEARCH,
                ResponseStrategy.DIRECT_ANSWER
            ],
            ResponseStrategy.WEB_SEARCH: [
                ResponseStrategy.RAG_RETRIEVAL,
                ResponseStrategy.DIRECT_ANSWER
            ],
            ResponseStrategy.HYBRID: [
                ResponseStrategy.RAG_RETRIEVAL,
                ResponseStrategy.WEB_SEARCH
            ],
            ResponseStrategy.DIRECT_ANSWER: [
                ResponseStrategy.RAG_RETRIEVAL,
                ResponseStrategy.WEB_SEARCH
            ]
        }
        
        return fallback_map.get(primary_strategy, [])
    
    def _get_confidence_threshold(
        self,
        strategy: ResponseStrategy,
        query_classification: QueryClassification
    ) -> float:
        """Get confidence threshold for a strategy."""
        base_threshold = self.routing_config['min_confidence_threshold']
        
        # Adjust based on query type
        type_adjustments = {
            'factual': 0.1,
            'definition': 0.05,
            'analytical': -0.05,
            'procedural': 0.0,
            'comparative': -0.1,
            'temporal': -0.05,
            'opinion': -0.15,
            'general': 0.0
        }
        
        adjustment = type_adjustments.get(
            query_classification.query_type.value,
            0.0
        )
        
        return max(0.4, min(base_threshold + adjustment, 0.9))
    
    def _generate_routing_reasoning(
        self,
        strategy: ResponseStrategy,
        query_classification: QueryClassification,
        threshold: float
    ) -> str:
        """Generate human-readable routing reasoning."""
        query_type = query_classification.query_type.value
        
        reasoning_map = {
            ResponseStrategy.RAG_RETRIEVAL: (
                f"Selected RAG retrieval for {query_type} query based on "
                f"processed OECD documents with confidence threshold {threshold:.2f}"
            ),
            ResponseStrategy.WEB_SEARCH: (
                f"Selected web search for {query_type} query to access "
                f"latest information with confidence threshold {threshold:.2f}"
            ),
            ResponseStrategy.HYBRID: (
                f"Selected hybrid approach for {query_type} query to combine "
                f"RAG and web search results with confidence threshold {threshold:.2f}"
            ),
            ResponseStrategy.DIRECT_ANSWER: (
                f"Selected direct answer for simple {query_type} query "
                f"with confidence threshold {threshold:.2f}"
            )
        }
        
        return reasoning_map.get(strategy, "Default routing decision")
    
    async def _execute_strategy(
        self,
        query: str,
        routing_decision: RoutingDecision,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute the chosen response strategy."""
        strategy = routing_decision.strategy
        
        try:
            if strategy == ResponseStrategy.RAG_RETRIEVAL:
                return await self._execute_rag_strategy(query, context)
            
            elif strategy == ResponseStrategy.WEB_SEARCH:
                return await self._execute_web_strategy(query, context)
            
            elif strategy == ResponseStrategy.HYBRID:
                return await self._execute_hybrid_strategy(query, context)
            
            elif strategy == ResponseStrategy.DIRECT_ANSWER:
                return await self._execute_direct_strategy(query, context)
            
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
                
        except Exception as e:
            logger.error(f"Error executing strategy {strategy}: {str(e)}")
            raise
    
    async def _execute_rag_strategy(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute RAG retrieval strategy."""
        rag_response = await self.rag_handler.generate_response(query)
        
        return {
            'answer': rag_response.answer,
            'sources': rag_response.sources,
            'confidence': rag_response.confidence,
            'metadata': {
                'chunks_used': rag_response.chunks_used,
                'retrieval_time': rag_response.retrieval_time
            }
        }
    
    async def _execute_web_strategy(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute web search strategy."""
        if not self.web_search_handler:
            raise ValueError("Web search handler not available")
        
        search_results = await self.web_search_handler.search(query)
        
        # Format web search results
        if search_results:
            answer = self.web_search_handler.format_results(search_results)
            sources = [
                {
                    'title': result.get('title', ''),
                    'url': result.get('url', ''),
                    'source': result.get('source', '')
                }
                for result in search_results
            ]
        else:
            answer = "No relevant web search results found."
            sources = []
        
        return {
            'answer': answer,
            'sources': sources,
            'results': search_results
        }
    
    async def _execute_hybrid_strategy(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute hybrid strategy combining RAG and web search."""
        # Execute both strategies in parallel
        rag_task = self._execute_rag_strategy(query, context)
        web_task = self._execute_web_strategy(query, context)
        
        rag_result, web_result = await asyncio.gather(
            rag_task,
            web_task,
            return_exceptions=True
        )
        
        # Handle exceptions
        if isinstance(rag_result, Exception):
            logger.error(f"RAG strategy failed: {str(rag_result)}")
            rag_result = {'answer': '', 'sources': [], 'confidence': 0.0}
        
        if isinstance(web_result, Exception):
            logger.error(f"Web strategy failed: {str(web_result)}")
            web_result = {'answer': '', 'sources': [], 'results': []}
        
        # Combine results
        combined_answer = self._combine_hybrid_answers(
            rag_result.get('answer', ''),
            web_result.get('answer', '')
        )
        
        combined_sources = (
            rag_result.get('sources', []) + 
            web_result.get('sources', [])
        )
        
        return {
            'answer': combined_answer,
            'sources': combined_sources,
            'rag_confidence': rag_result.get('confidence', 0.0),
            'web_confidence': 0.7 if web_result.get('results') else 0.0
        }
    
    async def _execute_direct_strategy(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute direct answer strategy for simple queries."""
        # Simple direct answers for common questions
        direct_answers = {
            'what is beps': 'BEPS (Base Erosion and Profit Shifting) refers to tax planning strategies that exploit gaps and mismatches in tax rules to artificially shift profits to low or no-tax locations.',
            'what are beps actions': 'The BEPS project consists of 15 Actions that address various aspects of international tax planning, including digital economy challenges, treaty abuse, transfer pricing, and more.',
            'what is pillar one': 'Pillar One addresses the allocation of taxing rights for large multinational enterprises, ensuring fairer distribution of profits and taxing rights among countries.',
            'what is pillar two': 'Pillar Two introduces a global minimum corporate tax rate of 15% to ensure large multinational enterprises pay a minimum level of tax regardless of where they operate.'
        }
        
        query_lower = query.lower().strip()
        
        # Find best matching direct answer
        best_match = None
        best_score = 0.0
        
        for key, answer in direct_answers.items():
            if key in query_lower:
                score = len(key) / len(query_lower)
                if score > best_score:
                    best_score = score
                    best_match = answer
        
        if best_match:
            return {
                'answer': best_match,
                'sources': [],
                'confidence': 0.9
            }
        else:
            # Fallback to RAG
            return await self._execute_rag_strategy(query, context)
    
    def _combine_hybrid_answers(
        self,
        rag_answer: str,
        web_answer: str
    ) -> str:
        """Combine RAG and web search answers for hybrid strategy."""
        if not rag_answer and not web_answer:
            return "No relevant information found."
        
        if not rag_answer:
            return web_answer
        
        if not web_answer:
            return rag_answer
        
        # Combine both answers
        combined = f"""Based on processed OECD documents and recent web sources:

{rag_answer}

Recent updates and additional context:
{web_answer}

This combined response incorporates both established BEPS framework information and the latest developments."""
        
        return combined
    
    async def _handle_fallback(
        self,
        query: str,
        routing_decision: RoutingDecision,
        confidence_score: ConfidenceScore,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle fallback to alternative strategies."""
        logger.info("Initiating fallback strategies")
        
        for fallback_strategy in routing_decision.fallback_strategies:
            try:
                logger.info(f"Trying fallback strategy: {fallback_strategy}")
                
                # Create new routing decision for fallback
                fallback_decision = RoutingDecision(
                    strategy=fallback_strategy,
                    confidence_threshold=routing_decision.confidence_threshold * 0.8,
                    fallback_strategies=[],
                    reasoning=f"Fallback to {fallback_strategy.value}",
                    metadata={'fallback': True}
                )
                
                response_data = await self._execute_strategy(
                    query,
                    fallback_decision,
                    context
                )
                
                # Check if fallback is better
                new_confidence = self.confidence_scorer.calculate_confidence(
                    QueryClassification(
                        query_type=QueryClassification.QueryType.GENERAL,
                        confidence=0.5,
                        keywords=[],
                        domain_specific=False
                    ),
                    response_data,
                    fallback_strategy
                )
                
                if new_confidence.overall_score >= fallback_decision.confidence_threshold:
                    logger.info(f"Fallback strategy {fallback_strategy} succeeded")
                    return response_data
                    
            except Exception as e:
                logger.error(f"Fallback strategy {fallback_strategy} failed: {str(e)}")
                continue
        
        # Final fallback - direct answer
        logger.info("Using final fallback - direct answer")
        return await self._execute_direct_strategy(query, context)
    
    async def _handle_error(
        self,
        query: str,
        error_message: str,
        start_time: float
    ) -> AgentResponse:
        """Handle errors gracefully."""
        processing_time = asyncio.get_event_loop().time() - start_time
        
        return AgentResponse(
            answer=f"I encountered an error processing your query: {error_message}. Please try rephrasing your question.",
            strategy="ERROR",
            confidence_score=ConfidenceScore(
                overall_score=0.0,
                factors=None,
                breakdown={},
                recommendations=["Please try rephrasing your query"],
                strategy="ERROR"
            ),
            sources=[],
            routing_decision=RoutingDecision(
                strategy=ResponseStrategy.DIRECT_ANSWER,
                confidence_threshold=0.0,
                fallback_strategies=[],
                reasoning="Error handling",
                metadata={'error': error_message}
            ),
            processing_time=processing_time,
            metadata={'error': True, 'error_message': error_message}
        )