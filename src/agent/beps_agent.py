"""
BEPS Decision-Making Agent

Main agent class that orchestrates all components for intelligent
query processing and response generation.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json

from config import ProcessingConfig
from vector_store import VectorStore
from .response_router import ResponseRouter, AgentResponse
from .query_classifier import QueryClassification
from .web_search_handler import WebSearchHandler

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for the BEPS agent."""
    enable_web_search: bool = True
    enable_fallback: bool = True
    max_processing_time: float = 30.0
    confidence_threshold: float = 0.6
    enable_caching: bool = True
    cache_ttl: int = 3600


class BEPSAgent:
    """
    Main decision-making agent for BEPS-related queries.
    
    This agent combines RAG retrieval, web search, and intelligent routing
    to provide comprehensive answers about BEPS (Base Erosion and Profit Shifting)
    and related international tax matters.
    """
    
    def __init__(
        self,
        config: ProcessingConfig,
        agent_config: Optional[AgentConfig] = None
    ):
        """
        Initialize the BEPS agent.
        
        Args:
            config: Processing configuration
            agent_config: Agent-specific configuration
        """
        self.config = config
        self.agent_config = agent_config or AgentConfig()
        
        # Initialize components
        self.vector_store = None
        self.response_router = None
        self.web_search_handler = None
        
        # Cache for responses
        self._response_cache = {}
        
        # Statistics
        self._stats = {
            'total_queries': 0,
            'successful_responses': 0,
            'fallback_used': 0,
            'average_confidence': 0.0,
            'average_processing_time': 0.0
        }
        
        logger.info("BEPS Agent initialized")
    
    async def initialize(self):
        """Initialize all agent components."""
        try:
            # Initialize vector store
            self.vector_store = VectorStore(self.config)
            await self.vector_store.initialize()
            
            # Initialize web search handler if enabled
            if self.agent_config.enable_web_search:
                self.web_search_handler = WebSearchHandler()
                await self.web_search_handler.initialize()
            
            # Initialize response router
            self.response_router = ResponseRouter(
                config=self.config,
                vector_store=self.vector_store,
                web_search_handler=self.web_search_handler
            )
            await self.response_router.initialize()
            
            logger.info("BEPS Agent fully initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize BEPS Agent: {str(e)}")
            raise
    
    async def query(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Process a BEPS-related query.
        
        Args:
            question: User's question about BEPS
            context: Additional context (user preferences, history, etc.)
            
        Returns:
            AgentResponse with answer and metadata
        """
        if not self.response_router:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        
        # Check cache
        cache_key = self._generate_cache_key(question, context)
        if self.agent_config.enable_caching and cache_key in self._response_cache:
            logger.info("Returning cached response")
            return self._response_cache[cache_key]
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Update statistics
            self._stats['total_queries'] += 1
            
            # Process query with timeout
            response = await asyncio.wait_for(
                self.response_router.route_query(question, context),
                timeout=self.agent_config.max_processing_time
            )
            
            # Update statistics
            self._stats['successful_responses'] += 1
            self._stats['average_confidence'] = (
                (self._stats['average_confidence'] * (self._stats['successful_responses'] - 1) + 
                 response.confidence_score.overall_score) / 
                self._stats['successful_responses']
            )
            self._stats['average_processing_time'] = (
                (self._stats['average_processing_time'] * (self._stats['successful_responses'] - 1) + 
                 response.processing_time) / 
                self._stats['successful_responses']
            )
            
            if response.metadata.get('fallback_used'):
                self._stats['fallback_used'] += 1
            
            # Cache response
            if self.agent_config.enable_caching:
                self._response_cache[cache_key] = response
            
            logger.info(
                f"Query processed successfully. "
                f"Strategy: {response.strategy}, "
                f"Confidence: {response.confidence_score.overall_score:.2f}, "
                f"Time: {response.processing_time:.2f}s"
            )
            
            return response
            
        except asyncio.TimeoutError:
            logger.error("Query processing timed out")
            return await self._handle_timeout(question, start_time)
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return await self._handle_error(question, str(e), start_time)
    
    async def batch_query(
        self,
        questions: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> List[AgentResponse]:
        """
        Process multiple queries in batch.
        
        Args:
            questions: List of questions to process
            context: Shared context for all queries
            
        Returns:
            List of AgentResponse objects
        """
        logger.info(f"Processing batch of {len(questions)} queries")
        
        # Process queries concurrently
        tasks = [
            self.query(question, context)
            for question in questions
        ]
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        results = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(f"Error processing question {i+1}: {str(response)}")
                error_response = await self._handle_error(
                    questions[i],
                    str(response),
                    asyncio.get_event_loop().time()
                )
                results.append(error_response)
            else:
                results.append(response)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get agent performance statistics."""
        return {
            **self._stats,
            'cache_size': len(self._response_cache),
            'vector_store_ready': self.vector_store is not None,
            'web_search_enabled': self.web_search_handler is not None
        }
    
    def clear_cache(self):
        """Clear the response cache."""
        self._response_cache.clear()
        logger.info("Response cache cleared")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information and capabilities."""
        return {
            'agent_config': {
                'enable_web_search': self.agent_config.enable_web_search,
                'enable_fallback': self.agent_config.enable_fallback,
                'max_processing_time': self.agent_config.max_processing_time,
                'confidence_threshold': self.agent_config.confidence_threshold,
                'enable_caching': self.agent_config.enable_caching,
                'cache_ttl': self.agent_config.cache_ttl
            },
            'vector_store': {
                'initialized': self.vector_store is not None,
                'collection_name': self.config.collection_name if self.vector_store else None
            },
            'capabilities': [
                'RAG-based retrieval from OECD documents',
                'Web search integration',
                'Query classification',
                'Confidence scoring',
                'Intelligent routing',
                'Fallback mechanisms',
                'Response caching'
            ],
            'supported_query_types': [
                'Factual questions',
                'Definition requests',
                'Analytical queries',
                'Procedural questions',
                'Comparative analysis',
                'Temporal queries',
                'General BEPS information'
            ]
        }
    
    def _generate_cache_key(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate cache key for a query."""
        key_data = {
            'question': question.lower().strip(),
            'context': context or {}
        }
        return json.dumps(key_data, sort_keys=True)
    
    async def _handle_timeout(
        self,
        question: str,
        start_time: float
    ) -> AgentResponse:
        """Handle query timeout."""
        processing_time = asyncio.get_event_loop().time() - start_time
        
        from agent.response_router import RoutingDecision
        from agent.confidence_scorer import ConfidenceScore
        
        return AgentResponse(
            answer=(
                "I apologize, but your query took too long to process. "
                "This might be due to complex search requirements or system load. "
                "Please try rephrasing your question or ask a more specific question."
            ),
            strategy="TIMEOUT",
            confidence_score=ConfidenceScore(
                overall_score=0.0,
                factors=None,
                breakdown={},
                recommendations=["Try a more specific question"],
                strategy="TIMEOUT"
            ),
            sources=[],
            routing_decision=RoutingDecision(
                strategy=None,
                confidence_threshold=0.0,
                fallback_strategies=[],
                reasoning="Query timeout",
                metadata={'timeout': True}
            ),
            processing_time=processing_time,
            metadata={'timeout': True}
        )
    
    async def _handle_error(
        self,
        question: str,
        error_message: str,
        start_time: float
    ) -> AgentResponse:
        """Handle general errors."""
        processing_time = asyncio.get_event_loop().time() - start_time
        
        from agent.response_router import RoutingDecision
        from agent.confidence_scorer import ConfidenceScore
        
        return AgentResponse(
            answer=(
                f"I encountered an error processing your query: {error_message}. "
                f"Please try rephrasing your question or contact support if the issue persists."
            ),
            strategy="ERROR",
            confidence_score=ConfidenceScore(
                overall_score=0.0,
                factors=None,
                breakdown={},
                recommendations=["Try rephrasing your question"],
                strategy="ERROR"
            ),
            sources=[],
            routing_decision=RoutingDecision(
                strategy=None,
                confidence_threshold=0.0,
                fallback_strategies=[],
                reasoning=f"Error: {error_message}",
                metadata={'error': True, 'error_message': error_message}
            ),
            processing_time=processing_time,
            metadata={'error': True, 'error_message': error_message}
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all components."""
        health_status = {
            'agent': 'healthy',
            'vector_store': 'unknown',
            'response_router': 'unknown',
            'web_search': 'unknown'
        }
        
        try:
            # Check vector store
            if self.vector_store:
                health_status['vector_store'] = 'healthy'
            else:
                health_status['vector_store'] = 'uninitialized'
            
            # Check response router
            if self.response_router:
                health_status['response_router'] = 'healthy'
            else:
                health_status['response_router'] = 'uninitialized'
            
            # Check web search
            if self.agent_config.enable_web_search:
                if self.web_search_handler:
                    health_status['web_search'] = 'healthy'
                else:
                    health_status['web_search'] = 'uninitialized'
            else:
                health_status['web_search'] = 'disabled'
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                'agent': 'unhealthy',
                'error': str(e)
            }


# Convenience function for quick setup
async def create_beps_agent(
    config_path: Optional[str] = None,
    agent_config: Optional[AgentConfig] = None
) -> BEPSAgent:
    """
    Create and initialize a BEPS agent.
    
    Args:
        config_path: Path to configuration file
        agent_config: Agent configuration
        
    Returns:
        Initialized BEPSAgent instance
    """
    from config import load_config
    
    if config_path:
        config = load_config(config_path)
    else:
        config = ProcessingConfig()
    
    agent = BEPSAgent(config, agent_config)
    await agent.initialize()
    
    return agent