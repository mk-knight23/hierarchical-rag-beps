"""
Comprehensive tests for the BEPS Decision-Making Agent
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import json

from src.agent.beps_agent import BEPSAgent, AgentConfig, create_beps_agent
from src.agent.query_classifier import QueryClassification, ResponseStrategy
from src.agent.confidence_scorer import ConfidenceScore
from src.config import ProcessingConfig


class TestBEPSAgent:
    """Test suite for BEPS Agent."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = ProcessingConfig()
        config.collection_name = "test_collection"
        return config
    
    @pytest.fixture
    def mock_agent_config(self):
        """Create mock agent configuration."""
        return AgentConfig(
            enable_web_search=False,
            enable_caching=False,
            max_processing_time=10.0
        )
    
    @pytest.fixture
    async def agent(self, mock_config, mock_agent_config):
        """Create initialized agent for testing."""
        agent = BEPSAgent(mock_config, mock_agent_config)
        
        # Mock vector store
        agent.vector_store = Mock()
        agent.vector_store.initialize = AsyncMock()
        
        # Mock response router
        agent.response_router = Mock()
        agent.response_router.initialize = AsyncMock()
        
        yield agent
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, mock_config):
        """Test agent initialization."""
        agent = BEPSAgent(mock_config)
        
        with patch('src.agent.beps_agent.VectorStore') as mock_vs:
            mock_vs_instance = Mock()
            mock_vs_instance.initialize = AsyncMock()
            mock_vs.return_value = mock_vs_instance
            
            await agent.initialize()
            
            assert agent.vector_store is not None
            assert agent.response_router is not None
    
    @pytest.mark.asyncio
    async def test_query_processing(self, agent):
        """Test basic query processing."""
        mock_response = Mock()
        mock_response.answer = "Test answer"
        mock_response.strategy = "RAG_RETRIEVAL"
        mock_response.confidence_score = Mock()
        mock_response.confidence_score.overall_score = 0.85
        mock_response.sources = []
        mock_response.routing_decision = Mock()
        mock_response.processing_time = 1.5
        mock_response.metadata = {}
        
        agent.response_router.route_query = AsyncMock(return_value=mock_response)
        
        response = await agent.query("What is BEPS?")
        
        assert response.answer == "Test answer"
        assert response.strategy == "RAG_RETRIEVAL"
        assert response.confidence_score.overall_score == 0.85
    
    @pytest.mark.asyncio
    async def test_batch_query_processing(self, agent):
        """Test batch query processing."""
        mock_responses = [
            Mock(answer=f"Answer {i}", strategy="RAG_RETRIEVAL")
            for i in range(3)
        ]
        
        agent.response_router.route_query = AsyncMock(
            side_effect=mock_responses
        )
        
        questions = ["Query 1", "Query 2", "Query 3"]
        responses = await agent.batch_query(questions)
        
        assert len(responses) == 3
        assert responses[0].answer == "Answer 0"
        assert responses[1].answer == "Answer 1"
        assert responses[2].answer == "Answer 2"
    
    @pytest.mark.asyncio
    async def test_query_timeout(self, agent):
        """Test query timeout handling."""
        async def slow_response():
            await asyncio.sleep(2)
            return Mock()
        
        agent.response_router.route_query = slow_response
        
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                agent.query("Slow query"),
                timeout=0.1
            )
    
    def test_statistics(self, agent):
        """Test statistics collection."""
        agent._stats = {
            'total_queries': 10,
            'successful_responses': 8,
            'fallback_used': 2,
            'average_confidence': 0.75,
            'average_processing_time': 2.5
        }
        
        stats = agent.get_statistics()
        
        assert stats['total_queries'] == 10
        assert stats['successful_responses'] == 8
        assert stats['fallback_used'] == 2
        assert stats['average_confidence'] == 0.75
    
    def test_system_info(self, agent):
        """Test system information."""
        info = agent.get_system_info()
        
        assert 'agent_config' in info
        assert 'capabilities' in info
        assert 'supported_query_types' in info
        assert info['agent_config']['enable_web_search'] is False
    
    def test_cache_operations(self, agent):
        """Test cache operations."""
        agent.agent_config.enable_caching = True
        
        # Test cache key generation
        key1 = agent._generate_cache_key("test query")
        key2 = agent._generate_cache_key("test query", {"context": "test"})
        
        assert key1 != key2
        
        # Test cache clearing
        agent._response_cache = {"test": "data"}
        agent.clear_cache()
        assert len(agent._response_cache) == 0
    
    @pytest.mark.asyncio
    async def test_health_check(self, agent):
        """Test health check."""
        agent.vector_store = Mock()
        agent.response_router = Mock()
        
        health = await agent.health_check()
        
        assert health['agent'] == 'healthy'
        assert health['vector_store'] == 'healthy'
        assert health['response_router'] == 'healthy'
    
    @pytest.mark.asyncio
    async def test_create_beps_agent(self):
        """Test convenience function for agent creation."""
        with patch('src.agent.beps_agent.load_config') as mock_load:
            mock_config = ProcessingConfig()
            mock_load.return_value = mock_config
            
            with patch('src.agent.beps_agent.BEPSAgent') as mock_agent_class:
                mock_agent = Mock()
                mock_agent.initialize = AsyncMock()
                mock_agent_class.return_value = mock_agent
                
                agent = await create_beps_agent("test_config.yaml")
                
                assert agent is not None
                mock_agent.initialize.assert_called_once()


class TestQueryClassifier:
    """Test suite for Query Classifier."""
    
    @pytest.fixture
    def classifier(self):
        """Create query classifier."""
        from src.agent.query_classifier import QueryClassifier
        return QueryClassifier()
    
    def test_factual_query_classification(self, classifier):
        """Test factual query classification."""
        query = "What is the minimum tax rate under Pillar Two?"
        classification = classifier.classify_query(query)
        
        assert classification.query_type.value == "factual"
        assert classification.domain_specific is True
    
    def test_analytical_query_classification(self, classifier):
        """Test analytical query classification."""
        query = "How does Pillar One affect digital services taxation?"
        classification = classifier.classify_query(query)
        
        assert classification.query_type.value == "analytical"
    
    @pytest.mark.asyncio
    async def test_temporal_query_classification(self, classifier):
        """Test temporal query classification."""
        query = "What are the latest BEPS developments in 2024?"
        classification = await classifier.classify(query)
        
        assert classification.query_type.value == "temporal"
    
    def test_strategy_mapping(self):
        """Test strategy mapping."""
        from src.agent.query_classifier import ResponseStrategy
        
        assert ResponseStrategy.RAG_RETRIEVAL.value == "rag_retrieval"
        assert ResponseStrategy.WEB_SEARCH.value == "web_search"
        assert ResponseStrategy.HYBRID.value == "hybrid"
        assert ResponseStrategy.DIRECT_ANSWER.value == "direct_answer"


class TestConfidenceScorer:
    """Test suite for Confidence Scorer."""
    
    @pytest.fixture
    def scorer(self):
        """Create confidence scorer."""
        from src.agent.confidence_scorer import ConfidenceScorer
        return ConfidenceScorer()
    
    @pytest.fixture
    def mock_classification(self):
        """Create mock query classification."""
        return QueryClassification(
            query_type=QueryType.FACTUAL,
            confidence=0.9,
            keywords=['beps', 'pillar', 'tax'],
            domain_specific=True
        )
    
    def test_confidence_calculation(self, scorer, mock_classification):
        """Test confidence score calculation."""
        response_data = {
            'answer': 'Test answer with sufficient detail',
            'sources': [{'source': 'oecd', 'score': 0.9}],
            'confidence': 0.85
        }
        
        score = scorer.calculate_confidence(
            mock_classification,
            response_data,
            ResponseStrategy.RAG_RETRIEVAL
        )
        
        assert 0.0 <= score.overall_score <= 1.0
        assert len(score.recommendations) > 0
    
    def test_confidence_levels(self, scorer):
        """Test confidence level mapping."""
        assert scorer.get_confidence_level(0.95) == "Very High"
        assert scorer.get_confidence_level(0.85) == "High"
        assert scorer.get_confidence_level(0.75) == "Medium-High"
        assert scorer.get_confidence_level(0.65) == "Medium"
        assert scorer.get_confidence_level(0.55) == "Medium-Low"
        assert scorer.get_confidence_level(0.45) == "Low"
        assert scorer.get_confidence_level(0.35) == "Very Low"
    
    def test_query_refinement_suggestion(self, scorer):
        """Test query refinement suggestion."""
        from src.agent.confidence_scorer import ConfidenceFactors
        
        factors = ConfidenceFactors(
            query_confidence=0.4,
            retrieval_confidence=0.3,
            source_reliability=0.5,
            answer_completeness=0.6,
            temporal_relevance=0.7,
            domain_alignment=0.5,
            consistency_score=0.8
        )
        
        score = ConfidenceScore(
            overall_score=0.5,
            factors=factors,
            breakdown={},
            recommendations=[],
            strategy="test"
        )
        
        assert scorer.should_refine_query(score) is True


class TestWebSearchHandler:
    """Test suite for Web Search Handler."""
    
    @pytest.fixture
    def handler(self):
        """Create web search handler."""
        from src.agent.web_search_handler import WebSearchHandler
        return WebSearchHandler()
    
    @pytest.mark.asyncio
    async def test_mock_search(self, handler):
        """Test mock search functionality."""
        async with handler:
            results = await handler.search("pillar two minimum tax")
            
            assert results.total_results > 0
            assert len(results.results) > 0
            assert results.query == "pillar two minimum tax"
    
    @pytest.mark.asyncio
    async def test_search_with_fallback(self, handler):
        """Test search with fallback."""
        async with handler:
            response = await handler.search_with_fallback("test query")
            
            assert response is not None
            assert response.query == "test query"
    
    def test_result_formatting(self, handler):
        """Test result formatting."""
        from src.agent.web_search_handler import WebSearchResponse, SearchResult
        
        response = WebSearchResponse(
            results=[
                SearchResult(
                    title="Test Title",
                    url="http://test.com",
                    snippet="Test snippet",
                    source="Test Source"
                )
            ],
            query="test",
            total_results=1,
            search_time=1.0,
            sources=["Test Source"]
        )
        
        formatted = handler.format_results(response)
        assert "Test Title" in formatted
        assert "Test Source" in formatted


class TestIntegration:
    """Integration tests for the complete agent."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_query(self):
        """Test complete query processing pipeline."""
        config = ProcessingConfig()
        agent_config = AgentConfig(
            enable_web_search=False,
            enable_caching=False
        )
        
        agent = BEPSAgent(config, agent_config)
        
        # Mock all dependencies
        with patch('src.agent.beps_agent.VectorStore') as mock_vs:
            mock_vs_instance = Mock()
            mock_vs_instance.initialize = AsyncMock()
            mock_vs.return_value = mock_vs_instance
            
            with patch('src.agent.beps_agent.ResponseRouter') as mock_router:
                mock_router_instance = Mock()
                mock_router_instance.initialize = AsyncMock()
                
                mock_response = Mock()
                mock_response.answer = "BEPS stands for Base Erosion and Profit Shifting"
                mock_response.strategy = "RAG_RETRIEVAL"
                mock_response.confidence_score = Mock()
                mock_response.confidence_score.overall_score = 0.9
                mock_response.sources = []
                mock_response.routing_decision = Mock()
                mock_response.processing_time = 1.2
                mock_response.metadata = {}
                
                mock_router_instance.route_query = AsyncMock(return_value=mock_response)
                mock_router.return_value = mock_router_instance
                
                await agent.initialize()
                
                response = await agent.query("What is BEPS?")
                
                assert response.answer == "BEPS stands for Base Erosion and Profit Shifting"
                assert response.strategy == "RAG_RETRIEVAL"
                assert response.confidence_score.overall_score == 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])