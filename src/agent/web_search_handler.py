"""
Web Search Handler for Decision-Making Agent

This module provides web search capabilities to fetch real-time information
when RAG-based retrieval is insufficient or when temporal information is needed.
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json
from datetime import datetime
import re

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Data class for web search results."""
    title: str
    url: str
    snippet: str
    source: str
    date: Optional[str] = None
    relevance_score: float = 0.0


@dataclass
class WebSearchResponse:
    """Data class for web search response."""
    results: List[SearchResult]
    query: str
    total_results: int
    search_time: float
    sources: List[str]


class WebSearchHandler:
    """
    Web search handler for fetching real-time information from various sources.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize web search handler.
        
        Args:
            config: Configuration dictionary with API keys and settings
        """
        self.config = config or {}
        self.timeout = self.config.get('timeout', 10)
        self.max_results = self.config.get('max_results', 5)
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
            
    async def search(self, query: str, max_results: int = None) -> WebSearchResponse:
        """
        Perform web search for the given query.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            WebSearchResponse with search results
        """
        max_results = max_results or self.max_results
        start_time = asyncio.get_event_loop().time()
        
        try:
            # For demonstration, we'll use a mock search
            # In production, integrate with real APIs like Google Custom Search, Bing, etc.
            results = await self._mock_search(query, max_results)
            
            search_time = asyncio.get_event_loop().time() - start_time
            
            return WebSearchResponse(
                results=results,
                query=query,
                total_results=len(results),
                search_time=search_time,
                sources=list(set([r.source for r in results]))
            )
            
        except Exception as e:
            logger.error(f"Web search failed: {str(e)}")
            return WebSearchResponse(
                results=[],
                query=query,
                total_results=0,
                search_time=asyncio.get_event_loop().time() - start_time,
                sources=[]
            )
    
    async def _mock_search(self, query: str, max_results: int) -> List[SearchResult]:
        """
        Mock search implementation for demonstration.
        Replace with actual API integrations.
        """
        # Mock data for OECD BEPS related queries
        mock_results = {
            "pillar two minimum tax": [
                SearchResult(
                    title="OECD Pillar Two Model Rules - Global Minimum Tax",
                    url="https://www.oecd.org/tax/beps/pillar-two-model-rules.pdf",
                    snippet="The Pillar Two model rules provide for a global minimum tax rate of 15% for multinational enterprises...",
                    source="OECD",
                    date="2023-12-15",
                    relevance_score=0.95
                ),
                SearchResult(
                    title="Global Anti-Base Erosion Rules (GloBE) - Implementation",
                    url="https://taxfoundation.org/pillar-two-globe-rules/",
                    snippet="Analysis of the Global Anti-Base Erosion Rules implementation across different jurisdictions...",
                    source="Tax Foundation",
                    date="2024-01-10",
                    relevance_score=0.88
                ),
                SearchResult(
                    title="Singapore's Approach to Pillar Two Implementation",
                    url="https://www.iras.gov.sg/pillar-two",
                    snippet="Singapore's implementation plan for the global minimum tax rules under Pillar Two...",
                    source="IRAS",
                    date="2024-02-01",
                    relevance_score=0.85
                )
            ],
            "beps action 5 substance requirements": [
                SearchResult(
                    title="BEPS Action 5: Agreement on Modified Nexus Approach",
                    url="https://www.oecd.org/tax/beps-action-5-substance-requirements.pdf",
                    snippet="The modified nexus approach for substantial activities requirements under BEPS Action 5...",
                    source="OECD",
                    date="2023-11-20",
                    relevance_score=0.92
                ),
                SearchResult(
                    title="Economic Substance Requirements - Practical Guide",
                    url="https://www2.deloitte.com/substance-requirements",
                    snippet="Comprehensive guide to implementing economic substance requirements under BEPS Action 5...",
                    source="Deloitte",
                    date="2024-01-15",
                    relevance_score=0.87
                )
            ],
            "digital services tax": [
                SearchResult(
                    title="Global Digital Services Tax Developments",
                    url="https://home.kpmg/xx/en/home/insights/2024/01/digital-services-tax.html",
                    snippet="Latest developments in digital services tax implementation across jurisdictions...",
                    source="KPMG",
                    date="2024-01-25",
                    relevance_score=0.90
                ),
                SearchResult(
                    title="OECD Progress Report on Digital Taxation",
                    url="https://www.oecd.org/tax/beps/inclusive-framework-on-beps-progress-report-january-2024.pdf",
                    snippet="Progress report on the implementation of the two-pillar solution for digital taxation...",
                    source="OECD",
                    date="2024-01-20",
                    relevance_score=0.93
                )
            ]
        }
        
        # Find relevant mock results
        query_lower = query.lower()
        results = []
        
        for key, value in mock_results.items():
            if any(term in query_lower for term in key.split()):
                results.extend(value)
        
        # If no specific results, generate generic ones
        if not results:
            results = [
                SearchResult(
                    title=f"Latest information on {query}",
                    url=f"https://search.example.com/{query.replace(' ', '-')}",
                    snippet=f"Current information and developments regarding {query}...",
                    source="Search Engine",
                    date=datetime.now().strftime("%Y-%m-%d"),
                    relevance_score=0.7
                )
            ]
        
        return results[:max_results]
    
    async def search_with_fallback(self, query: str, max_results: int = None) -> WebSearchResponse:
        """
        Search with fallback mechanisms for reliability.
        
        Args:
            query: Search query string
            max_results: Maximum number of results
            
        Returns:
            WebSearchResponse with search results
        """
        try:
            # Try primary search
            return await self.search(query, max_results)
        except Exception as e:
            logger.warning(f"Primary search failed, using fallback: {str(e)}")
            
            # Fallback to basic search
            return WebSearchResponse(
                results=await self._basic_fallback_search(query, max_results),
                query=query,
                total_results=0,
                search_time=0.0,
                sources=["fallback"]
            )
    
    async def _basic_fallback_search(self, query: str, max_results: int) -> List[SearchResult]:
        """
        Basic fallback search when primary search fails.
        """
        return [
            SearchResult(
                title=f"Search results for: {query}",
                url="https://example.com/search",
                snippet="Search service temporarily unavailable. Please try again later.",
                source="fallback",
                relevance_score=0.0
            )
        ]
    
    def format_results(self, search_response: WebSearchResponse) -> str:
        """
        Format search results for presentation.
        
        Args:
            search_response: Web search response
            
        Returns:
            Formatted string of search results
        """
        if not search_response.results:
            return "No relevant search results found."
        
        formatted = f"Found {search_response.total_results} results for '{search_response.query}':\n\n"
        
        for i, result in enumerate(search_response.results, 1):
            formatted += f"{i}. **{result.title}**\n"
            formatted += f"   Source: {result.source}"
            if result.date:
                formatted += f" ({result.date})"
            formatted += f"\n   {result.snippet}\n"
            formatted += f"   URL: {result.url}\n\n"
        
        return formatted
    
    async def search_and_summarize(self, query: str, max_results: int = 3) -> Dict[str, Any]:
        """
        Search and provide a summarized response.
        
        Args:
            query: Search query string
            max_results: Maximum number of results
            
        Returns:
            Dictionary with summary and detailed results
        """
        search_response = await self.search(query, max_results)
        
        if not search_response.results:
            return {
                "summary": "No relevant information found.",
                "results": [],
                "sources": []
            }
        
        # Generate summary from top results
        summary = self._generate_summary(search_response.results)
        
        return {
            "summary": summary,
            "results": [
                {
                    "title": r.title,
                    "url": r.url,
                    "snippet": r.snippet,
                    "source": r.source,
                    "date": r.date
                }
                for r in search_response.results
            ],
            "sources": search_response.sources
        }
    
    def _generate_summary(self, results: List[SearchResult]) -> str:
        """Generate a summary from search results."""
        if not results:
            return "No information available."
        
        # Use top 3 results for summary
        top_results = results[:3]
        
        summary = "Based on the latest available information:\n\n"
        
        for result in top_results:
            summary += f"- {result.snippet}\n"
        
        summary += f"\nSources: {', '.join(set([r.source for r in top_results]))}"
        
        return summary