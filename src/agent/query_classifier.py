"""
Query Classification System for Decision-Making Agent

This module provides intelligent classification of user queries to determine
the optimal response strategy: RAG-based retrieval, direct answering, or web search.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Enumeration of query types for routing decisions."""
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    PROCEDURAL = "procedural"
    DEFINITION = "definition"
    COMPARATIVE = "comparative"
    TEMPORAL = "temporal"
    OPINION = "opinion"
    GENERAL = "general"


class ResponseStrategy(Enum):
    """Enumeration of response strategies."""
    RAG_RETRIEVAL = "rag_retrieval"
    DIRECT_ANSWER = "direct_answer"
    WEB_SEARCH = "web_search"
    HYBRID = "hybrid"


@dataclass
class QueryClassification:
    """Data class for query classification results."""
    query_type: QueryType
    confidence: float
    keywords: List[str]
    entities: List[str]
    temporal_indicators: List[str]
    domain_specific: bool
    complexity_score: float


class QueryClassifier:
    """
    Intelligent query classifier that determines query type and optimal response strategy.
    """
    
    def __init__(self):
        """Initialize the query classifier with patterns and rules."""
        self._initialize_patterns()
        self._initialize_tax_keywords()
        
    def _initialize_patterns(self):
        """Initialize regex patterns for query analysis."""
        # Temporal patterns
        self.temporal_patterns = [
            r'\b(when|what year|what date|timeline|schedule|deadline|period)\b',
            r'\b(20\d{2}|19\d{2})\b',  # Year patterns
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',
            r'\b(q[1-4]|quarter|annual|monthly|quarterly)\b'
        ]
        
        # Comparative patterns
        self.comparative_patterns = [
            r'\b(compare|versus|vs|difference|similar|better|worse|more|less)\b',
            r'\b(how does.*compare|what.*difference)\b'
        ]
        
        # Definition patterns
        self.definition_patterns = [
            r'\b(what is|what are|define|definition|meaning|explain)\b',
            r'\b(how.*defined|what.*mean)\b'
        ]
        
        # Procedural patterns
        self.procedural_patterns = [
            r'\b(how to|how do|steps|process|procedure|implement|apply|calculate)\b',
            r'\b(requirement|needed|necessary|must|should)\b'
        ]
        
        # Analytical patterns
        self.analytical_patterns = [
            r'\b(why|impact|effect|consequence|analysis|evaluate|assess)\b',
            r'\b(advantage|disadvantage|benefit|risk|implication)\b'
        ]
        
        # Factual patterns
        self.factual_patterns = [
            r'\b(what|which|who|where|how much|how many)\b',
            r'\b(is|are|was|were|has|have)\b'
        ]
        
    def _initialize_tax_keywords(self):
        """Initialize OECD BEPS and tax-specific keywords."""
        self.tax_keywords = {
            'pillar_one': ['pillar one', 'p1', 'amount a', 'amount b', 'nexus', 'profit allocation'],
            'pillar_two': ['pillar two', 'p2', 'globe', 'minimum tax', '15%', 'qdart', 'substance carve-out'],
            'beps': ['beps', 'base erosion', 'profit shifting', 'oecd', 'g20', 'inclusive framework'],
            'digital_tax': ['digital tax', 'digital services tax', 'dst', 'significant economic presence'],
            'transfer_pricing': ['transfer pricing', 'arm\'s length', 'related party', 'intercompany'],
            'substance': ['substance requirements', 'economic substance', 'substantial activities'],
            'withholding': ['withholding tax', 'wht', 'royalty', 'interest', 'dividend'],
            'treaty': ['tax treaty', 'double taxation', 'treaty benefits', 'mfn', 'most favored nation']
        }
        
    def classify_query(self, query: str) -> QueryClassification:
        """
        Classify a query to determine its type and characteristics.
        
        Args:
            query: The user's query string
            
        Returns:
            QueryClassification object with detailed analysis
        """
        query_lower = query.lower().strip()
        
        # Extract keywords and entities
        keywords = self._extract_keywords(query_lower)
        entities = self._extract_entities(query_lower)
        temporal_indicators = self._extract_temporal_indicators(query_lower)
        
        # Determine query type
        query_type = self._determine_query_type(query_lower, keywords)
        
        # Calculate complexity score
        complexity_score = self._calculate_complexity(query_lower, keywords)
        
        # Check if domain-specific
        domain_specific = self._is_domain_specific(query_lower)
        
        # Calculate confidence
        confidence = self._calculate_confidence(query_lower, query_type, keywords)
        
        return QueryClassification(
            query_type=query_type,
            confidence=confidence,
            keywords=keywords,
            entities=entities,
            temporal_indicators=temporal_indicators,
            domain_specific=domain_specific,
            complexity_score=complexity_score
        )
    
    def determine_strategy(self, classification: QueryClassification) -> ResponseStrategy:
        """
        Determine the optimal response strategy based on query classification.
        
        Args:
            classification: The query classification result
            
        Returns:
            ResponseStrategy enum value
        """
        # High confidence domain-specific queries -> RAG
        if classification.domain_specific and classification.confidence > 0.8:
            return ResponseStrategy.RAG_RETRIEVAL
            
        # Complex analytical queries -> RAG
        if classification.query_type == QueryType.ANALYTICAL and classification.complexity_score > 0.7:
            return ResponseStrategy.RAG_RETRIEVAL
            
        # Simple factual queries with low complexity -> Direct answer
        if classification.query_type == QueryType.FACTUAL and classification.complexity_score < 0.4:
            return ResponseStrategy.DIRECT_ANSWER
            
        # Temporal queries requiring latest info -> Web search
        if classification.temporal_indicators and classification.confidence < 0.6:
            return ResponseStrategy.WEB_SEARCH
            
        # Comparative queries -> Hybrid approach
        if classification.query_type == QueryType.COMPARATIVE:
            return ResponseStrategy.HYBRID
            
        # Default fallback based on confidence
        if classification.confidence > 0.7:
            return ResponseStrategy.RAG_RETRIEVAL
        elif classification.confidence > 0.4:
            return ResponseStrategy.HYBRID
        else:
            return ResponseStrategy.WEB_SEARCH
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract relevant keywords from the query."""
        keywords = []
        
        # Check tax-specific keywords
        for category, kw_list in self.tax_keywords.items():
            for kw in kw_list:
                if kw in query:
                    keywords.append(kw)
        
        # Extract general keywords (nouns and important terms)
        words = re.findall(r'\b[a-z]{3,}\b', query)
        keywords.extend([w for w in words if w not in {'the', 'and', 'for', 'are', 'how', 'what', 'when', 'where', 'why', 'who'}])
        
        return list(set(keywords))
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities from the query."""
        entities = []
        
        # Country names
        countries = ['singapore', 'united states', 'uk', 'germany', 'france', 'japan', 'australia', 'canada']
        for country in countries:
            if country in query:
                entities.append(country)
        
        # Organization names
        orgs = ['oecd', 'g20', 'eu', 'irs', 'hmrc', 'ato']
        for org in orgs:
            if org in query:
                entities.append(org)
        
        # Percentages and amounts
        amounts = re.findall(r'\d+(?:\.\d+)?%', query)
        entities.extend(amounts)
        
        # Year patterns
        years = re.findall(r'\b20\d{2}\b|\b19\d{2}\b', query)
        entities.extend(years)
        
        return entities
    
    def _extract_temporal_indicators(self, query: str) -> List[str]:
        """Extract temporal indicators from the query."""
        indicators = []
        for pattern in self.temporal_patterns:
            matches = re.findall(pattern, query)
            indicators.extend(matches)
        return indicators
    
    def _determine_query_type(self, query: str, keywords: List[str]) -> QueryType:
        """Determine the query type based on patterns and keywords."""
        # Check each pattern type
        for pattern in self.definition_patterns:
            if re.search(pattern, query):
                return QueryType.DEFINITION
                
        for pattern in self.comparative_patterns:
            if re.search(pattern, query):
                return QueryType.COMPARATIVE
                
        for pattern in self.procedural_patterns:
            if re.search(pattern, query):
                return QueryType.PROCEDURAL
                
        for pattern in self.analytical_patterns:
            if re.search(pattern, query):
                return QueryType.ANALYTICAL
                
        for pattern in self.temporal_patterns:
            if re.search(pattern, query):
                return QueryType.TEMPORAL
                
        for pattern in self.factual_patterns:
            if re.search(pattern, query):
                return QueryType.FACTUAL
                
        return QueryType.GENERAL
    
    def _calculate_complexity(self, query: str, keywords: List[str]) -> float:
        """Calculate query complexity score (0-1)."""
        complexity_factors = 0
        
        # Length factor
        word_count = len(query.split())
        if word_count > 15:
            complexity_factors += 0.3
        elif word_count > 10:
            complexity_factors += 0.2
            
        # Keyword diversity
        if len(keywords) > 5:
            complexity_factors += 0.2
            
        # Question complexity
        if 'why' in query or 'how' in query:
            complexity_factors += 0.3
            
        # Multiple concepts
        concept_indicators = ['and', 'or', 'but', 'compared to', 'versus']
        for indicator in concept_indicators:
            if indicator in query:
                complexity_factors += 0.1
                
        return min(complexity_factors, 1.0)
    
    def _is_domain_specific(self, query: str) -> bool:
        """Check if query is domain-specific (tax/BEPS related)."""
        domain_keywords = []
        for kw_list in self.tax_keywords.values():
            domain_keywords.extend(kw_list)
            
        query_lower = query.lower()
        for kw in domain_keywords:
            if kw in query_lower:
                return True
                
        return False
    
    def _calculate_confidence(self, query: str, query_type: QueryType, keywords: List[str]) -> float:
        """Calculate classification confidence score."""
        confidence = 0.5  # Base confidence
        
        # Boost confidence for domain-specific queries
        if self._is_domain_specific(query):
            confidence += 0.3
            
        # Boost for clear query patterns
        if query_type != QueryType.GENERAL:
            confidence += 0.2
            
        # Reduce confidence for very short queries
        if len(query.split()) < 3:
            confidence -= 0.2
            
        # Boost for good keyword coverage
        if len(keywords) >= 3:
            confidence += 0.1
            
        return max(0.1, min(confidence, 1.0))