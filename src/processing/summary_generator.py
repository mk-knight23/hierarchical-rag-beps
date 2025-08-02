import logging
from typing import List, Dict, Optional
import asyncio
from models.data_structures import Summary, Document, Chunk

logger = logging.getLogger(__name__)

class SummaryGenerator:
    """Generate summaries using Phi3 model"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model_name = config.get('summary', {}).get('model_name', 'microsoft/Phi-3-mini-4k-instruct')
        self.max_summary_length = config.get('summary', {}).get('max_summary_length', 500)
        self.max_input_length = config.get('summary', {}).get('max_input_length', 3000)
        self.temperature = config.get('summary', {}).get('temperature', 0.3)
        
    async def generate_document_summary(self, document: Document) -> Summary:
        """Generate summary for a complete document"""
        try:
            # Prepare content for summarization
            content = self._prepare_content(document.content)
            
            # Generate summary
            summary_text = await self._generate_summary(content)
            
            # Extract key topics and keywords
            topics = await self._extract_topics(content)
            keywords = await self._extract_keywords(content)
            
            return Summary(
                document_id=document.id,
                summary=summary_text,
                key_topics=topics,
                keywords=keywords
            )
            
        except Exception as e:
            logger.error(f"Failed to generate summary for {document.id}: {str(e)}")
            return Summary(
                document_id=document.id,
                summary="",
                key_topics=[],
                keywords=[]
            )
    
    async def generate_chunk_summaries(self, chunks: List[Chunk]) -> List[Summary]:
        """Generate summaries for chunks (for future use)"""
        summaries = []
        
        # Process chunks in batches
        batch_size = 5
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            batch_summaries = await asyncio.gather(
                *[self._generate_chunk_summary(chunk) for chunk in batch]
            )
            summaries.extend(batch_summaries)
        
        return summaries
    
    def _prepare_content(self, content: str) -> str:
        """Prepare content for summarization"""
        # Remove page markers
        content = content.replace('[Page', '\n[Page')
        
        # Truncate if too long
        if len(content) > self.max_input_length:
            # Take beginning and end
            half_length = self.max_input_length // 2
            beginning = content[:half_length]
            end = content[-half_length:]
            content = f"{beginning}\n\n[... truncated ...]\n\n{end}"
        
        return content
    
    async def _generate_summary(self, content: str) -> str:
        """Generate summary using Phi3 model"""
        # For now, use a simple extraction-based approach
        # In production, this would use the actual Phi3 model
        
        # Extract first paragraph as summary
        paragraphs = content.split('\n\n')
        if paragraphs:
            summary = paragraphs[0][:self.max_summary_length]
            if len(summary) < 100 and len(paragraphs) > 1:
                summary += " " + paragraphs[1][:self.max_summary_length - len(summary)]
        else:
            summary = content[:self.max_summary_length]
        
        # Clean up summary
        summary = summary.strip()
        if len(summary) > self.max_summary_length:
            summary = summary[:self.max_summary_length-3] + "..."
        
        return summary
    
    async def _extract_topics(self, content: str) -> List[str]:
        """Extract key topics from content"""
        # Simple topic extraction based on frequency
        words = content.lower().split()
        
        # Filter out common words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'within',
            'without', 'toward', 'against', 'upon', 'regarding', 'concerning',
            'including', 'containing', 'consisting', 'following', 'preceding',
            'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'can', 'shall'
        }
        
        # Count word frequencies
        word_freq = {}
        for word in words:
            word = word.strip('.,!?;:"()[]{}')
            if word and word not in stop_words and len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top topics
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        topics = [word for word, freq in sorted_words[:10] if freq > 2]
        
        return topics
    
    async def _extract_keywords(self, content: str) -> List[str]:
        """Extract keywords from content"""
        # Extract phrases that appear to be technical terms
        keywords = []
        
        # Look for capitalized phrases (likely technical terms)
        import re
        capitalized_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
        keywords.extend(capitalized_phrases)
        
        # Look for phrases in quotes
        quoted_phrases = re.findall(r'"([^"]*)"', content)
        keywords.extend(quoted_phrases)
        
        # Look for parenthetical phrases
        paren_phrases = re.findall(r'\(([^)]*)\)', content)
        keywords.extend(paren_phrases)
        
        # Clean and deduplicate
        keywords = list(set([k.strip() for k in keywords if len(k.strip()) > 3]))
        
        return keywords[:15]
    
    async def _generate_chunk_summary(self, chunk: Chunk) -> Summary:
        """Generate summary for a single chunk"""
        try:
            content = self._prepare_content(chunk.text)  # Changed from chunk.content to chunk.text
            summary_text = await self._generate_summary(content)
            
            return Summary(
                document_id=chunk.document_id,
                summary=summary_text,
                key_topics=await self._extract_topics(content),
                keywords=await self._extract_keywords(content)
            )
            
        except Exception as e:
            logger.error(f"Failed to generate chunk summary for {chunk.id}: {str(e)}")
            return Summary(
                document_id=chunk.document_id,
                summary="",
                key_topics=[],
                keywords=[]
            )
    
    async def batch_generate_summaries(self, documents: List[Document]) -> List[Summary]:
        """Generate summaries for multiple documents in parallel"""
        tasks = [self.generate_document_summary(doc) for doc in documents]
        summaries = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        valid_summaries = []
        for summary in summaries:
            if isinstance(summary, Exception):
                logger.error(f"Summary generation failed: {str(summary)}")
            else:
                valid_summaries.append(summary)
        
        return valid_summaries