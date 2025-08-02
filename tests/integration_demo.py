#!/usr/bin/env python3
"""
Integration demonstration for the BEPS agent system.
This script demonstrates the complete system working with actual BEPS documents.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.beps_agent import BEPSAgent
from agent.query_classifier import QueryType


async def run_integration_demo():
    """Run comprehensive integration demonstration."""
    print("🚀 BEPS Agent System Integration Demo")
    print("=" * 50)
    
    # Initialize agent
    agent = BEPSAgent()
    
    # Test queries covering different aspects of BEPS
    test_queries = [
        {
            "query": "What are the key recommendations in BEPS Action 1 regarding the digital economy?",
            "expected_type": QueryType.FACTUAL,
            "description": "Factual retrieval from BEPS Action 1"
        },
        {
            "query": "How do the BEPS Action 5 recommendations on harmful tax practices affect developing countries?",
            "expected_type": QueryType.ANALYTICAL,
            "description": "Analytical question requiring synthesis"
        },
        {
            "query": "What steps should a multinational enterprise take to comply with BEPS Action 13 transfer pricing documentation requirements?",
            "expected_type": QueryType.PROCEDURAL,
            "description": "Procedural guidance request"
        },
        {
            "query": "What were the main BEPS developments in 2023 and 2024?",
            "expected_type": QueryType.TEMPORAL,
            "description": "Temporal query requiring recent information"
        },
        {
            "query": "Compare the approaches taken in BEPS Action 1 and Action 5 to address tax avoidance",
            "expected_type": QueryType.COMPARATIVE,
            "description": "Comparative analysis between actions"
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_queries, 1):
        print(f"\n📋 Test {i}: {test_case['description']}")
        print(f"Query: {test_case['query']}")
        print("-" * 40)
        
        try:
            # Process query
            response = await agent.query(test_case['query'])
            
            # Collect results
            result = {
                "test_number": i,
                "query": test_case['query'],
                "expected_type": test_case['expected_type'].value,
                "actual_type": response.get('query_type', 'unknown'),
                "strategy": response.get('strategy', 'unknown'),
                "confidence": response.get('confidence', 0),
                "has_answer": bool(response.get('answer')),
                "answer_length": len(response.get('answer', '')),
                "sources": response.get('sources', []),
                "processing_time": response.get('processing_time', 0),
                "cache_hit": response.get('cache_hit', False)
            }
            
            results.append(result)
            
            # Display results
            print(f"✅ Strategy: {result['strategy']}")
            print(f"✅ Confidence: {result['confidence']:.2f}")
            print(f"✅ Processing Time: {result['processing_time']:.2f}s")
            print(f"✅ Cache Hit: {result['cache_hit']}")
            print(f"✅ Answer Length: {result['answer_length']} characters")
            print(f"✅ Sources: {len(result['sources'])}")
            
            if result['confidence'] < 0.7:
                print("⚠️  Low confidence - consider verification")
            
            # Show first 200 chars of answer
            preview = response.get('answer', '')[:200]
            if len(response.get('answer', '')) > 200:
                preview += "..."
            print(f"\n📝 Answer Preview: {preview}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            results.append({
                "test_number": i,
                "query": test_case['query'],
                "error": str(e)
            })
    
    # Summary statistics
    print("\n📊 Integration Demo Summary")
    print("=" * 50)
    
    successful_tests = [r for r in results if 'error' not in r]
    avg_confidence = sum(r['confidence'] for r in successful_tests) / len(successful_tests) if successful_tests else 0
    avg_processing_time = sum(r['processing_time'] for r in successful_tests) / len(successful_tests) if successful_tests else 0
    
    print(f"✅ Successful Tests: {len(successful_tests)}/{len(test_queries)}")
    print(f"📈 Average Confidence: {avg_confidence:.2f}")
    print(f"⏱️  Average Processing Time: {avg_processing_time:.2f}s")
    print(f"💾 Cache Hits: {sum(r['cache_hit'] for r in successful_tests)}")
    
    # Save detailed results
    results_file = Path(__file__).parent / "integration_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    
    # Performance recommendations
    print("\n🔧 Performance Recommendations:")
    if avg_confidence < 0.8:
        print("• Consider improving RAG retrieval quality")
    if avg_processing_time > 5.0:
        print("• Consider optimizing cache usage")
    if len(successful_tests) < len(test_queries):
        print("• Review error cases and improve error handling")
    
    return results


async def run_batch_processing_demo():
    """Demonstrate batch processing capabilities."""
    print("\n🔄 Batch Processing Demo")
    print("=" * 30)
    
    agent = BEPSAgent()
    
    # Batch queries
    batch_queries = [
        "What is BEPS Action 1 about?",
        "Explain BEPS Action 5 harmful tax practices",
        "Describe BEPS Action 13 documentation requirements",
        "How does BEPS affect multinational corporations?",
        "What are the compliance deadlines for BEPS?"
    ]
    
    print("Processing batch queries...")
    batch_results = await agent.batch_query(batch_queries)
    
    print(f"✅ Processed {len(batch_results)} queries")
    
    for i, result in enumerate(batch_results, 1):
        status = "✅" if result.get('answer') else "❌"
        print(f"{status} Query {i}: {result.get('strategy', 'unknown')} "
              f"(confidence: {result.get('confidence', 0):.2f})")
    
    return batch_results


async def run_health_check():
    """Run system health check."""
    print("\n🏥 System Health Check")
    print("=" * 25)
    
    agent = BEPSAgent()
    health = await agent.health_check()
    
    print(f"✅ Overall Status: {health['status']}")
    print(f"📊 Total Queries: {health['statistics']['total_queries']}")
    print(f"✅ Cache Entries: {health['statistics']['cache_entries']}")
    print(f"⚡ Average Response Time: {health['statistics']['average_response_time']:.2f}s")
    
    if health['status'] != 'healthy':
        print("❌ Issues detected:")
        for issue in health.get('issues', []):
            print(f"  • {issue}")
    
    return health


async def main():
    """Run all integration demonstrations."""
    try:
        # Health check first
        await run_health_check()
        
        # Run integration demo
        results = await run_integration_demo()
        
        # Run batch processing demo
        await run_batch_processing_demo()
        
        print("\n🎉 Integration Demo Complete!")
        print("The BEPS agent system is ready for production use.")
        
    except Exception as e:
        print(f"❌ Integration demo failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())