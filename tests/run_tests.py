#!/usr/bin/env python3
"""
Test runner for BEPS Agent system
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def run_async_tests():
    """Run async tests with proper event loop handling."""
    logger.info("Starting BEPS Agent test suite...")
    
    # Run pytest with async support
    exit_code = pytest.main([
        "tests/test_agent.py",
        "-v",
        "--tb=short",
        "--asyncio-mode=auto"
    ])
    
    return exit_code


def run_unit_tests():
    """Run unit tests for individual components."""
    logger.info("Running unit tests...")
    
    # Test imports
    try:
        from src.agent.beps_agent import BEPSAgent
        from src.agent.query_classifier import QueryClassifier
        from src.agent.confidence_scorer import ConfidenceScorer
        from src.agent.web_search_handler import WebSearchHandler
        from src.agent.rag_handler import RAGHandler
        from src.agent.response_router import ResponseRouter
        
        logger.info("✓ All imports successful")
        
        # Test basic instantiation
        agent = BEPSAgent(None)
        classifier = QueryClassifier()
        scorer = ConfidenceScorer()
        handler = WebSearchHandler()
        
        logger.info("✓ All components instantiated successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Import/instantiation failed: {e}")
        return False


def test_configuration():
    """Test configuration loading."""
    logger.info("Testing configuration...")
    
    try:
        from src.config import load_config
        
        # Test config loading
        config = load_config("config/processing_config.yaml")
        logger.info(f"✓ Configuration loaded: {config.collection_name}")
        
        return True
        
    except Exception as e:
        logger.error(f"Configuration test failed: {e}")
        return False


def test_pdf_processing():
    """Test PDF processing capabilities."""
    logger.info("Testing PDF processing...")
    
    try:
        from examples.batch_process_pdfs import process_pdfs
        
        # Check if PDF files exist
        pdf_dir = project_root / "data" / "raw"
        if pdf_dir.exists():
            pdf_files = list(pdf_dir.glob("*.pdf"))
            logger.info(f"Found {len(pdf_files)} PDF files")
            
            if pdf_files:
                logger.info("✓ PDF processing setup ready")
                return True
            else:
                logger.warning("No PDF files found for testing")
                return True
        else:
            logger.warning("PDF directory not found")
            return True
            
    except Exception as e:
        logger.error(f"PDF processing test failed: {e}")
        return False


async def main():
    """Main test runner."""
    logger.info("=" * 50)
    logger.info("BEPS Agent Test Suite")
    logger.info("=" * 50)
    
    # Run basic tests
    tests_passed = 0
    total_tests = 4
    
    if run_unit_tests():
        tests_passed += 1
    
    if test_configuration():
        tests_passed += 1
    
    if test_pdf_processing():
        tests_passed += 1
    
    # Run async tests
    try:
        exit_code = await run_async_tests()
        if exit_code == 0:
            tests_passed += 1
            logger.info("✓ All async tests passed")
        else:
            logger.error(f"Async tests failed with exit code: {exit_code}")
    except Exception as e:
        logger.error(f"Async test execution failed: {e}")
    
    logger.info("=" * 50)
    logger.info(f"Test Results: {tests_passed}/{total_tests} test suites passed")
    
    if tests_passed == total_tests:
        logger.info("🎉 All tests passed!")
        return 0
    else:
        logger.error("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)