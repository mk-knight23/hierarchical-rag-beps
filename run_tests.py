#!/usr/bin/env python3
"""
Test runner script for the Hierarchical RAG document processing pipeline.
"""

import asyncio
import logging
import sys
from pathlib import Path
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_unit_tests():
    """Run the unit tests using pytest."""
    logger.info("Running unit tests...")
    
    test_file = Path(__file__).parent / "tests" / "test_pipeline.py"
    
    if not test_file.exists():
        logger.error(f"Test file not found: {test_file}")
        return False
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", str(test_file), "-v"
        ], capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        return False


async def run_integration_test():
    """Run a simple integration test."""
    logger.info("Running integration test...")
    
    try:
        from examples.basic_usage import main as basic_usage_main
        
        logger.info("Testing basic usage example...")
        await basic_usage_main()
        
        logger.info("Integration test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        return False


async def main():
    """Main test runner."""
    logger.info("=" * 60)
    logger.info("HIERARCHICAL RAG DOCUMENT PROCESSING TESTS")
    logger.info("=" * 60)
    
    # Run unit tests
    unit_tests_passed = run_unit_tests()
    
    if unit_tests_passed:
        logger.info("✅ Unit tests passed!")
    else:
        logger.error("❌ Unit tests failed!")
    
    # Run integration test
    integration_passed = await run_integration_test()
    
    if integration_passed:
        logger.info("✅ Integration test passed!")
    else:
        logger.error("❌ Integration test failed!")
    
    # Final summary
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Unit Tests: {'PASS' if unit_tests_passed else 'FAIL'}")
    logger.info(f"Integration Test: {'PASS' if integration_passed else 'FAIL'}")
    
    if unit_tests_passed and integration_passed:
        logger.info("🎉 All tests passed!")
        return 0
    else:
        logger.error("💥 Some tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)