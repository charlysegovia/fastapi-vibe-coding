#!/usr/bin/env python3
"""
Simple script to run the RAG integration tests.
"""

import asyncio
import sys
import os

async def main():
    """Run the integration tests."""
    try:
        # Import and run the test
        from test_rag import main as run_tests
        exit_code = await run_tests()
        
        if exit_code == 0:
            print("\nAll integration tests passed!")
            print("The RAG system is working correctly and is protected against abuse.")
        else:
            print(f"\nSome tests failed. Exit code: {exit_code}")
            print("Please review the test results above.")
        
        return exit_code
        
    except ImportError as e:
        print(f"Error importing test module: {e}")
        print("Make sure test_rag.py is in the current directory.")
        return 1
    except Exception as e:
        print(f"Unexpected error running tests: {e}")
        return 1

if __name__ == "__main__":
    # Check if server is running
    print("Checking if RAG server is running...")
    print("Make sure the server is running on http://localhost:8000")
    print("You can start it with: python start_working.py")
    print("-" * 50)
    
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 