#!/usr/bin/env python3
"""
Test script to verify the TF-IDF and chunking fixes work correctly.
"""

import asyncio
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

async def test_tfidf_fix():
    """Test the TF-IDF fix with a simple example."""
    from milvus_dual import MilvusDualService
    from chunking_strategies import ChunkingStrategies
    
    print("Testing TF-IDF fix...")
    
    # Create test chunks
    chunking = ChunkingStrategies()
    
    # Mock file info and repo info
    file_info = {
        'path': 'test.py',
        'size': 100,
        'sha': 'test123',
        'author': 'test',
        'last_modified': '2024-01-01'
    }
    
    repo_info = {
        'name': 'test-repo',
        'default_branch': 'main'
    }
    
    # Test with different content sizes
    test_contents = [
        "This is a simple test file with some content.",
        "def hello_world():\n    print('Hello, World!')\n\n# This is a comment\n\nclass TestClass:\n    def __init__(self):\n        self.value = 42",
        "Just a few words",
        "A" * 50  # Very short content
    ]
    
    for i, content in enumerate(test_contents):
        print(f"\nTest {i+1}: Content length = {len(content)}")
        
        # Create chunks
        dense_chunks = chunking.create_dense_chunks(content, file_info, repo_info)
        sparse_chunks = chunking.create_sparse_chunks(content, file_info, repo_info)
        
        print(f"  Dense chunks: {len(dense_chunks)}")
        print(f"  Sparse chunks: {len(sparse_chunks)}")
        
        if dense_chunks:
            print(f"  First dense chunk type: {dense_chunks[0]['chunk_type']}")
        if sparse_chunks:
            print(f"  First sparse chunk type: {sparse_chunks[0]['chunk_type']}")

async def test_milvus_service():
    """Test the Milvus service initialization."""
    try:
        from milvus_dual import MilvusDualService
        
        print("\nTesting Milvus service initialization...")
        
        # Check if environment variables are set
        required_vars = ['MILVUS_URI', 'MILVUS_TOKEN']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            print(f"Missing environment variables: {missing_vars}")
            print("Skipping Milvus test - please set the required variables")
            return
        
        service = MilvusDualService()
        await service.initialize()
        print("OK - Milvus service initialized successfully")
        
    except Exception as e:
        print(f"ERROR - Milvus service initialization failed: {e}")

async def main():
    """Run all tests."""
    print("Running TF-IDF and chunking fixes tests...")
    
    await test_tfidf_fix()
    await test_milvus_service()
    
    print("\nTests completed!")

if __name__ == "__main__":
    asyncio.run(main()) 