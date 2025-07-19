#!/usr/bin/env python3
"""
Test script to verify re-ranking functionality.
"""

import asyncio
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_reranking():
    """Test re-ranking functionality."""
    print("Testing Re-ranking Functionality")
    print("=" * 35)
    
    try:
        from milvus_dual import MilvusDualService
        from reranking_service import ReRankingService
        from pymilvus import utility, connections
        
        milvus_service = MilvusDualService()
        
        # Connect to Milvus
        connections.connect(uri=milvus_service.milvus_uri, token=milvus_service.milvus_token)
        print("[OK] Connected to Milvus")
        
        # Test query
        test_query = "How does the system work?"
        
        print(f"\nTesting search with re-ranking for query: '{test_query}'")
        
        # Perform search with re-ranking
        search_results = await milvus_service.search_hybrid(
            query=test_query,
            top_k=10,
            enable_reranking=True,
            rerank_top_k=5,
            show_scores=True,
            show_justification=True
        )
        
        print(f"[OK] Search completed")
        print(f"Total results: {search_results['total_results']}")
        print(f"Dense results: {search_results['dense_count']}")
        print(f"Sparse results: {search_results['sparse_count']}")
        
        # Check re-ranking info
        re_ranking_info = search_results.get('re_ranking', {})
        print(f"\nRe-ranking information:")
        print(f"  Enabled: {re_ranking_info.get('enabled', False)}")
        print(f"  Successful: {re_ranking_info.get('successful', False)}")
        print(f"  Evaluated count: {re_ranking_info.get('evaluated_count', 0)}")
        
        if re_ranking_info.get('error'):
            print(f"  Error: {re_ranking_info['error']}")
        
        # Display results
        print(f"\nRe-ranked results:")
        for i, result in enumerate(search_results['results']):
            print(f"\nResult {i+1}:")
            print(f"  File: {result.get('file_path', 'Unknown')}")
            print(f"  Text: {result.get('text', '')[:100]}...")
            print(f"  Score: {result.get('score', 0):.4f}")
            print(f"  Search type: {result.get('search_type', 'Unknown')}")
            
            if result.get('relevance_score'):
                print(f"  Relevance score: {result['relevance_score']}/10")
                print(f"  Original rank: {result.get('original_rank', 'N/A')}")
                if result.get('justification'):
                    print(f"  Justification: {result['justification']}")
        
        print(f"\n[SUCCESS] Re-ranking test completed!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_reranking_service_directly():
    """Test re-ranking service directly with mock data."""
    print("\nTesting Re-ranking Service Directly")
    print("=" * 40)
    
    try:
        from reranking_service import ReRankingService
        
        reranking_service = ReRankingService()
        
        # Mock search results
        mock_results = [
            {
                "text": "This is a comprehensive guide about how the system works. It explains all the components and their interactions.",
                "file_path": "docs/system_guide.md",
                "chunk_type": "paragraph",
                "score": 0.85,
                "search_type": "dense"
            },
            {
                "text": "The system architecture consists of multiple microservices that communicate via REST APIs.",
                "file_path": "docs/architecture.md",
                "chunk_type": "sentence",
                "score": 0.78,
                "search_type": "dense"
            },
            {
                "text": "To install the system, run: npm install && npm start",
                "file_path": "README.md",
                "chunk_type": "file",
                "score": 0.65,
                "search_type": "sparse"
            }
        ]
        
        test_query = "How does the system work?"
        
        print(f"Testing with query: '{test_query}'")
        print(f"Mock results: {len(mock_results)}")
        
        # Test re-ranking
        reranking_result = await reranking_service.re_rank_results(
            query=test_query,
            results=mock_results,
            top_k=3,
            show_scores=True,
            show_justification=True
        )
        
        print(f"[OK] Re-ranking completed")
        print(f"Successful: {reranking_result['re_ranking_successful']}")
        print(f"Evaluated count: {reranking_result['evaluated_count']}")
        print(f"Total results: {reranking_result['total_results']}")
        
        if reranking_result.get('error'):
            print(f"Error: {reranking_result['error']}")
        
        # Display results
        print(f"\nRe-ranked results:")
        for i, result in enumerate(reranking_result['re_ranked_results']):
            print(f"\nResult {i+1}:")
            print(f"  File: {result.get('file_path', 'Unknown')}")
            print(f"  Text: {result.get('text', '')[:80]}...")
            print(f"  Original score: {result.get('score', 0):.4f}")
            print(f"  Relevance score: {result.get('relevance_score', 'N/A')}/10")
            print(f"  Original rank: {result.get('original_rank', 'N/A')}")
            if result.get('justification'):
                print(f"  Justification: {result['justification']}")
        
        print(f"\n[SUCCESS] Direct re-ranking test completed!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Direct test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests."""
    print("Re-ranking System Testing")
    print("=" * 30)
    
    # Test re-ranking service directly
    direct_ok = await test_reranking_service_directly()
    
    # Test with Milvus integration
    milvus_ok = await test_reranking()
    
    print("\n" + "=" * 30)
    print("Test Results:")
    print(f"Direct Re-ranking: {'[OK]' if direct_ok else '[ERROR]'}")
    print(f"Milvus Integration: {'[OK]' if milvus_ok else '[ERROR]'}")
    
    if all([direct_ok, milvus_ok]):
        print("\n[SUCCESS] All re-ranking tests passed!")
        print("The re-ranking system is working correctly.")
    else:
        print("\n[WARNING] Some tests failed. Check the errors above.")

if __name__ == "__main__":
    asyncio.run(main()) 