#!/usr/bin/env python3
"""
Simple test to verify basic query functionality.
"""

import asyncio
import httpx
import json

async def test_simple_query():
    """Test a simple query to identify issues."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # Test 1: Check if server is running
            print("1. Checking server status...")
            response = await client.get("http://localhost:8000/")
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"   Service ready: {data.get('service_status', {}).get('service_ready', False)}")
            
            # Test 2: Check initialization status
            print("\n2. Checking initialization status...")
            response = await client.get("http://localhost:8000/api/init-status")
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"   Init complete: {data.get('initialization_complete', False)}")
                print(f"   Service available: {data.get('service_available', False)}")
            
            # Test 3: Try a simple query
            print("\n3. Testing simple query...")
            query_data = {
                "question": "What is this system about?",
                "top_k": 3,
                "enable_reranking": False,
                "show_scores": False,
                "show_justification": False
            }
            
            response = await client.post(
                "http://localhost:8000/api/query",
                json=query_data
            )
            
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"   Answer length: {len(data.get('answer', ''))}")
                print(f"   Sources count: {len(data.get('sources', []))}")
                print(f"   Answer preview: {data.get('answer', '')[:100]}...")
            else:
                print(f"   Error: {response.text}")
            
            # Test 4: Check search stats
            print("\n4. Checking search stats...")
            response = await client.get("http://localhost:8000/api/search-stats")
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"   Total repositories: {data.get('total_repositories', 0)}")
                print(f"   Total chunks: {data.get('total_chunks', 0)}")
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_simple_query()) 