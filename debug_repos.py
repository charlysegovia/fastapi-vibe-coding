#!/usr/bin/env python3
"""
Debug script to see what the repos endpoint returns.
"""

import asyncio
import httpx
import json

async def debug_repos():
    """Debug the repos endpoint."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            print("Testing repos endpoint...")
            response = await client.get("http://localhost:8000/api/repos")
            
            print(f"Status code: {response.status_code}")
            print(f"Response headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"Response type: {type(data)}")
                    print(f"Response length: {len(data) if isinstance(data, list) else 'Not a list'}")
                    print(f"Response: {json.dumps(data, indent=2)[:500]}...")
                    
                    if isinstance(data, list) and len(data) > 0:
                        print(f"First item: {data[0]}")
                        print(f"Keys in first item: {list(data[0].keys()) if isinstance(data[0], dict) else 'Not a dict'}")
                    else:
                        print("Response is not a list or is empty")
                        
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")
                    print(f"Raw response: {response.text[:200]}...")
            else:
                print(f"Error response: {response.text}")
                
        except Exception as e:
            print(f"Exception: {e}")

if __name__ == "__main__":
    asyncio.run(debug_repos()) 