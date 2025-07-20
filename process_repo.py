#!/usr/bin/env python3
"""
Simple script to process a repository and add data to the RAG system.
"""

import asyncio
import httpx
import json

async def list_repositories():
    """List available repositories."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get("http://localhost:8000/api/repos")
            if response.status_code == 200:
                repos = response.json()
                print(f"Found {len(repos)} repositories:")
                for i, repo in enumerate(repos[:5]):  # Show first 5
                    description = repo.get('description', 'No description')
                    if description:
                        # Remove Unicode characters that cause issues on Windows
                        clean_description = ''.join(char for char in description if ord(char) < 128)
                        desc_preview = clean_description[:50] + "..." if len(clean_description) > 50 else clean_description
                    else:
                        desc_preview = "No description"
                    print(f"  {i+1}. {repo['name']} - {desc_preview}")
                return repos
            else:
                print(f"Error listing repositories: {response.status_code}")
                return []
        except Exception as e:
            print(f"Error: {e}")
            return []

async def process_repository(repo_name):
    """Process a specific repository."""
    async with httpx.AsyncClient(timeout=300.0) as client:  # 5 minute timeout
        try:
            print(f"Processing repository: {repo_name}")
            
            # Start processing
            response = await client.post(
                f"http://localhost:8000/api/process/{repo_name}",
                json={}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"Successfully processed {repo_name}")
                print(f"   Files processed: {result.get('processed_files', 0)}")
                print(f"   Total chunks: {result.get('total_chunks', 0)}")
                print(f"   Dense chunks: {result.get('dense_chunks', 0)}")
                print(f"   Sparse chunks: {result.get('sparse_chunks', 0)}")
                return True
            else:
                print(f"Failed to process {repo_name}: {response.status_code}")
                print(f"   Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"Error processing {repo_name}: {e}")
            return False

async def main():
    """Main function to process a repository."""
    print("Listing available repositories...")
    repos = await list_repositories()
    
    if not repos:
        print("No repositories found or error listing repositories")
        return
    
    # Process the first repository
    first_repo = repos[0]['name']
    print(f"\nProcessing first repository: {first_repo}")
    
    success = await process_repository(first_repo)
    
    if success:
        print(f"\nRepository {first_repo} processed successfully!")
        print("Now you can run the integration tests again.")
    else:
        print(f"\nFailed to process repository {first_repo}")
        print("Check the server logs for more details.")

if __name__ == "__main__":
    asyncio.run(main()) 