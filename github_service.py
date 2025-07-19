import os
import httpx
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

class GitHubService:
    """Service for interacting with GitHub API."""
    
    def __init__(self):
        self.token = os.getenv("GITHUB_TOKEN")
        self.username = os.getenv("GITHUB_USERNAME")
        self.base_url = "https://api.github.com"
        self.headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "FastAPI-RAG-App"
        }
        
        if not self.token:
            raise ValueError("GITHUB_TOKEN environment variable is required")
        if not self.username:
            raise ValueError("GITHUB_USERNAME environment variable is required")
    
    async def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make a request to GitHub API."""
        url = f"{self.base_url}{endpoint}"
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
    
    async def list_repositories(self) -> List[Dict[str, Any]]:
        """List all repositories for the authenticated user."""
        try:
            repos = await self._make_request(f"/users/{self.username}/repos")
            return [
                {
                    "name": repo["name"],
                    "full_name": repo["full_name"],
                    "description": repo["description"],
                    "language": repo["language"],
                    "stars": repo["stargazers_count"],
                    "forks": repo["forks_count"],
                    "updated_at": repo["updated_at"],
                    "private": repo["private"]
                }
                for repo in repos
            ]
        except Exception as e:
            logging.error(f"Error listing repositories: {e}")
            raise
    
    async def get_repository_info(self, repo_name: str) -> Dict[str, Any]:
        """Get detailed information about a repository."""
        try:
            repo = await self._make_request(f"/repos/{self.username}/{repo_name}")
            return {
                "name": repo["name"],
                "full_name": repo["full_name"],
                "description": repo["description"],
                "language": repo["language"],
                "default_branch": repo["default_branch"],
                "created_at": repo["created_at"],
                "updated_at": repo["updated_at"],
                "size": repo["size"],
                "stars": repo["stargazers_count"],
                "forks": repo["forks_count"]
            }
        except Exception as e:
            logging.error(f"Error getting repository info for {repo_name}: {e}")
            raise
    
    async def get_repository_files(self, repo_name: str, path: str = "") -> List[Dict[str, Any]]:
        """Get all files from a repository recursively."""
        try:
            files = []
            contents = await self._make_request(f"/repos/{self.username}/{repo_name}/contents/{path}")
            
            for item in contents:
                if item["type"] == "file":
                    files.append({
                        "name": item["name"],
                        "path": item["path"],
                        "size": item["size"],
                        "sha": item["sha"],
                        "type": item["type"],
                        "url": item["url"],
                        "download_url": item["download_url"]
                    })
                elif item["type"] == "dir":
                    # Recursively get files from subdirectories
                    subfiles = await self.get_repository_files(repo_name, item["path"])
                    files.extend(subfiles)
            
            return files
        except Exception as e:
            logging.error(f"Error getting files for {repo_name}: {e}")
            raise
    
    async def get_file_content(self, repo_name: str, file_path: str) -> str:
        """Get the content of a specific file."""
        try:
            # Get file info first
            file_info = await self._make_request(f"/repos/{self.username}/{repo_name}/contents/{file_path}")
            
            # Get the latest commit for this file
            commits = await self._make_request(f"/repos/{self.username}/{repo_name}/commits", {
                "path": file_path,
                "per_page": 1
            })
            
            latest_commit = commits[0] if commits else None
            
            # Download content
            if file_info["download_url"]:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(file_info["download_url"])
                    response.raise_for_status()
                    content = response.text
            else:
                # For files that can't be downloaded directly, use the API
                content = file_info.get("content", "")
                if file_info.get("encoding") == "base64":
                    import base64
                    content = base64.b64decode(content).decode('utf-8')
            
            return content
            
        except Exception as e:
            logging.error(f"Error getting file content for {file_path}: {e}")
            raise
    
    async def get_file_metadata(self, repo_name: str, file_path: str) -> Dict[str, Any]:
        """Get detailed metadata for a file including commit info."""
        try:
            # Get file info
            file_info = await self._make_request(f"/repos/{self.username}/{repo_name}/contents/{file_path}")
            
            # Get latest commit
            commits = await self._make_request(f"/repos/{self.username}/{repo_name}/commits", {
                "path": file_path,
                "per_page": 1
            })
            
            latest_commit = commits[0] if commits else None
            
            return {
                "name": file_info["name"],
                "path": file_info["path"],
                "size": file_info["size"],
                "sha": file_info["sha"],
                "commit_hash": latest_commit["sha"] if latest_commit else None,
                "author": latest_commit["commit"]["author"]["name"] if latest_commit else None,
                "last_modified": latest_commit["commit"]["author"]["date"] if latest_commit else None,
                "message": latest_commit["commit"]["message"] if latest_commit else None
            }
            
        except Exception as e:
            logging.error(f"Error getting file metadata for {file_path}: {e}")
            raise 