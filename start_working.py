#!/usr/bin/env python3
"""
Startup script for the working GitHub RAG System.
"""

import uvicorn
import sys
import os

def main():
    """Start the server with the working version."""
    try:
        print("Starting GitHub RAG System (Working Version)...")
        print("Server will be available at: http://localhost:8000")
        print("Web interface: http://localhost:8000/static/index.html")
        print("Initialization status: http://localhost:8000/api/init-status")
        print("Press Ctrl+C to stop the server")
        print("-" * 50)
        
        # Start uvicorn with the working version
        uvicorn.run(
            "main_working:app",
            host="127.0.0.1",
            port=8000,
            reload=True,  # Enable reload for development
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 