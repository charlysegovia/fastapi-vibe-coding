#!/usr/bin/env python3
"""
Windows-compatible server startup script.
"""

import uvicorn
import sys
import os

def main():
    """Start the server with Windows-compatible settings."""
    try:
        print("Starting GitHub RAG System...")
        print("Server will be available at: http://localhost:8000")
        print("Web interface: http://localhost:8000/static/index.html")
        print("Press Ctrl+C to stop the server")
        print("-" * 50)
        
        # Start uvicorn with Windows-compatible settings
        uvicorn.run(
            "main_simple:app",
            host="127.0.0.1",
            port=8000,
            reload=False,  # Disable reload to avoid signal issues
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