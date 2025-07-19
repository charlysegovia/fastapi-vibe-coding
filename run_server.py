#!/usr/bin/env python3
"""
Alternative server startup script to avoid uvicorn signal handling issues on Windows.
"""

import uvicorn
import os
import sys

if __name__ == "__main__":
    # Configure for Windows compatibility
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
        # Disable signal handling that causes issues on Windows
        loop="asyncio",
        access_log=True
    ) 