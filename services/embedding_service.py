import os
import httpx
import asyncio
from typing import List

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-ada-002"

async def get_embedding(text: str) -> List[float]:
    """Get the embedding of a text using OpenAI asynchronously."""
    url = "https://api.openai.com/v1/embeddings"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    data = {"input": text, "model": EMBEDDING_MODEL}
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(url, json=data, headers=headers)
                resp.raise_for_status()
                return resp.json()["data"][0]["embedding"]
        except Exception as e:
            if attempt == 2:
                raise
            await asyncio.sleep(2 ** attempt) 