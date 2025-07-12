import pytest
from unittest.mock import patch, AsyncMock
from services.embedding_service import get_embedding

@pytest.mark.asyncio
@patch('httpx.AsyncClient.post')
async def test_get_embedding(mock_post):
    # Mock OpenAI API response
    mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value={
        "data": [{"embedding": [0.1] * 1536}]
    })
    mock_post.return_value.__aenter__.return_value.raise_for_status = AsyncMock()
    embedding = await get_embedding("test text")
    assert isinstance(embedding, list)
    assert len(embedding) == 1536
    assert all(isinstance(x, float) for x in embedding) 