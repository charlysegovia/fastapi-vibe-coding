import pytest
from unittest.mock import MagicMock
from services.milvus_service import search_similar

@pytest.mark.asyncio
async def test_search_similar():
    # Mock a Milvus collection and search result
    mock_collection = MagicMock()
    mock_hit = MagicMock()
    mock_hit.entity.get.side_effect = lambda k: {'chunk_id': 'c1', 'page_number': 1, 'text': 'abc', 'filename': 'doc.pdf'}[k]
    mock_hit.distance = 0.99
    mock_collection.search.return_value = [[mock_hit]]
    results = await search_similar(mock_collection, [0.1]*1536, 'doc.pdf', 1)
    assert isinstance(results, list)
    assert results[0]['chunk_id'] == 'c1'
    assert results[0]['page_number'] == 1
    assert results[0]['filename'] == 'doc.pdf' 