import pytest
from utils.pdf_utils import extract_text_by_page
from PyPDF2 import PdfWriter
import io

@pytest.mark.asyncio
async def test_extract_text_by_page():
    # Create a simple PDF in memory with two pages
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.add_blank_page(width=72, height=72)
    pdf_bytes = io.BytesIO()
    writer.write(pdf_bytes)
    pdf_bytes.seek(0)
    # This PDF has no text, so all pages should be empty strings
    pages = extract_text_by_page(pdf_bytes.getvalue())
    assert isinstance(pages, list)
    assert len(pages) == 2
    assert all(isinstance(p, str) for p in pages) 