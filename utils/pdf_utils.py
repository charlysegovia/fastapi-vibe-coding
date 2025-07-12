from PyPDF2 import PdfReader
from typing import List
import io

def extract_text_by_page(pdf_bytes: bytes) -> List[str]:
    """Extracts the text of each page of a PDF as a list of strings."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return [page.extract_text() or "" for page in reader.pages] 