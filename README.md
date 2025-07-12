# FastAPI RAG Example

This project implements a Retrieval-Augmented Generation (RAG) workflow using FastAPI, Milvus (Zilliz Cloud), and OpenAI.

## Project Structure
- Modular: routers/, services/, utils/, schemas/, types/, static/, tests/

## Requirements
- Python 3.9+
- OpenAI and Zilliz Cloud (Milvus) accounts

## Installation

1. Clone the repository and enter the directory:
   ```bash
   git clone <repo-url>
   cd fastapi-vibe-coding
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy the example environment file and edit it:
   ```bash
   cp .env.example .env
   # Edit .env with your keys
   ```

## Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key
- `MILVUS_URI`: Your Zilliz Cloud instance URL
- `MILVUS_TOKEN`: Milvus authentication token

## Running the App

```bash
uvicorn main:app --reload
```

## Web Interface

A simple graphical interface is available for uploading PDF files and querying the RAG system.

- Open your browser and go to: [http://localhost:8000/static/index.html](http://localhost:8000/static/index.html)
- You can upload a PDF and make queries directly from this page.

## Endpoints

### 1. Upload PDF Document
`POST /upload`
- Accepts only PDF files (max 50MB)
- Splits the PDF by page
- Generates embeddings and stores them in Milvus

#### Example (Python):
```python
import requests

with open('document.pdf', 'rb') as f:
    files = {'file': ('document.pdf', f, 'application/pdf')}
    resp = requests.post('http://localhost:8000/upload', files=files)
    print(resp.json())
```

### 2. Query (RAG)
`POST /query`
- Receives: question, document filename, top_k
- Returns: generated answer, source chunks (text, page, ids, filename)

#### Example (Python):
```python
import requests

payload = {
    'question': 'What is the summary of the document?',
    'filename': 'document.pdf',
    'top_k': 3
}
resp = requests.post('http://localhost:8000/query', json=payload)
print(resp.json())
```

## Milvus Notes
- The collection is created automatically if it does not exist.
- Embedding dimension: 1536 (OpenAI text-embedding-ada-002)

## Testing
```bash
pytest
```

## License
MIT
