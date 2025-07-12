Create a FastAPI app that implements a simple Retrieval-Augmented Generation (RAG) workflow using the following components:

- Milvus as the vector database (Zilliz Cloud), using secure connection via token authentication with pymilvus
- OpenAI as the LLM provider
- FastAPI for the web API
- Pydantic v2 for input/output models
- Async operations wherever I/O is involved

The app must support the following endpoints:

1. POST /upload
    - Accept a document file (plain text or PDF for now)
    - Split the document into chunks (by paragraph or sentence)
    - Generate embeddings using OpenAI's embedding API
    - Store embeddings and metadata (doc_id, chunk_id, original text) in a Milvus collection

2. POST /query
    - Accept a question as input
    - Embed the question using the same OpenAI model
    - Perform vector similarity search in Milvus
    - Retrieve top-k chunks and inject them as context into a prompt
    - Use OpenAI chat completion API to answer the question using the retrieved context
    - Return the answer along with source chunk texts

3. Use async I/O (httpx, Milvus client, OpenAI if possible)

4. Follow CursorRules:
    - Use functional, modular file structure
    - Use Pydantic models
    - Add docstrings and inline comments in English
    - Follow best practices for error handling, request validation, and type hints
    - Use pyproject.toml to configure tooling (black, isort, ruff)

5. Use a .env file to manage all environment variables:
    - `OPENAI_API_KEY`
    - `MILVUS_URI` (e.g. https://xxx.api.zillizcloud.com)
    - `MILVUS_TOKEN`

6. Include a sample `.env.example` file for reference

7. Include a requirements.txt file with all necessary dependencies

8. Add a README.md with **clear step-by-step instructions** to:
    - Set up the environment
    - Install dependencies
    - Configure `.env` variables
    - Run the app locally
    - Upload documents
    - Ask questions via the query endpoint
