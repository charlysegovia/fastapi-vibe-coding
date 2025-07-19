# GitHub RAG System

A Retrieval-Augmented Generation (RAG) system that processes GitHub repositories with dual indexing (dense + sparse) using FastAPI, Milvus, and OpenAI.

## Features

- **GitHub Integration**: Connect to your GitHub account and process repositories
- **Dual Indexing**: 
  - **Dense embeddings** (OpenAI) for semantic search
  - **Sparse embeddings** (TF-IDF) for keyword-based search
- **Hybrid Search**: Combines results from both dense and sparse collections
- **Smart Chunking**: Multiple strategies for different types of content
- **Web Interface**: Modern UI for repository management and querying
- **Real-time Processing**: Process repositories with progress tracking

## Architecture

### Dual Indexing Strategy

1. **Dense Chunks** (Semantic Search):
   - Paragraph-based chunking
   - Sentence-based chunking
   - OpenAI embeddings (text-embedding-ada-002)

2. **Sparse Chunks** (Keyword Search):
   - File-based chunking (entire files)
   - Section-based chunking (large sections)
   - TF-IDF vectors

### File Processing

- Supports all text-based files (code, documentation, config files)
- Automatic language detection
- File size limits (configurable, default 1MB)
- Binary file filtering

## Project Structure

```
fastapi-vibe-coding/
├── load_data.py              # Main orchestration service
├── github_service.py         # GitHub API integration
├── file_processor.py         # File validation and filtering
├── chunking_strategies.py    # Dual chunking strategies
├── milvus_dual.py           # Dual Milvus collections
├── main.py                  # FastAPI application
├── routers/
│   ├── github_routes.py     # GitHub repository endpoints
│   └── query_routes.py      # Query and search endpoints
├── services/
│   ├── embedding_service.py # OpenAI embeddings
│   └── milvus_service.py    # Legacy Milvus service
├── schemas/
│   └── query.py            # Pydantic models
├── static/
│   └── index.html          # Web interface
└── tests/                  # Test files
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd fastapi-vibe-coding
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

## Environment Variables

Create a `.env` file with the following variables:

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Milvus Configuration (Zilliz Cloud)
MILVUS_URI=https://your-instance.api.zillizcloud.com
MILVUS_TOKEN=your_milvus_token_here

# GitHub Configuration
GITHUB_TOKEN=your_github_personal_access_token
GITHUB_USERNAME=your_github_username

# File Processing Configuration
MAX_FILE_SIZE=1048576  # 1MB in bytes
```

### GitHub Token Setup

1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Generate a new token with the following scopes:
   - `repo` (for private repositories)
   - `read:user` (for user information)
   - `read:org` (for organization repositories)

## Running the Application

1. **Start the server**:
   ```bash
   uvicorn main:app --reload
   ```

2. **Access the web interface**:
   - Open your browser and go to: http://localhost:8000/static/index.html

## API Endpoints

### GitHub Management

- `GET /api/repos` - List all repositories
- `GET /api/repo/{repo_name}/files` - List files in a repository
- `POST /api/process/{repo_name}` - Process a repository
- `GET /api/process/{repo_name}/status` - Get processing status
- `GET /api/health` - System health check

### Query and Search

- `POST /api/query` - Ask questions about processed repositories
- `GET /api/search-stats` - Get system statistics

### Example Usage

#### List Repositories
```bash
curl http://localhost:8000/api/repos
```

#### Process a Repository
```bash
curl -X POST http://localhost:8000/api/process/my-repo-name
```

#### Ask a Question
```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How does the authentication work?",
    "top_k": 5
  }'
```

## Web Interface

The web interface provides:

1. **Repository Management**:
   - View all your GitHub repositories
   - Select repositories for processing

2. **Processing**:
   - Process repositories with progress tracking
   - View processing results and statistics

3. **Query Interface**:
   - Ask questions about your repositories
   - View detailed source information
   - See hybrid search results

4. **Statistics**:
   - System overview
   - Repository processing status
   - Chunk counts and metrics

## Chunking Strategies

### Dense Chunking (Semantic)
- **Paragraphs**: Split by double newlines
- **Sentences**: Group sentences for optimal size
- **Size limits**: 50-2000 characters
- **Purpose**: Semantic similarity search

### Sparse Chunking (Keyword)
- **Files**: Entire files as single chunks
- **Sections**: Large sections based on headers/code blocks
- **Purpose**: Keyword-based retrieval

## Search Algorithm

The system uses a hybrid approach:

1. **Dense Search**: Semantic similarity using OpenAI embeddings
2. **Sparse Search**: Keyword matching using TF-IDF vectors
3. **Result Combination**: Weighted combination of both results
   - Dense results: 70% weight
   - Sparse results: 30% weight

## Milvus Collections

### Dense Collection (`rag_documents_dense`)
- OpenAI embeddings (1536 dimensions)
- Semantic search capabilities
- Metadata: repo_name, file_path, author, etc.

### Sparse Collection (`rag_documents_sparse`)
- TF-IDF vectors (1000 dimensions)
- Keyword search capabilities
- Same metadata structure

## Supported File Types

### Code Files
- Python, JavaScript, TypeScript, Java, C++, C#, PHP, Ruby, Go, Rust, Swift, Kotlin

### Documentation
- Markdown, Text, reStructuredText, LaTeX

### Configuration
- JSON, XML, YAML, TOML, INI, Config files

### Scripts
- Shell scripts, Batch files, PowerShell, Dockerfiles

### Git Files
- .gitignore, .gitattributes, .editorconfig

## Configuration

### File Size Limits
- Default: 1MB per file
- Configurable via `MAX_FILE_SIZE` environment variable

### Chunking Parameters
- Minimum chunk size: 50 characters
- Maximum chunk size: 2000 characters
- Overlap size: 100 characters

### Search Parameters
- Default top_k: 5 results
- Maximum top_k: 20 results
- Dense weight: 70%
- Sparse weight: 30%

## Testing

Run the test suite:

```bash
pytest
```

## Performance Considerations

- **Large Repositories**: Processing time scales with repository size
- **API Limits**: GitHub API has rate limits (5000 requests/hour for authenticated users)
- **Memory Usage**: TF-IDF vectorizer loads vocabulary into memory
- **Milvus**: Ensure adequate resources for vector database

## Troubleshooting

### Common Issues

1. **GitHub Token Issues**:
   - Verify token has correct scopes
   - Check token expiration
   - Ensure username matches token owner

2. **Milvus Connection**:
   - Verify URI and token
   - Check network connectivity
   - Ensure collections are created

3. **OpenAI API**:
   - Verify API key
   - Check account credits
   - Monitor rate limits

### Logs

Application logs are written to `app.log` with detailed information about:
- Repository processing
- File operations
- Search queries
- Error details

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.
