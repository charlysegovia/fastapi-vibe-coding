# fastapi-vibe-coding
This repo is to learn how to vibe code with Cursor

## Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd fastapi-vibe-coding
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

#### Option 1: Direct Python execution
```bash
python main.py
```

#### Option 2: Using uvicorn (recommended for development)
```bash
uvicorn main:app --reload
```

The server will start on `http://localhost:8000`

### Available Endpoints

- `GET /` - Hello World message
- `GET /hello/{name}` - Personalized greeting
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation (Swagger UI)

### Example Usage

```bash
# Hello World
curl http://localhost:8000/

# Personalized greeting
curl http://localhost:8000/hello/YourName

# Health check
curl http://localhost:8000/health
```

### API Documentation

Once the server is running, visit `http://localhost:8000/docs` to see the interactive API documentation powered by Swagger UI.
