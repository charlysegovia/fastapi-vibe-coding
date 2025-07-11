from fastapi import FastAPI, UploadFile, File, Form, Body
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Optional, Union, List
import io
import PyPDF2
import docx
import os

import openai
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import uuid

# Milvus Managed (Zilliz Cloud) credentials
MILVUS_URI = "https://in03-5cdb1040a1ef831.serverless.gcp-us-west1.cloud.zilliz.com" # e.g., "https://in01-xxxx.api.gcp-us-west1.zillizcloud.com"
MILVUS_TOKEN = "028c1197455d1c62eaf13bfc6a68036ab4e0b45bb9b940a4657e1958ed8dfda55ccb4e38706e14ee132fc37784f0d0d5ab362a94"
COLLECTION_NAME = "DataExpertLabTestVishal"
DIMENSION = 3270  # OpenAI Ada embedding dimension

# Read OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

app = FastAPI(
    title="FastAPI Vibe Coding, first program by Vishal",
    description="A simple FastAPI application for learning vibe coding with Cursor",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables
document_content = ""
document_filename = ""

# Milvus configuration
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "DataExpertLabTestVishal"
DIMENSION = 3270  # Dimension for sentence-transformers embeddings

# Initialize Milvus connection
def init_milvus():
    try:
        connections.connect(
            alias="default",
            uri=MILVUS_URI,
            token=MILVUS_TOKEN,
            secure=True
        )
        print("✅ Connected to Milvus Managed (Zilliz Cloud) with token successfully")
        
        # Create collection if it doesn't exist
        if not utility.has_collection(COLLECTION_NAME):
            create_collection()
        else:
            print(f"✅ Collection '{COLLECTION_NAME}' already exists")
            
    except Exception as e:
        print(f"❌ Error connecting to Milvus Managed: {e}")

def create_collection():
    """Create Milvus collection for document chunks."""
    try:
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="document_name", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION)
        ]
        
        schema = CollectionSchema(fields=fields, description="Document chunks with embeddings")
        collection = Collection(name=COLLECTION_NAME, schema=schema)
        
        # Create index for vector search
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        
        print(f"✅ Collection '{COLLECTION_NAME}' created successfully")
        
    except Exception as e:
        print(f"❌ Error creating collection: {e}")

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    words = text.split()
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks

def get_embedding(text: str) -> list:
    """Get embedding for text using OpenAI's embedding API and pad to DIMENSION."""
    if not OPENAI_API_KEY:
        print("❌ OpenAI API key not set. Set the OPENAI_API_KEY environment variable.")
        return []
    try:
        response = openai.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        emb = response.data[0].embedding
        if len(emb) < DIMENSION:
            emb = emb + [0.0] * (DIMENSION - len(emb))
        elif len(emb) > DIMENSION:
            emb = emb[:DIMENSION]
        return emb
    except Exception as e:
        print(f"❌ Error generating embedding with OpenAI: {e}")
        return []

def insert_document_chunks(document_text: str, document_name: str):
    """Insert document chunks into Milvus."""
    try:
        collection = Collection(COLLECTION_NAME)
        collection.load()
        
        # Chunk the document
        chunks = chunk_text(document_text)
        
        # Prepare data for insertion (each field is a list)
        ids = []
        docnames = []
        chunk_texts = []
        embeddings = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{document_name}_{i}_{uuid.uuid4().hex[:8]}"
            embedding = get_embedding(chunk)
            if embedding and len(embedding) == DIMENSION:
                ids.append(chunk_id)
                docnames.append(document_name)
                chunk_texts.append(chunk)
                embeddings.append(embedding)
            else:
                print(f"❌ Skipping chunk {i}: embedding failure or dimension mismatch (got {len(embedding) if embedding else 0})")
        
        if ids:
            data = [ids, docnames, chunk_texts, embeddings]
            collection.insert(data)
            collection.flush()
            print(f"✅ Inserted {len(ids)} chunks for document '{document_name}'")
        else:
            print("❌ No valid chunks were inserted (embedding failure or dimension mismatch)")
            
    except Exception as e:
        print(f"❌ Error inserting document chunks: {e}")

def search_similar_chunks(query: str, top_k: int = 3) -> List[Dict]:
    """Search for similar chunks in Milvus."""
    try:
        collection = Collection(COLLECTION_NAME)
        collection.load()
        
        # Get query embedding
        query_embedding = get_embedding(query)
        if not query_embedding:
            return []
        
        # Search for similar chunks
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["document_name", "chunk_text"]
        )
        
        similar_chunks = []
        for hits in results:
            for hit in hits:
                similar_chunks.append({
                    "document_name": hit.entity.get("document_name"),
                    "chunk_text": hit.entity.get("chunk_text"),
                    "score": hit.score
                })
        
        return similar_chunks
        
    except Exception as e:
        print(f"❌ Error searching similar chunks: {e}")
        return []


@app.get("/")
async def root() -> FileResponse:
    """Serve the index.html file."""
    return FileResponse("static/index.html")


@app.get("/api/hello")
async def hello_world() -> Dict[str, str]:
    """Hello world API endpoint."""
    return {"message": "Hello World!"}


@app.get("/hello/{name}")
async def hello_name(name: str) -> Dict[str, str]:
    """Personalized hello endpoint that takes a name parameter."""
    return {"message": f"Hello {name}!"}


def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF file."""
    try:
        pdf_file = io.BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        return f"Error extracting PDF text: {str(e)}"

def extract_text_from_docx(file_content: bytes) -> str:
    """Extract text from DOCX file."""
    try:
        doc_file = io.BytesIO(file_content)
        doc = docx.Document(doc_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        return f"Error extracting DOCX text: {str(e)}"

def extract_text_from_txt(file_content: bytes) -> str:
    """Extract text from TXT file."""
    try:
        return file_content.decode('utf-8').strip()
    except Exception as e:
        return f"Error extracting TXT text: {str(e)}"

@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)) -> Dict[str, Union[str, bool]]:
    """Upload and process a document, then add its chunks to Milvus."""
    global document_content, document_filename
    
    if not file.filename:
        return {"error": "No file provided"}
    
    # Check file type
    allowed_extensions = {'.pdf', '.docx', '.txt'}
    file_extension = os.path.splitext(file.filename.lower())[1]
    
    if file_extension not in allowed_extensions:
        return {"error": f"Unsupported file type. Please upload PDF, DOCX, or TXT files."}
    
    try:
        # Read file content
        content = await file.read()
        
        # Extract text based on file type
        if file_extension == '.pdf':
            extracted_text = extract_text_from_pdf(content)
        elif file_extension == '.docx':
            extracted_text = extract_text_from_docx(content)
        elif file_extension == '.txt':
            extracted_text = extract_text_from_txt(content)
        else:
            return {"error": "Unsupported file type"}
        
        if extracted_text.startswith("Error"):
            return {"error": extracted_text}
        
        # Store document content and filename
        document_content = extracted_text
        document_filename = file.filename
        
        # --- Milvus Insertion Logic ---
        try:
            insert_document_chunks(extracted_text, file.filename)
            rag_status = "Document chunks and embeddings inserted into Milvus."
        except Exception as e:
            rag_status = f"Failed to insert into Milvus: {str(e)}"
        # --- End Milvus Insertion Logic ---
        
        return {
            "success": True,
            "message": f"Document '{file.filename}' uploaded and processed!",
            "filename": file.filename,
            "content_preview": extracted_text[:200] + "..." if len(extracted_text) > 200 else extracted_text,
            "rag_status": rag_status
        }
        
    except Exception as e:
        return {"error": f"Error processing file: {str(e)}"}

@app.post("/ask")
async def ask_chatgpt(message: Dict[str, str]) -> Dict[str, str]:
    """RAG-enabled endpoint that takes a message and returns a response based on document chunks."""
    global document_content, document_filename
    
    user_message = message.get("message", "")
    if not user_message:
        return {"error": "No message provided"}
    
    if not document_content:
        return {"error": "No document uploaded. Please upload a document first."}
    
    # Use RAG to find relevant document chunks
    try:
        similar_chunks = search_similar_chunks(user_message, top_k=3)
        
        if similar_chunks:
            # Create context from relevant chunks
            context = "\n\n".join([chunk["chunk_text"] for chunk in similar_chunks])
            context_response = generate_rag_response(user_message, context, document_filename, similar_chunks)
        else:
            # Fallback to simple keyword matching
            context_response = generate_context_response(user_message, document_content, document_filename)
            
    except Exception as e:
        print(f"❌ RAG search failed: {e}")
        # Fallback to simple keyword matching
        context_response = generate_context_response(user_message, document_content, document_filename)
    
    return {"response": context_response}

@app.post("/add-document-string")
async def add_document_string(
    text: str = Body(..., embed=True),
    document_name: str = Body("manual_entry", embed=True)
) -> dict:
    """Add a string as a document, chunk, embed, and insert into Milvus for RAG."""
    if not text.strip():
        return {"error": "No text provided"}
    try:
        insert_document_chunks(text, document_name)
        return {"success": True, "message": f"Text added to RAG as document '{document_name}'"}
    except Exception as e:
        return {"error": f"Failed to add text to RAG: {str(e)}"}

def generate_rag_response(question: str, context: str, filename: str, similar_chunks: List[Dict]) -> str:
    """Generate a response using RAG with retrieved context."""
    
    # Create a more sophisticated response using the retrieved context
    question_lower = question.lower()
    
    # Build response based on retrieved chunks
    if similar_chunks:
        chunk_info = f"Based on the most relevant sections from '{filename}', here's what I found:\n\n"
        
        for i, chunk in enumerate(similar_chunks, 1):
            chunk_info += f"**Section {i}** (Relevance: {chunk['score']:.2f}):\n{chunk['chunk_text']}\n\n"
        
        # Generate contextual response
        if any(word in question_lower for word in ['summary', 'summarize', 'overview']):
            return f"{chunk_info}This provides a comprehensive overview of the relevant information in your document."
        
        elif any(word in question_lower for word in ['what', 'how', 'why', 'when', 'where']):
            return f"{chunk_info}These sections directly address your question about the document content."
        
        elif any(word in question_lower for word in ['find', 'search', 'locate']):
            return f"{chunk_info}I found these relevant sections that match your search criteria."
        
        else:
            return f"{chunk_info}This information should help answer your question about the document."
    
    else:
        return f"I couldn't find specific information related to your question in the document '{filename}'. Could you please rephrase or ask about a different aspect of the document?"

def generate_context_response(question: str, document_content: str, filename: str) -> str:
    """Generate a response based on document content and user question."""
    
    # Simple keyword-based response generation
    # In a real implementation, you would use OpenAI API here
    
    question_lower = question.lower()
    content_lower = document_content.lower()
    
    # Check if question contains common keywords
    if any(word in question_lower for word in ['summary', 'summarize', 'overview']):
        return f"Based on the document '{filename}', here's a summary: {document_content[:300]}..."
    
    elif any(word in question_lower for word in ['what', 'how', 'why', 'when', 'where']):
        # Look for relevant content in the document
        relevant_sections = []
        sentences = document_content.split('.')
        for sentence in sentences:
            if any(word in sentence.lower() for word in question_lower.split()):
                relevant_sections.append(sentence.strip())
        
        if relevant_sections:
            return f"Based on the document '{filename}', here's what I found: {' '.join(relevant_sections[:2])}..."
        else:
            return f"I've reviewed the document '{filename}', but I couldn't find specific information related to your question. Could you please rephrase or ask about a different aspect of the document?"
    
    elif any(word in question_lower for word in ['find', 'search', 'locate']):
        return f"I've searched through the document '{filename}' for relevant information. Here's what I found: {document_content[:250]}..."
    
    else:
        return f"Based on the document '{filename}', I can help you with questions about its content. The document contains information about various topics. What specific aspect would you like to know more about?"


@app.on_event("startup")
async def startup_event():
    """Initialize Milvus connection on startup."""
    init_milvus()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 