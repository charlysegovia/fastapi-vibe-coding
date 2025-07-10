from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Optional, Union
import io
import PyPDF2
import docx
import os

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

# Global variable to store document content
document_content = ""
document_filename = ""


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
    """Upload and process a document."""
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
        
        return {
            "success": True,
            "message": f"Document '{file.filename}' uploaded successfully!",
            "filename": file.filename,
            "content_preview": extracted_text[:200] + "..." if len(extracted_text) > 200 else extracted_text
        }
        
    except Exception as e:
        return {"error": f"Error processing file: {str(e)}"}

@app.post("/ask")
async def ask_chatgpt(message: Dict[str, str]) -> Dict[str, str]:
    """ChatGPT endpoint that takes a message and returns a response based on uploaded document."""
    global document_content, document_filename
    
    user_message = message.get("message", "")
    if not user_message:
        return {"error": "No message provided"}
    
    if not document_content:
        return {"error": "No document uploaded. Please upload a document first."}
    
    # Create context-aware response based on document content
    context_response = generate_context_response(user_message, document_content, document_filename)
    
    return {"response": context_response}

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 