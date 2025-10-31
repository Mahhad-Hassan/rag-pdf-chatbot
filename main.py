from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import uuid
import openai
from pinecone import Pinecone, ServerlessSpec
import PyPDF2
from io import BytesIO
from typing import List
import numpy as np
from pydantic import BaseModel
from fastapi import Request
import uvicorn

# Environment variables
OPENAI_API_KEY = "Use Valid API"
PINECONE_API_KEY = "Use Valid Api"
INDEX_NAME = "uno-rag"
ENVIRONMENT = "us-east-1"

# Initialize OpenAI
openai.api_key = OPENAI_API_KEY

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create or connect to index
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,  # OpenAI ada-002 embedding dimensions
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region=ENVIRONMENT
        )
    )

index = pc.Index(INDEX_NAME)

app = FastAPI(title="RAG System API")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Mount static files (if you have any CSS/JS files)
# app.mount("/static", StaticFiles(directory="static"), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    user_id: str

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the frontend HTML page"""
    return templates.TemplateResponse("index.html", {"request": request})

def extract_text_from_pdf(pdf_file: bytes) -> str:
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error extracting text from PDF: {str(e)}")

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into chunks for better embeddings"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings using OpenAI ada-002 model"""
    try:
        response = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=texts
        )
        return [embedding.embedding for embedding in response.data]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")

def query_embeddings(query_text: str, user_id: str, top_k: int = 5) -> List[dict]:
    """Query Pinecone for similar embeddings"""
    try:
        # Generate embedding for query
        query_embedding = get_embeddings([query_text])[0]
        
        # Query Pinecone with user_id filter
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            filter={"user_id": user_id},
            include_metadata=True
        )
        
        return results.matches
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying embeddings: {str(e)}")

def generate_contextual_answer(query: str, context: str, similarity_threshold: float = 0.5) -> str:
    """Generate answer using OpenAI GPT with enhanced prompt for better accuracy"""
    try:
        # Enhanced and more accurate prompt
        system_prompt = """You are an intelligent AI assistant that answers questions based ONLY on the provided document context.

CRITICAL INSTRUCTIONS:
1. If the provided context contains clear information to answer the question, provide a detailed and accurate response
2. Use exact words and phrases from the context when answering
3. If the context does not contain the answer, respond: "I apologize, but your uploaded documents do not contain information about [topic]. Please upload relevant documents or ask questions related to the content in your uploaded documents."
4. Only use information from the provided context - do not add any external knowledge
5. If context is partially relevant, mention what information is available and what is missing
6. Cite relevant parts of the context in your answer
7. If the answer comes from multiple documents, mention that
8. Use exact facts, numbers, names, and dates from the context
9. If context contains contradictory information, present both perspectives
10. Be very precise and stick strictly to what is written in the context
11. If specific details are asked but only general information is available, clearly state the limitation

Context Information:
{context}

User Question: {query}

Instructions for Response:
- Be specific and cite relevant parts from the context
- Use bullet points for multiple pieces of information if needed
- If context contains examples, include them in your response
- Maintain professional and helpful tone
- Only answer what can be directly supported by the context"""

        gpt_response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt.format(context=context, query=query)},
                {"role": "user", "content": f"Based on the provided context, please answer: {query}"}
            ],
            max_tokens=800,
            temperature=0.1  # Lower temperature for more accurate responses
        )
        
        return gpt_response.choices[0].message.content
    except Exception as e:
        return f"Error generating AI response: {str(e)}"

@app.post("/upload-document")
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Form(...)
):
    """Upload PDF document and store embeddings in Pinecone"""
    
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Read PDF content
        pdf_content = await file.read()
        
        # Extract text from PDF
        text = extract_text_from_pdf(pdf_content)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text found in PDF")
        
        # Split text into chunks
        chunks = chunk_text(text)
        
        # Generate embeddings for chunks
        embeddings = get_embeddings(chunks)
        
        # Prepare vectors for Pinecone
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vector_id = f"{user_id}_{uuid.uuid4().hex[:8]}_{i}"
            vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": {
                    "user_id": user_id,
                    "text": chunk,
                    "filename": file.filename,
                    "chunk_index": i
                }
            })
        
        # Store in Pinecone
        index.upsert(vectors=vectors)
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "Document uploaded and processed successfully",
                "user_id": user_id,
                "filename": file.filename,
                "chunks_processed": len(chunks),
                "total_characters": len(text)
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/query")
async def query_documents(request: QueryRequest):
    """Query documents for a specific user with improved context awareness"""
    
    try:
        # Get similar chunks
        matches = query_embeddings(request.query, request.user_id, top_k=8)
        
        if not matches:
            return JSONResponse(
                status_code=200,
                content={
                    "query": request.query,
                    "user_id": request.user_id,
                    "ai_response": f"I apologize, but I couldn't find any documents for user ID '{request.user_id}'. Please make sure you have uploaded documents and are using the correct user ID.",
                    "source_chunks": [],
                    "total_matches": 0,
                    "message": "No documents found for this user"
                }
            )
        
        # Filter matches by similarity threshold
        similarity_threshold = 0.6
        relevant_matches = [match for match in matches if match.score >= similarity_threshold]
        
        # Format results
        results = []
        context_texts = []
        
        for match in matches[:5]:  # Use top 5 for context
            chunk_text = match.metadata.get("text", "")
            context_texts.append(chunk_text)
            results.append({
                "text": chunk_text,
                "filename": match.metadata.get("filename", ""),
                "similarity_score": match.score,
                "chunk_index": match.metadata.get("chunk_index", 0)
            })
        
        # Generate context-aware response
        full_context = "\n\n".join(context_texts)
        ai_response = generate_contextual_answer(request.query, full_context, similarity_threshold)
        
        # Determine response quality
        has_relevant_info = len(relevant_matches) > 0 and not ai_response.startswith("I apologize, but your uploaded documents do not contain")
        
        return JSONResponse(
            status_code=200,
            content={
                "query": request.query,
                "user_id": request.user_id,
                "ai_response": ai_response,
                "source_chunks": results,
                "total_matches": len(matches),
                "relevant_matches": len(relevant_matches),
                "has_relevant_info": has_relevant_info,
                "similarity_threshold": similarity_threshold,
                "message": "Query processed successfully"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying documents: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "message": "RAG API is running",
        "openai_configured": bool(OPENAI_API_KEY),
        "pinecone_configured": bool(PINECONE_API_KEY)
    }

@app.get("/user-documents/{user_id}")
async def get_user_documents(user_id: str):
    """Get list of documents for a user"""
    try:
        # Query Pinecone to get all vectors for this user
        results = index.query(
            vector=[0.0] * 1536,  # Dummy vector
            top_k=10000,  # Large number to get all
            filter={"user_id": user_id},
            include_metadata=True
        )
        
        # Extract unique filenames and count chunks
        documents = {}
        for match in results.matches:
            filename = match.metadata.get("filename", "Unknown")
            if filename not in documents:
                documents[filename] = {
                    "filename": filename,
                    "chunks": 0,
                    "last_updated": match.metadata.get("created_at", "Unknown")
                }
            documents[filename]["chunks"] += 1
        
        return JSONResponse(
            status_code=200,
            content={
                "user_id": user_id,
                "documents": list(documents.values()),
                "total_documents": len(documents),
                "total_chunks": len(results.matches)
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching user documents: {str(e)}")

@app.delete("/user-documents/{user_id}")
async def delete_user_documents(user_id: str):
    """Delete all documents for a user"""
    try:
        # Get all vector IDs for this user
        results = index.query(
            vector=[0.0] * 1536,  # Dummy vector
            top_k=10000,
            filter={"user_id": user_id},
            include_metadata=False
        )
        
        if not results.matches:
            return JSONResponse(
                status_code=404,
                content={"message": f"No documents found for user ID: {user_id}"}
            )
        
        # Delete vectors
        vector_ids = [match.id for match in results.matches]
        index.delete(ids=vector_ids)
        
        return JSONResponse(
            status_code=200,
            content={
                "message": f"Successfully deleted all documents for user {user_id}",
                "deleted_chunks": len(vector_ids)
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting user documents: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
