from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import os
import uuid
import warnings
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
import PyPDF2
from io import BytesIO
from typing import List
from pydantic import BaseModel
from fastapi import Request
import uvicorn

# Suppress the deprecation warning (optional)
warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")

# ========== CONFIGURATION ==========
GOOGLE_API_KEY = "AIzaSyA2ajZLcpxOz9ryXTmTnpQc2tuG5nurHq4"
PINECONE_API_KEY = "pcsk_5zRjdc_CSVkLthDtUvsfXLGEn8Xn9v9gtZoz4zXLJRqUJuHVJPBLFJCd2Zf9YmeKQpFY8X"
INDEX_NAME = "git"
ENVIRONMENT = "us-east-1"
EMBEDDING_DIM = 3072                    # Gemini embedding dimension
EMBEDDING_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-2.5-flash"   

# Initialize Google Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# ========== HANDLE INDEX DIMENSION MISMATCH ==========
if INDEX_NAME in pc.list_indexes().names():
    # Describe the existing index
    index_desc = pc.describe_index(INDEX_NAME)
    if index_desc.dimension != EMBEDDING_DIM:
        print(f"⚠️ Index '{INDEX_NAME}' has dimension {index_desc.dimension}, but we need {EMBEDDING_DIM}. Deleting and recreating...")
        pc.delete_index(INDEX_NAME)
        # Wait a few seconds for deletion to propagate
        import time
        time.sleep(5)
        # Create new index with correct dimension
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region=ENVIRONMENT)
        )
        print(f"✅ Index '{INDEX_NAME}' recreated with dimension {EMBEDDING_DIM}.")
else:
    # Create index if it doesn't exist
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIM,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region=ENVIRONMENT)
    )
    print(f"✅ Index '{INDEX_NAME}' created with dimension {EMBEDDING_DIM}.")

index = pc.Index(INDEX_NAME)

app = FastAPI(title="RAG System API")

# ========== TEMPLATES ==========
# Check if templates folder exists, otherwise create a simple fallback
if not os.path.exists("templates"):
    os.makedirs("templates")
    # Create a default index.html if missing
    with open("templates/index.html", "w") as f:
        f.write("""<!DOCTYPE html>
<html>
<head><title>RAG System</title></head>
<body>
<h1>Document Restricted RAG</h1>
<form id="uploadForm" enctype="multipart/form-data">
    <input type="text" id="userId" placeholder="User ID" required><br>
    <input type="file" id="pdfFile" accept=".pdf" required><br>
    <button type="submit">Upload PDF</button>
</form>
<div id="uploadResult"></div>
<hr>
<form id="queryForm">
    <input type="text" id="queryUserId" placeholder="User ID" required><br>
    <input type="text" id="queryText" placeholder="Ask a question" required><br>
    <button type="submit">Ask</button>
</form>
<div id="queryResult"></div>
<script>
    document.getElementById('uploadForm').onsubmit = async (e) => {
        e.preventDefault();
        const formData = new FormData();
        formData.append('user_id', document.getElementById('userId').value);
        formData.append('file', document.getElementById('pdfFile').files[0]);
        const res = await fetch('/upload-document', { method: 'POST', body: formData });
        const data = await res.json();
        document.getElementById('uploadResult').innerHTML = JSON.stringify(data);
    };
    document.getElementById('queryForm').onsubmit = async (e) => {
        e.preventDefault();
        const payload = {
            user_id: document.getElementById('queryUserId').value,
            query: document.getElementById('queryText').value
        };
        const res = await fetch('/query', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
        const data = await res.json();
        document.getElementById('queryResult').innerHTML = '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
    };
</script>
</body>
</html>""")

templates = Jinja2Templates(directory="templates")

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

# ========== HELPER FUNCTIONS ==========
def extract_text_from_pdf(pdf_file: bytes) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file))
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF extraction error: {str(e)}")

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def get_embeddings(texts: List[str]) -> List[List[float]]:
    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=texts,
            task_type="retrieval_document"
        )
        return result['embedding']
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding error: {str(e)}")

def generate_answer(query: str, context: str) -> str:
    prompt = f"""You are a strict assistant. Answer ONLY from the provided context.
If the context does NOT contain the answer, say exactly:
"I apologize, but your uploaded documents do not contain information about that. Please upload relevant documents."

Context:
{context}

Question: {query}

Answer (based only on context):"""
    try:
        model = genai.GenerativeModel(CHAT_MODEL)
        response = model.generate_content(prompt, generation_config={"temperature": 0.1})
        return response.text
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# ========== API ENDPOINTS ==========
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...), user_id: str = Form(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")
    try:
        pdf_content = await file.read()
        text = extract_text_from_pdf(pdf_content)
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text found in PDF")
        
        chunks = chunk_text(text)
        embeddings = get_embeddings(chunks)
        
        vectors = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            vector_id = f"{user_id}_{uuid.uuid4().hex[:8]}_{i}"
            vectors.append({
                "id": vector_id,
                "values": emb,
                "metadata": {
                    "user_id": user_id,
                    "text": chunk,
                    "filename": file.filename,
                    "chunk_index": i
                }
            })
        index.upsert(vectors=vectors)
        return JSONResponse(content={
            "message": "Upload successful",
            "user_id": user_id,
            "filename": file.filename,
            "chunks_processed": len(chunks)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

@app.post("/query")
async def query_documents(request: QueryRequest):
    try:
        query_emb = get_embeddings([request.query])[0]
        results = index.query(
            vector=query_emb,
            top_k=5,
            filter={"user_id": request.user_id},
            include_metadata=True
        )
        if not results.matches:
            return JSONResponse(content={
                "query": request.query,
                "ai_response": f"No documents found for user '{request.user_id}'.",
                "source_chunks": []
            })
        context_parts = []
        sources = []
        for match in results.matches:
            context_parts.append(match.metadata.get("text", ""))
            sources.append({
                "text": match.metadata.get("text", "")[:300],
                "filename": match.metadata.get("filename", ""),
                "score": match.score
            })
        full_context = "\n\n".join(context_parts)
        ai_response = generate_answer(request.query, full_context)
        return JSONResponse(content={
            "query": request.query,
            "user_id": request.user_id,
            "ai_response": ai_response,
            "source_chunks": sources,
            "total_matches": len(results.matches)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "embedding_dim": EMBEDDING_DIM}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)