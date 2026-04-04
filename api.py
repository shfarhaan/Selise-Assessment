#!/usr/bin/env python
"""
REST API for Agentic RAG System
Simple FastAPI implementation for HTTP-based access
"""
import os
import logging
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import (
    GEMINI_API_KEY, VECTOR_STORE_PATH, DOCUMENTS_PATH,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_DOCUMENTS,
    CHAT_MODEL_NAME, EMBEDDING_MODEL_NAME
)
from src.document_processor import DocumentProcessor
from src.embeddings import EmbeddingManager, FAISSVectorStore
from src.retriever import RAGRetriever
from src.agent import AgenticRAG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Agentic RAG API",
    description="REST API for Domain Knowledge QA using Agentic RAG",
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

# Global agent instance (initialized on first use)
rag_agent: Optional[AgenticRAG] = None


# Request/Response Models
class ChatRequest(BaseModel):
    question: str
    max_iterations: Optional[int] = 3


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: int
    iterations: int
    reasoning_steps: List[dict]


class ProcessRequest(BaseModel):
    chunk_size: Optional[int] = CHUNK_SIZE
    chunk_overlap: Optional[int] = CHUNK_OVERLAP


class ProcessResponse(BaseModel):
    success: bool
    message: str
    chunks_created: int
    documents_processed: int


class InfoResponse(BaseModel):
    status: str
    chat_model: str
    embedding_model: str
    vector_store_ready: bool
    documents_count: int
    chunk_size: int
    chunk_overlap: int
    top_k: int


def initialize_agent() -> AgenticRAG:
    """Initialize RAG agent (cached)"""
    global rag_agent
    
    if rag_agent is not None:
        return rag_agent
    
    try:
        # Check credentials
        if not GEMINI_API_KEY:
            raise ValueError("Gemini API key not set")
        
        # Initialize embedding manager
        embedding_manager = EmbeddingManager(
            api_key=GEMINI_API_KEY,
            model_name=EMBEDDING_MODEL_NAME
        )
        
        # Load vector store
        vector_store = FAISSVectorStore(vector_store_path=VECTOR_STORE_PATH)
        if not vector_store.load():
            raise ValueError("Vector store not found. Process documents first.")
        
        # Initialize retriever
        retriever = RAGRetriever(
            embedding_manager=embedding_manager,
            vector_store=vector_store,
            top_k=TOP_K_DOCUMENTS
        )
        
        # Initialize agent
        agent = AgenticRAG(
            api_key=GEMINI_API_KEY,
            retriever=retriever,
            model_name=CHAT_MODEL_NAME
        )
        
        rag_agent = agent
        logger.info("RAG agent initialized successfully")
        return agent
    
    except Exception as e:
        logger.error(f"Error initializing agent: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize RAG agent: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Agentic RAG API",
        "version": "1.0.0",
        "endpoints": {
            "GET /": "This help message",
            "GET /health": "Health check",
            "GET /info": "System information",
            "POST /chat": "Ask a question",
            "POST /process": "Process documents",
            "POST /upload": "Upload and process documents",
            "POST /clear": "Clear conversation history",
            "DELETE /cleanup": "Remove uploaded documents"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/info", response_model=InfoResponse)
async def get_info():
    """Get system information"""
    try:
        # Check vector store
        vector_store_exists = (
            os.path.exists(VECTOR_STORE_PATH)
            and os.path.exists(os.path.join(VECTOR_STORE_PATH, "faiss.index"))
            and os.path.exists(os.path.join(VECTOR_STORE_PATH, "metadata.pkl"))
        )
        
        # Count documents
        doc_count = 0
        if os.path.exists(DOCUMENTS_PATH):
            doc_count = len([f for f in os.listdir(DOCUMENTS_PATH) 
                           if f.endswith(('.txt', '.md', '.pdf'))])
        
        return InfoResponse(
            status="ready" if vector_store_exists else "not_initialized",
            chat_model=CHAT_MODEL_NAME,
            embedding_model=EMBEDDING_MODEL_NAME,
            vector_store_ready=vector_store_exists,
            documents_count=doc_count,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            top_k=TOP_K_DOCUMENTS
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Ask a question and get an answer
    
    - **question**: Your question about the documents
    - **max_iterations**: Maximum reasoning iterations (default: 3)
    """
    try:
        agent = initialize_agent()
        
        # Process query
        result = agent.reason(request.question)
        
        return ChatResponse(
            answer=result["answer"],
            sources=result.get("source_documents", []),
            confidence=result.get("confidence", 0),
            iterations=result.get("iterations", 0),
            reasoning_steps=result.get("reasoning_steps", [])
        )
    
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process", response_model=ProcessResponse)
async def process_documents(request: ProcessRequest):
    """
    Process documents from the documents folder
    
    - **chunk_size**: Size of each text chunk (default: 1000)
    - **chunk_overlap**: Overlap between chunks (default: 200)
    """
    try:
        global rag_agent
        
        # Check credentials
        if not GEMINI_API_KEY:
            raise HTTPException(status_code=500, detail="Gemini API key not set")
        
        # Check documents directory
        if not os.path.exists(DOCUMENTS_PATH):
            raise HTTPException(status_code=404, detail=f"Documents directory not found: {DOCUMENTS_PATH}")
        
        # Validate chunking configuration to prevent infinite loops / invalid slicing
        if request.chunk_size is None or request.chunk_overlap is None:
            raise HTTPException(status_code=400, detail="chunk_size and chunk_overlap are required")
        if request.chunk_size < 1:
            raise HTTPException(status_code=400, detail="chunk_size must be >= 1")
        if request.chunk_overlap < 0:
            raise HTTPException(status_code=400, detail="chunk_overlap must be >= 0")
        if request.chunk_overlap >= request.chunk_size:
            raise HTTPException(status_code=400, detail="chunk_overlap must be smaller than chunk_size")

        # Process documents
        processor = DocumentProcessor(
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap
        )
        chunks = processor.process_documents(DOCUMENTS_PATH)
        
        if not chunks:
            raise HTTPException(status_code=404, detail="No documents found to process")
        
        # Count unique documents
        unique_docs = len(set(chunk["source"] for chunk in chunks))
        
        # Generate embeddings
        embedding_manager = EmbeddingManager(
            api_key=GEMINI_API_KEY,
            model_name=EMBEDDING_MODEL_NAME
        )
        
        texts = [chunk["content"] for chunk in chunks]
        embeddings = embedding_manager.embed_batch(texts)
        
        # Store in vector database
        vector_store = FAISSVectorStore(vector_store_path=VECTOR_STORE_PATH)
        vector_store.add_documents(chunks, embeddings)
        vector_store.save()
        
        # Clear agent cache to reload
        rag_agent = None
        
        logger.info(f"Processed {unique_docs} documents into {len(chunks)} chunks")
        
        return ProcessResponse(
            success=True,
            message="Documents processed successfully",
            chunks_created=len(chunks),
            documents_processed=unique_docs
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload documents and process them
    
    - **files**: List of files to upload (.txt, .md, .pdf)
    """
    try:
        # Create documents directory if not exists
        Path(DOCUMENTS_PATH).mkdir(exist_ok=True)
        
        uploaded_files = []
        for file in files:
            # Check file type
            if not file.filename.endswith(('.txt', '.md', '.pdf')):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file type: {file.filename}. Only .txt, .md, .pdf supported"
                )
            
            # Save file
            file_path = os.path.join(DOCUMENTS_PATH, file.filename)
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            uploaded_files.append(file.filename)
            logger.info(f"Uploaded: {file.filename}")
        
        return {
            "success": True,
            "message": f"Uploaded {len(uploaded_files)} files",
            "files": uploaded_files,
            "next_step": "Call POST /process to process the uploaded documents"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear")
async def clear_history():
    """Clear conversation history"""
    try:
        if rag_agent is not None:
            rag_agent.clear_history()
            return {"success": True, "message": "Conversation history cleared"}
        return {"success": True, "message": "No active conversation"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/cleanup")
async def cleanup_documents():
    """Remove all uploaded documents from disk"""
    try:
        processor = DocumentProcessor()
        success = processor.cleanup_documents(DOCUMENTS_PATH)
        
        if success:
            return {
                "success": True,
                "message": "Uploaded documents removed from disk"
            }
        else:
            return {
                "success": False,
                "message": "No documents found to clean up"
            }
    
    except Exception as e:
        logger.error(f"Error cleaning up documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║     Agentic RAG API Server                               ║
    ║     Starting on http://localhost:8001                    ║
    ╚═══════════════════════════════════════════════════════════╝
    
    API Documentation: http://localhost:8001/docs
    """)
    
    uvicorn.run(app, host="0.0.0.0", port=8001)
