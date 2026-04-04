"""
Configuration settings for the Agentic RAG System
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Model Configuration
CHAT_MODEL_NAME = os.getenv("GEMINI_CHAT_MODEL", "gemini-1.5-flash")
EMBEDDING_MODEL_NAME = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
TEMPERATURE = 0.3

# Chunking Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Vector Store Configuration
VECTOR_STORE_PATH = "vector_store/faiss_index"
EMBEDDINGS_MODEL = "gemini"

# Document Configuration
DOCUMENTS_PATH = "documents"
SUPPORTED_FORMATS = [".txt", ".pdf", ".md"]

# Agent Configuration
MAX_RETRIEVAL_ATTEMPTS = 3
TOP_K_DOCUMENTS = 5
SIMILARITY_THRESHOLD = 0.3

# Streamlit Configuration
STREAMLIT_PAGE_TITLE = "Agentic RAG System for Domain Knowledge QA"
STREAMLIT_PAGE_ICON = "🤖"
