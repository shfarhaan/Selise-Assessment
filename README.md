# Agentic RAG System for Domain Knowledge QA

A sophisticated mini-RAG system with agentic capabilities for accurate, grounded domain knowledge question answering. The system combines document retrieval, self-reflection, and intelligent answer generation to minimize hallucinations.

## 🎯 System Architecture

```
┌─────────────────┐
│  User Query     │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│  Agentic RAG System         │
│  ┌─────────────────────┐    │
│  │ Tool: Retriever     │    │
│  ├─────────────────────┤    │
│  │ Critic: Evaluator   │    │
│  ├─────────────────────┤    │
│  │ Generator: LLM      │    │
│  └─────────────────────┘    │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────┐
│  Grounded       │
│  Answer with    │
│  Citations      │
└─────────────────┘
```

## 📋 Features

### 1. **Document Processing**
- Intelligent document chunking with overlap
- Support for multiple formats (.txt, .md, .pdf)
- Metadata preservation for source tracking

### 2. **Embedding & Vector Storage**
- Gemini text embeddings (`models/text-embedding-004`)
- FAISS vector database for efficient similarity search
- Persistent storage and loading of vectors

### 3. **Retrieval Logic**
- Context-aware document retrieval
- Similarity ranking and filtering
- Formatted context extraction

### 4. **Agentic Components**
- **Tool Use**: Document retriever as callable tool
- **Self-Reflection**: Critic evaluates document relevance and coverage
- **Answer Generation**: LLM generates grounded responses with citations
- **Iterative Refinement**: Multi-iteration retrieval with query refinement

### 5. **Streamlit UI**
- Intuitive chat interface
- Document processing pipeline
- System information dashboard
- Conversation history with metadata

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Gemini API key

### Installation

1. **Clone/Create the project:**
```bash
cd "f:\Selise Assessment"
```

2. **Set up virtual environment and install dependencies:**

**Setup Instructions**
```bash
# Step 1: Create virtual environment
python -m venv venv

# Step 2: Activate it
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Windows Command Prompt:
venv\Scripts\activate.bat
# Mac/Linux:
source venv/bin/activate

# Step 3: You should see (venv) in your prompt
# Step 4: Install dependencies
pip install -r requirements.txt

# Step 5: Verify setup
python setup.py
```

> **💡 Important**: Always activate the virtual environment before running the application!

3. **Set up API credentials:**

Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your-gemini-api-key-here
```

Or configure Streamlit secrets:
```bash
mkdir .streamlit
# Create .streamlit/secrets.toml with GEMINI_API_KEY
```

### Running the System

The system provides **three interfaces** for interaction:

#### 1. Streamlit Web UI (Recommended)

**Always activate your virtual environment first!**
```bash
# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Windows Command Prompt
venv\Scripts\activate.bat

# Mac/Linux
source venv/bin/activate

# You should see (venv) in your prompt
```

**Then start the Streamlit application:**
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

> **Tip**: If you get "command not found" errors, make sure your virtual environment is activated (look for `(venv)` in your prompt).

#### 2. Command-Line Interface (CLI)

For terminal-based interaction:

```bash
# Process documents
python cli.py process

# Start interactive chat
python cli.py chat

# Show system info
python cli.py info

# Help
python cli.py help
```

**CLI Features:**
- Colored terminal output
- Interactive chat loop
- Commands: `quit`, `clear`, `history`
- Same RAG agent as Streamlit

#### 3. REST API

For programmatic access and integration:

**Start the API server:**
```bash
python api.py
```

Server will start at `http://localhost:8000`

**API Documentation:** `http://localhost:8000/docs` (interactive Swagger UI)

**Available Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information and endpoints |
| GET | `/health` | Health check |
| GET | `/info` | System information and status |
| POST | `/chat` | Ask a question |
| POST | `/process` | Process documents |
| POST | `/upload` | Upload and save documents |
| POST | `/clear` | Clear conversation history |
| DELETE | `/cleanup` | Remove uploaded documents |

**Example API Usage:**

```bash
# Check health
curl http://localhost:8000/health

# Get system info
curl http://localhost:8000/info

# Upload documents
curl -X POST "http://localhost:8000/upload" \
  -F "files=@document1.txt" \
  -F "files=@document2.pdf"

# Process documents
curl -X POST "http://localhost:8000/process" \
  -H "Content-Type: application/json" \
  -d '{"chunk_size": 1000, "chunk_overlap": 200}'

# Ask a question
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is machine learning?"}'

# Clean up documents
curl -X DELETE "http://localhost:8000/cleanup"
```

**Python Client Example:**

```python
import requests

# Base URL
API_URL = "http://localhost:8000"

# Ask a question
response = requests.post(
    f"{API_URL}/chat",
    json={"question": "What is RAG?"}
)

result = response.json()
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
print(f"Confidence: {result['confidence']}%")
```

## 📚 Usage Guide

### Step 1: Process Documents

1. Navigate to the **"Process Documents"** tab
2. Configure chunking parameters (optional):
   - Chunk Size: 1000 (default)
   - Chunk Overlap: 200 (default)
3. Upload .txt or .md files containing your domain knowledge
4. Click **"Process Documents"** button

The system will:
- Load and parse documents
- Split into chunks with overlap
- Generate embeddings using Google's API
- Store in FAISS vector database

### Step 2: Ask Questions

1. Navigate to the **"Ask Questions"** tab
2. Enter your question in the chat input
3. The agent will:
   - Retrieve relevant documents
   - Evaluate document relevance (critic)
   - Generate a grounded answer with citations

### Step 3: Monitor Performance

- View **Confidence Score**: How confident is the system in this answer?
- View **Iterations**: How many retrieval attempts were made?
- Check **Metadata**: See which documents were used

## 🏗️ Project Structure

```
f:\Selise Assessment/
├── src/
│   ├── config.py              # Configuration settings
│   ├── document_processor.py  # Document loading and chunking
│   ├── embeddings.py          # Embedding generation and FAISS storage
│   ├── retriever.py           # Retrieval logic
│   └── agent.py               # Agentic RAG with reflection
├── documents/                 # Your domain documents
│   ├── ml_fundamentals.txt
│   └── rag_guide.txt
├── vector_store/              # FAISS index and metadata
├── app.py                     # Streamlit UI
├── requirements.txt           # Python dependencies
├── .env                       # API keys (not in git)
└── README.md                  # This file
```

## 🔧 Configuration

Edit `src/config.py` to customize:

```python
# Model Configuration
GEMINI_API_KEY = "your-key"                     # Gemini API key
CHAT_MODEL_NAME = "gemini-1.5-flash"            # Chat model
EMBEDDING_MODEL_NAME = "models/text-embedding-004"  # Embedding model
TEMPERATURE = 0.3                                # Response temperature

# Chunking
CHUNK_SIZE = 1000              # Size of each chunk
CHUNK_OVERLAP = 200            # Overlap between chunks

# Retrieval
TOP_K_DOCUMENTS = 5            # Number of documents to retrieve
SIMILARITY_THRESHOLD = 0.3     # Minimum relevance score

# Paths
DOCUMENTS_PATH = "documents"
VECTOR_STORE_PATH = "vector_store/faiss_index"
```

## 🤖 Agentic RAG Workflow

### Agent Loop

1. **Initial Request**: User asks a question
2. **Retrieval Tool**: Agent calls retriever to get relevant documents
3. **Critic Evaluation**: Evaluates relevance and coverage of retrieved docs
4. **Reflection**: Checks if retrieval quality is sufficient
5. **Query Refinement** (if needed): Refine query and retry retrieval
6. **Answer Generation**: Generate final answer grounded in documents
7. **Response**: Return answer with confidence score and citations

### Key Algorithms

**Embedding Generation:**
- Uses Gemini `models/text-embedding-004`
- Dense vector representations of text for semantic retrieval

**Similarity Search:**
- FAISS L2 distance metric
- Top-K retrieval with ranking
- Distance to similarity conversion: `similarity = 1 / (1 + distance)`

**Critic Evaluation:**
- Evaluates relevance, coverage, and confidence
- Recommends re-retrieval if quality is low
- Structured JSON output

## 🎯 Minimizing Hallucinations

The system minimizes hallucinations through:

1. **Grounding**: All answers based on retrieved documents
2. **Explicit Acknowledgment**: States when information is unavailable
3. **Source Citations**: References which documents were used
4. **Confidence Scoring**: Shows how confident the system is
5. **Threshold Filtering**: Ignores low-relevance documents
6. **Critic Evaluation**: Validates retrieved document quality
7. **Iterative Refinement**: Multiple retrieval attempts if needed

## 📊 Example Queries

With the provided sample documents, try:

**On ML Fundamentals:**
- "What are the different types of machine learning?"
- "How do we prevent overfitting in machine learning?"
- "Explain the difference between precision and recall"

**On RAG Systems:**
- "What are the advantages of RAG systems?"
- "How do vector databases improve retrieval?"
- "What is the difference between FAISS and Pinecone?"

**Cross-domain:**
- "Compare machine learning and RAG systems"
- "How do embeddings work in RAG?"

## 🔍 Troubleshooting

### Issue: Import warnings in VS Code (e.g., "Import 'streamlit' could not be resolved")
- **Cause**: Packages not yet installed
- **Solution**: 
  ```bash
  # Install dependencies
  pip install -r requirements.txt
  
  # Or verify setup
  python setup.py
  ```
- These warnings will disappear after installation

### Issue: "Gemini API key not found"
- **Solution**: Set your API key in `.env` file or Streamlit secrets
- Check `GEMINI_API_KEY` is properly set

### Issue: "No relevant documents found"
- **Solution**: Process documents first in the "Process Documents" tab

### Issue: "Vector store files not found"
- **Solution**: Process documents to create the vector store

### Issue: Slow embeddings generation
- **Solution**: This is expected for large document sets. The API calls can take time.

### Issue: Out of memory with FAISS
- **Solution**: Reduce chunk size or use vector database service (Pinecone, Qdrant)

## 🚀 Performance Optimization

### For Large Document Sets

1. **Use external vector database:**
```python
# Replace FAISS with Qdrant:
from qdrant_client import QdrantClient
vector_store = QdrantClient(":memory:")
```

2. **Implement batch processing:**
```python
embeddings = embedding_manager.embed_batch(texts, batch_size=50)
```

3. **Use caching:**
```python
@st.cache_resource
def initialize_rag_system():
    ...
```

## 📈 Evaluation Metrics

The system tracks:
- **Relevance Score**: 0-100% relevance of retrieved docs
- **Coverage Score**: 0-100% how well docs cover the query
- **Confidence Score**: 0-100% confidence in the answer
- **Iterations**: Number of retrieval attempts needed

## 🔐 Security Considerations

- Store API keys in `.env` or Streamlit secrets, never in code
- Validate and sanitize user inputs
- Rate-limit API calls for production
- Implement authentication for sensitive deployments

## 📚 Further Reading

### RAG Systems
- [The Power of Semantic Search](https://arxiv.org/abs/2004.04906)
- [In-Context Retrieval-Augmented Language Models](https://arxiv.org/abs/2302.07402)
- [REALM: Retrieval-Augmented Language Model Pre-Training](https://arxiv.org/abs/2002.07016)

### Vector Databases
- [FAISS: A Library for Efficient Similarity Search](https://arxiv.org/abs/1702.08734)
- [Qdrant Vector Search Engine Documentation](https://qdrant.tech/documentation/)

### Large Language Models
- [Gemini API Documentation](https://ai.google.dev/gemini-api/docs)
- [LangChain Documentation](https://python.langchain.com/)

## 🤝 Contributing

Suggestions for improvements:
1. Add support for more document formats
2. Implement ensemble retrieval
3. Add support for multiple embedding models
4. Implement caching for embeddings
5. Add metrics dashboard
6. Support for images in documents

## 📄 License

This project is open source and available under the MIT License.

## 🙋 Support

For issues or questions:
1. Check the Troubleshooting section
2. Review logs in the terminal
3. Ensure API key is configured correctly
4. Verify documents are processed first

## 📝 Notes

- The system uses synchronous API calls. For production, consider async implementation.
- FAISS is suitable for <100M vectors. For larger scales, use professional services.
- Embedding generation has API rate limits. Monitor your usage.
- The critic evaluation requires additional API calls. Budget accordingly.

---

**Built with**: Gemini API • FAISS • Streamlit • Python

**Last Updated**: February 2026
