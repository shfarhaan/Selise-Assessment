"""
Streamlit UI for Agentic RAG System
"""
import streamlit as st
import os
import logging
from pathlib import Path
from datetime import datetime

from src.config import (
    GEMINI_API_KEY, VECTOR_STORE_PATH, DOCUMENTS_PATH,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_DOCUMENTS, TEMPERATURE,
    STREAMLIT_PAGE_TITLE, STREAMLIT_PAGE_ICON,
    CHAT_MODEL_NAME, EMBEDDING_MODEL_NAME
)
from src.document_processor import DocumentProcessor
from src.embeddings import EmbeddingManager, FAISSVectorStore
from src.retriever import RAGRetriever
from src.agent import AgenticRAG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def resolve_gemini_api_key() -> str:
    """Resolve Gemini API key from env first, then Streamlit secrets."""
    api_key = GEMINI_API_KEY or ""

    if api_key:
        return api_key

    try:
        api_key = api_key or st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        # st.secrets can raise when no secrets file is configured.
        pass

    return api_key

# Page configuration
st.set_page_config(
    page_title=STREAMLIT_PAGE_TITLE,
    page_icon=STREAMLIT_PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] button {
        height: 3em;
    }
    .reportview-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_rag_system():
    """Initialize RAG components (cached)"""
    try:
        # Check API key
        api_key = resolve_gemini_api_key()
        if not api_key:
            st.error("❌ Gemini API key not found. Please set GEMINI_API_KEY.")
            st.stop()
        
        # Initialize components
        with st.spinner("🔧 Initializing RAG System..."):
            # Embedding manager
            embedding_manager = EmbeddingManager(
                api_key=api_key,
                model_name=EMBEDDING_MODEL_NAME
            )
            
            # Vector store
            vector_store = FAISSVectorStore(vector_store_path=VECTOR_STORE_PATH)
            
            # Try to load existing vector store
            if not vector_store.load():
                st.info("📚 No existing vector store found. Please process documents first.")
                return None
            
            # Retriever
            retriever = RAGRetriever(
                embedding_manager=embedding_manager,
                vector_store=vector_store,
                top_k=TOP_K_DOCUMENTS
            )
            
            # Agent
            agent = AgenticRAG(
                api_key=api_key,
                retriever=retriever,
                model_name=CHAT_MODEL_NAME,
                temperature=TEMPERATURE
            )
            
            st.success("✅ RAG System initialized successfully!")
            return agent
    
    except Exception as e:
        st.error(f"❌ Error initializing RAG system: {str(e)}")
        logger.error(f"Initialization error: {str(e)}")
        return None


def process_documents_tab():
    """Tab for document processing"""
    st.header("📄 Document Processing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Configuration")
        chunk_size = st.slider("Chunk Size", min_value=250, max_value=2000, value=CHUNK_SIZE, step=250)
        chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=chunk_size-1, value=CHUNK_OVERLAP, step=50)
    
    with col2:
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload .txt, .md, .pdf files",
            type=["txt", "md", "pdf"],
            accept_multiple_files=True,
            help="Supported formats: .txt, .md, .pdf"
        )
    
    if uploaded_files:
        # Save uploaded files
        Path(DOCUMENTS_PATH).mkdir(exist_ok=True)
        
        st.subheader("Uploading Files...")
        for uploaded_file in uploaded_files:
            file_path = os.path.join(DOCUMENTS_PATH, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"✅ Saved {uploaded_file.name}")

    test_col1, test_col2 = st.columns([0.55, 0.45])
    with test_col1:
        if st.button("🧪 Test Gemini Connection", use_container_width=True):
            try:
                api_key = resolve_gemini_api_key()
                if not api_key:
                    st.error("❌ Gemini API key not found!")
                else:
                    with st.spinner("Testing Gemini connectivity..."):
                        tester = EmbeddingManager(
                            api_key=api_key,
                            model_name=EMBEDDING_MODEL_NAME
                        )
                        tester.test_connection()
                    st.success("✅ Gemini embedding connection is working")
            except Exception as e:
                st.error(f"❌ Connection test failed: {str(e)}")
                st.info(
                    "If this fails on Streamlit Cloud only, check internet egress and GEMINI_API_KEY configuration."
                )

    with test_col2:
        st.caption("Run this test before processing to avoid long retries when network access is blocked.")
    
    if st.button("🔄 Process Documents", type="primary", use_container_width=True):
        try:
            api_key = resolve_gemini_api_key()
            if not api_key:
                st.error("❌ Gemini API key not found!")
                return

            # Preflight Gemini connectivity before document chunking.
            with st.spinner("🔌 Checking Gemini connection..."):
                preflight_embedding_manager = EmbeddingManager(
                    api_key=api_key,
                    model_name=EMBEDDING_MODEL_NAME
                )
                preflight_embedding_manager.test_connection()
            st.success("✅ Gemini connection verified")
            
            with st.spinner("⏳ Processing documents..."):
                # Process documents
                processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                chunks = processor.process_documents(DOCUMENTS_PATH)
                
                if not chunks:
                    st.error("No documents found to process!")
                    return
                
                st.success(f"✅ Created {len(chunks)} chunks")
                
                # Generate embeddings
                st.info("🔄 Generating embeddings...")
                embedding_manager = preflight_embedding_manager
                texts = [chunk["content"] for chunk in chunks]
                embeddings = embedding_manager.embed_batch(texts)
                
                # Store in vector store
                st.info("💾 Storing in vector database...")
                vector_store = FAISSVectorStore(vector_store_path=VECTOR_STORE_PATH)
                vector_store.add_documents(chunks, embeddings)
                vector_store.save()
                
                # Clear the RAG initialization cache so it reloads with new vector store
                st.info("🔄 Refreshing system cache...")
                st.cache_resource.clear()
                st.success("✅ System cache refreshed")
                
                st.success("✅ Documents processed and stored successfully!")
                
                # Clean up uploaded documents
                with st.spinner("🧹 Cleaning up temporary files..."):
                    cleanup_success = processor.cleanup_documents(DOCUMENTS_PATH)
                    if cleanup_success:
                        st.success("✅ Temporary documents removed from disk")
                    else:
                        st.warning("⚠️ Could not clean up all temporary documents")
                
                st.balloons()
        
        except Exception as e:
            error_message = str(e)
            st.error(f"❌ Error processing documents: {error_message}")
            if "Could not connect to Gemini API" in error_message or "Connection error" in error_message:
                st.warning(
                    "Gemini API could not be reached from this environment. "
                    "Verify outbound internet and GEMINI_API_KEY."
                )
            logger.error(f"Document processing error: {error_message}")


def qa_chat_tab(agent):
    """Tab for QA chatting"""
    if agent is None:
        st.error("❌ RAG System not initialized. Please process documents first.")
        return
    
    st.header("💬 Domain Knowledge QA")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    chat_container = st.container(height=500)
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "metadata" in message:
                    with st.expander("📊 Metadata"):
                        st.json({
                            "Confidence": f"{message['metadata'].get('confidence', 0)}%",
                            "Iterations": message['metadata'].get('iterations', 0)
                        })
    
    # User input
    st.divider()
    col1, col2 = st.columns([0.85, 0.15])
    
    with col1:
        user_input = st.chat_input("Ask a question about the documents...")
    
    with col2:
        if st.button("Clear", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Process user input
    if user_input:
        # Display user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate response
        with st.spinner("🤔 Agent reasoning..."):
            try:
                result = agent.reason(user_input)
                answer = result["answer"]
                
                # Display assistant response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "metadata": {
                        "confidence": result["confidence"],
                        "iterations": result["iterations"],
                        "sources": result.get("source_documents", [])
                    }
                })
                
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    
                    # Show sources and metadata
                    with st.expander("📊 Response Metadata"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Confidence", f"{result['confidence']}%")
                        with col2:
                            st.metric("Iterations", result["iterations"])
                        with col3:
                            st.metric("Retrieved Docs", len(result.get("source_documents", [])))
                        
                        # Display sources
                        if result.get("source_documents"):
                            unique_sources = list(dict.fromkeys(result["source_documents"]))  # Remove duplicates while preserving order
                            st.subheader("📚 Sources:")
                            for source in unique_sources:
                                st.write(f"• {source}")
                        
                        if result["reasoning_steps"]:
                            st.subheader("Reasoning Steps:")
                            for i, step in enumerate(result["reasoning_steps"], 1):
                                st.write(f"{i}. {step['step'].replace('_', ' ').title()}")
                
                st.rerun()
            
            except Exception as e:
                st.error(f"❌ Error generating response: {str(e)}")
                logger.error(f"Response generation error: {str(e)}")


def system_info_tab():
    """Tab showing system information"""
    st.header("ℹ️ System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Configuration")
        st.info(f"""
        - **Chat Model**: {CHAT_MODEL_NAME}
        - **Embedding Model**: {EMBEDDING_MODEL_NAME}
        - **Chunk Size**: {CHUNK_SIZE}
        - **Chunk Overlap**: {CHUNK_OVERLAP}
        - **Top-K Results**: {TOP_K_DOCUMENTS}
        - **Temperature**: {TEMPERATURE}
        """)
    
    with col2:
        st.subheader("System Status")
        # Check if vector store exists
        vector_store_exists = os.path.exists(os.path.join(VECTOR_STORE_PATH, "faiss.index"))
        st.metric("Vector Store Ready", "✅ Yes" if vector_store_exists else "❌ No")
        
        # Count documents
        docs_path = DOCUMENTS_PATH
        doc_count = 0
        if os.path.exists(docs_path):
            doc_count = len([f for f in os.listdir(docs_path) if f.endswith(('.txt', '.md', '.pdf'))])
        st.metric("Documents in Folder", doc_count)
    
    st.divider()
    st.subheader("About Agentic RAG System")
    st.markdown("""
    This system implements a **Retrieval-Augmented Generation (RAG)** with **Agentic** capabilities:
    
    ### Key Features:
    1. **Document Processing**: Chunking and preprocessing of domain documents
    2. **Embedding Generation**: Using Gemini embeddings
    3. **Vector Storage**: FAISS for efficient similarity search
    4. **Smart Retrieval**: Context-aware document retrieval
    5. **Agentic Reasoning**:
       - 🔧 **Tool Use**: Document retrieval as a callable tool
       - 🤔 **Self-Reflection**: Critic evaluates retrieved documents
       - 📝 **Answer Generation**: Grounded response with citations
    6. **Multi-iteration**: Refines queries if initial retrieval is insufficient
    
    ### Architecture:
    ```
    User Query → Agent → Retriever Tool → Vector Search → 
    Retrieved Context → Critic Evaluation → Answer Generation → Response
    ```
    
    ### Minimal Hallucinations:
    - Answers are grounded in retrieved documents
    - System explicitly states when information is not found
    - Source citations included in responses
    """)


def main():
    """Main application"""
    st.title(f"{STREAMLIT_PAGE_ICON} {STREAMLIT_PAGE_TITLE}")
    
    st.markdown("""
    A sophisticated RAG system with agentic capabilities for accurate domain knowledge QA
    """)
    
    # Initialize RAG system
    agent = initialize_rag_system()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📄 Process Documents", "💬 Ask Questions", "ℹ️ System Info"])
    
    with tab1:
        process_documents_tab()
    
    with tab2:
        qa_chat_tab(agent)
    
    with tab3:
        system_info_tab()
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.8em;'>
    Built with Gemini API • FAISS Vector Store • Agentic RAG Architecture
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
