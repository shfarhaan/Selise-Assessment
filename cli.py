#!/usr/bin/env python
"""
Command-line interface for Agentic RAG System
Alternative to Streamlit UI for terminal-based interaction
"""
import os
import sys
import json
from pathlib import Path
from typing import Optional

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    GEMINI_API_KEY, VECTOR_STORE_PATH, DOCUMENTS_PATH,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_DOCUMENTS,
    CHAT_MODEL_NAME, EMBEDDING_MODEL_NAME
)
from src.document_processor import DocumentProcessor
from src.embeddings import EmbeddingManager, FAISSVectorStore
from src.retriever import RAGRetriever
from src.agent import AgenticRAG


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header():
    """Print application header"""
    print(f"""
{Colors.CYAN}{Colors.BOLD}
╔═══════════════════════════════════════════════════════════╗
║     Agentic RAG System for Domain Knowledge QA           ║
║     Command-Line Interface                               ║
╚═══════════════════════════════════════════════════════════╝
{Colors.ENDC}
    """)


def print_section(text: str):
    """Print section header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}>>> {text}{Colors.ENDC}")


def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")


def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.ENDC}")


def print_info(text: str):
    """Print info message"""
    print(f"{Colors.CYAN}ℹ {text}{Colors.ENDC}")


def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.ENDC}")


def process_documents_cli():
    """Process documents from command line"""
    print_section("Document Processing")
    
    if not GEMINI_API_KEY:
        print_error("Gemini API key not set. Please configure GEMINI_API_KEY in .env file.")
        return False
    
    if not os.path.exists(DOCUMENTS_PATH):
        print_warning(f"Documents directory '{DOCUMENTS_PATH}' does not exist.")
        return False
    
    # Count documents
    doc_files = [f for f in os.listdir(DOCUMENTS_PATH) 
                 if f.endswith(('.txt', '.md', '.pdf'))]
    
    if not doc_files:
        print_warning(f"No documents found in '{DOCUMENTS_PATH}'. Add .txt or .md files.")
        return False
    
    print_info(f"Found {len(doc_files)} documents to process")
    
    try:
        # Process documents
        print_info("Loading and chunking documents...")
        processor = DocumentProcessor(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = processor.process_documents(DOCUMENTS_PATH)
        print_success(f"Created {len(chunks)} chunks from {len(doc_files)} documents")
        
        # Generate embeddings
        print_info("Generating embeddings (this may take a minute)...")
        embedding_manager = EmbeddingManager(
            api_key=GEMINI_API_KEY,
            model_name=EMBEDDING_MODEL_NAME
        )
        texts = [chunk["content"] for chunk in chunks]
        embeddings = embedding_manager.embed_batch(texts)
        print_success(f"Generated {len(embeddings)} embeddings")
        
        # Store in vector database
        print_info("Storing in FAISS vector database...")
        vector_store = FAISSVectorStore(vector_store_path=VECTOR_STORE_PATH)
        vector_store.add_documents(chunks, embeddings)
        vector_store.save()
        print_success("Vector database created and saved successfully!")
        
        return True
    
    except Exception as e:
        print_error(f"Error processing documents: {str(e)}")
        return False


def initialize_rag_agent() -> Optional[AgenticRAG]:
    """Initialize RAG agent"""
    try:
        if not GEMINI_API_KEY:
            print_error("Gemini API key not configured")
            return None
        
        # Initialize embedding manager and vector store
        embedding_manager = EmbeddingManager(
            api_key=GEMINI_API_KEY,
            model_name=EMBEDDING_MODEL_NAME
        )
        vector_store = FAISSVectorStore(vector_store_path=VECTOR_STORE_PATH)
        
        # Load existing vector store
        if not vector_store.load():
            print_error("Vector store not found. Please process documents first.")
            return None
        
        print_success("Vector store loaded")
        
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
        
        return agent
    
    except Exception as e:
        print_error(f"Error initializing RAG agent: {str(e)}")
        return None


def chat_loop(agent: AgenticRAG):
    """Interactive chat loop"""
    print_section("Chat Mode")
    print_info("Type 'quit' or 'exit' to end conversation")
    print_info("Type 'clear' to clear history")
    print_info("Type 'history' to view conversation history")
    print()
    
    while True:
        try:
            # Get user input
            user_input = input(f"\n{Colors.BOLD}You:{Colors.ENDC} ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit']:
                print_info("Goodbye!")
                break
            
            if user_input.lower() == 'clear':
                agent.clear_history()
                print_info("Conversation history cleared")
                continue
            
            if user_input.lower() == 'history':
                history = agent.get_conversation_history()
                if not history:
                    print_warning("No conversation history")
                else:
                    print_section("Conversation History")
                    for msg in history:
                        print(f"{msg['role'].upper()}: {msg['content'][:100]}...")
                continue
            
            # Generate response
            print_info("Agent reasoning...")
            response = agent.chat(user_input)
            
            print(f"\n{Colors.GREEN}{Colors.BOLD}Agent:{Colors.ENDC}")
            print(response)
        
        except KeyboardInterrupt:
            print_info("\nConversation interrupted")
            break
        except Exception as e:
            print_error(f"Error: {str(e)}")


def show_help():
    """Show help menu"""
    print_section("Help Menu")
    print(f"""
{Colors.BOLD}Usage:{Colors.ENDC}
    python cli.py [command]

{Colors.BOLD}Commands:{Colors.ENDC}
    process     Process documents and create vector store
    chat        Start interactive chat mode
    info        Show system information
    help        Show this help message

{Colors.BOLD}Examples:{Colors.ENDC}
    python cli.py process   # Process documents
    python cli.py chat      # Start chatting
    python cli.py info      # Show system info

{Colors.BOLD}Workflow:{Colors.ENDC}
    1. Place your documents in the 'documents' folder
    2. Run: python cli.py process
    3. Run: python cli.py chat
    4. Ask questions about your documents!
    """)


def show_info():
    """Show system information"""
    print_section("System Information")
    
    # Check configuration
    print(f"\n{Colors.BOLD}Configuration:{Colors.ENDC}")
    print(f"  API Key Set: {'Yes' if GEMINI_API_KEY else 'No'}")
    print(f"  Chat Model: {CHAT_MODEL_NAME}")
    print(f"  Embedding Model: {EMBEDDING_MODEL_NAME}")
    print(f"  Chunk Size: {CHUNK_SIZE}")
    print(f"  Chunk Overlap: {CHUNK_OVERLAP}")
    print(f"  Top-K Results: {TOP_K_DOCUMENTS}")
    
    # Check documents
    print(f"\n{Colors.BOLD}Documents:{Colors.ENDC}")
    doc_count = 0
    if os.path.exists(DOCUMENTS_PATH):
        doc_count = len([f for f in os.listdir(DOCUMENTS_PATH) 
                        if f.endswith(('.txt', '.md', '.pdf'))])
    print(f"  Path: {DOCUMENTS_PATH}")
    print(f"  Documents Found: {doc_count}")
    
    # Check vector store
    print(f"\n{Colors.BOLD}Vector Store:{Colors.ENDC}")
    vector_store_exists = (os.path.exists(os.path.join(VECTOR_STORE_PATH, "faiss.index")) and
                          os.path.exists(os.path.join(VECTOR_STORE_PATH, "metadata.pkl")))
    print(f"  Path: {VECTOR_STORE_PATH}")
    print(f"  Ready: {'Yes' if vector_store_exists else 'No'}")
    
    # Next steps
    print(f"\n{Colors.BOLD}Next Steps:{Colors.ENDC}")
    if doc_count == 0:
        print(f"  1. Add .txt or .md files to '{DOCUMENTS_PATH}' folder")
        print(f"  2. Run: python cli.py process")
        print(f"  3. Run: python cli.py chat")
    elif not vector_store_exists:
        print(f"  1. Run: python cli.py process")
        print(f"  2. Run: python cli.py chat")
    else:
        print(f"  Ready to chat! Run: python cli.py chat")


def main():
    """Main function"""
    print_header()
    
    # Get command from arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
    else:
        command = "help"
    
    if command == "process":
        success = process_documents_cli()
        if success:
            print_success("\nDocuments ready for querying!")
    
    elif command == "chat":
        print_info("Initializing RAG system...")
        agent = initialize_rag_agent()
        if agent:
            chat_loop(agent)
    
    elif command == "info":
        show_info()
    
    elif command in ["help", "-h", "--help"]:
        show_help()
    
    else:
        print_error(f"Unknown command: {command}")
        print_info("Run 'python cli.py help' for help")


if __name__ == "__main__":
    main()
