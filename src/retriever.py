"""
Retrieval Logic for RAG System
"""
import logging
from typing import List, Tuple
from src.embeddings import EmbeddingManager, FAISSVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGRetriever:
    """Retrieves relevant documents based on query"""

    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        vector_store: FAISSVectorStore,
        top_k: int = 5,
        similarity_threshold: float = 0.3
    ):
        """
        Initialize RAG retriever
        
        Args:
            embedding_manager: Manager for generating embeddings
            vector_store: Vector store for similarity search
            top_k: Number of top results to retrieve
            similarity_threshold: Minimum similarity score
        """
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

    def retrieve(self, query: str) -> List[Tuple[str, float, str]]:
        """
        Retrieve relevant documents for query
        
        Args:
            query: User query
            
        Returns:
            List of (content, similarity_score, source) tuples
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_manager.embed_query(query)
            
            # Search vector store
            results = self.vector_store.search(query_embedding, top_k=self.top_k)
            
            # Filter by similarity threshold and format results
            retrieved_docs = []
            for doc, similarity in results:
                if similarity >= self.similarity_threshold:
                    retrieved_docs.append((
                        doc["content"],
                        similarity,
                        doc["metadata"]["source"]
                    ))
            
            logger.info(f"Retrieved {len(retrieved_docs)} documents for query")
            return retrieved_docs
        
        except Exception as e:
            logger.error(f"Error during retrieval: {str(e)}")
            raise

    def retrieve_with_context(self, query: str) -> str:
        """
        Retrieve documents and format as context string
        
        Args:
            query: User query
            
        Returns:
            Formatted context string with source information
        """
        retrieved_docs = self.retrieve(query)
        
        if not retrieved_docs:
            return "No relevant documents found in the knowledge base."
        
        context_parts = []
        for i, (content, similarity, source) in enumerate(retrieved_docs, 1):
            # Extract filename from source path
            filename = source.split('/')[-1].split('\\')[-1]
            context_parts.append(
                f"[Document {i}: {filename}] (Relevance: {similarity:.2%})\n"
                f"Source: {source}\n"
                f"{content}"
            )
        
        return "\n\n---\n\n".join(context_parts)
