"""
Embeddings and Vector Store Management
"""
import os
import pickle
import logging
from typing import List, Tuple
import numpy as np
from pathlib import Path

try:
    import faiss
except ImportError:
    faiss = None

try:
    from google import genai  # type: ignore[import-not-found]
    from google.genai import types as genai_types  # type: ignore[import-not-found]
except ImportError:
    genai = None
    genai_types = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manage embeddings using Gemini API."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "models/text-embedding-004"
    ):
        """
        Initialize embedding manager for Gemini
        
        Args:
            api_key: Gemini API key
            model_name: Embedding model name
        """
        if not genai:
            raise ImportError("google-genai is required. Install it with: pip install google-genai")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def _raise_embedding_error_with_hint(self, err: Exception) -> None:
        """Raise a clearer error for common connectivity failures."""
        message = str(err)
        if "Connection error" in message:
            raise RuntimeError(
                "Could not connect to Gemini API for embeddings. "
                "Check that outbound internet access is available in the runtime environment "
                "and that GEMINI_API_KEY is valid."
            ) from err

        raise err

    def test_connection(self) -> None:
        """Run a minimal embedding request to validate endpoint and deployment connectivity."""
        try:
            self.embed_text("ping", task_type="retrieval_document")
        except Exception as e:
            logger.error(f"Embedding connection test failed: {str(e)}")
            self._raise_embedding_error_with_hint(e)

    def _embed(self, text: str, task_type: str) -> List[float]:
        response = self.client.models.embed_content(
            model=self.model_name,
            contents=[text],
            config=genai_types.EmbedContentConfig(task_type=task_type.upper()),
        )
        embeddings = getattr(response, "embeddings", None) or []
        if not embeddings:
            return []

        first = embeddings[0]
        values = getattr(first, "values", None)
        if values is None and isinstance(first, dict):
            values = first.get("values", [])
        if values is None:
            return []

        if not isinstance(values, list):
            values = list(values)
        return values

    def embed_text(self, text: str, task_type: str = "retrieval_document") -> List[float]:
        """
        Generate embedding for text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            return self._embed(text, task_type=task_type)
        except Exception as e:
            logger.error(f"Error embedding text: {str(e)}")
            self._raise_embedding_error_with_hint(e)

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding optimized for retrieval queries."""
        return self.embed_text(query, task_type="retrieval_query")

    def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for API calls
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            logger.info(f"Embedding batch {batch_num} of {total_batches}")
            
            try:
                batch_embeddings = []
                for text in batch:
                    batch_embeddings.append(self._embed(text, task_type="retrieval_document"))
                
                embeddings.extend(batch_embeddings)
                logger.info(f"Batch {batch_num}: Generated {len(batch_embeddings)} embeddings")
                
            except Exception as e:
                logger.error(f"Error in batch embedding: {str(e)}")
                # Fallback to individual embeddings
                for text in batch:
                    try:
                        embedding = self.embed_text(text)
                        embeddings.append(embedding)
                    except Exception as text_error:
                        logger.error(f"Error embedding text: {str(text_error)}")
                        self._raise_embedding_error_with_hint(text_error)
        
        logger.info(f"Total embeddings generated: {len(embeddings)}")
        if embeddings:
            logger.info(f"Embedding dimension: {len(embeddings[0])}")
        return embeddings


class FAISSVectorStore:
    """FAISS-based vector store for efficient similarity search"""

    def __init__(self, vector_store_path: str = "vector_store", store_embeddings: bool = False):
        """
        Initialize FAISS vector store
        
        Args:
            vector_store_path: Path to store FAISS index
        """
        if not faiss:
            raise ImportError("faiss-cpu is required. Install it with: pip install faiss-cpu")
        
        self.vector_store_path = vector_store_path
        self.index = None
        self.documents = []
        self.embeddings = []
        self.store_embeddings = store_embeddings
        
        Path(vector_store_path).mkdir(parents=True, exist_ok=True)

    def add_documents(self, chunks: List[dict], embeddings: List[List[float]]) -> None:
        """
        Add documents and their embeddings to vector store
        
        Args:
            chunks: List of document chunks with metadata
            embeddings: List of embedding vectors
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match")
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings, dtype="float32")

        # Create or append to FAISS index
        dimension = embeddings_array.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dimension)
        else:
            if self.index.d != dimension:
                raise ValueError("Embedding dimension does not match existing index")

        self.index.add(embeddings_array)
        self.documents.extend(chunks)
        if self.store_embeddings:
            self.embeddings.extend(embeddings)
        
        logger.info(f"Added {len(chunks)} documents to vector store")

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[dict, float]]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of (document, similarity_score) tuples
        """
        if self.index is None:
            logger.warning("Vector store is empty")
            return []
        
        query_array = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_array, top_k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            similarity = 1 / (1 + distance)  # Convert distance to similarity
            results.append((self.documents[idx], similarity))
        
        return results

    def save(self) -> None:
        """Save vector store to disk"""
        try:
            index_path = os.path.join(self.vector_store_path, "faiss.index")
            metadata_path = os.path.join(self.vector_store_path, "metadata.pkl")
            
            if self.index is None:
                logger.warning("Vector store is empty, nothing to save")
                return
            
            # Save FAISS index
            faiss.write_index(self.index, index_path)
            logger.info(f"FAISS index saved to {index_path}")
            
            # Save metadata
            metadata = {
                "documents": self.documents
            }

            if self.store_embeddings:
                metadata["embeddings"] = self.embeddings
            
            with open(metadata_path, "wb") as f:
                pickle.dump(metadata, f)
            
            logger.info(f"Metadata saved to {metadata_path}")
            logger.info(f"Vector store saved successfully with {len(self.documents)} documents")
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            raise

    def load(self) -> bool:
        """
        Load vector store from disk
        
        Returns:
            True if loaded successfully, False otherwise
        """
        index_path = os.path.join(self.vector_store_path, "faiss.index")
        metadata_path = os.path.join(self.vector_store_path, "metadata.pkl")
        
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            logger.warning("Vector store files not found")
            return False
        
        try:
            self.index = faiss.read_index(index_path)
            
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
            
            self.documents = metadata.get("documents", [])
            self.embeddings = metadata.get("embeddings", [])
            
            logger.info(f"Vector store loaded from {self.vector_store_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return False
