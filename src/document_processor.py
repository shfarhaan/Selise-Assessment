"""
Document Processor: Handles document loading and chunking
"""
import os
from typing import List, Tuple
from pathlib import Path
import logging

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Process and chunk documents for RAG"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize document processor
        
        Args:
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
        """
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_documents(self, directory: str) -> List[Tuple[str, str]]:
        """
        Load documents from directory
        
        Args:
            directory: Path to directory containing documents
            
        Returns:
            List of (content, filename) tuples
        """
        documents = []
        supported_formats = [".txt", ".pdf", ".md"]
        
        if not os.path.exists(directory):
            logger.warning(f"Directory {directory} does not exist")
            return documents
        
        for file_path in Path(directory).rglob("*"):
            if file_path.suffix.lower() in supported_formats:
                try:
                    if file_path.suffix.lower() == ".pdf":
                        if not PdfReader:
                            logger.warning(
                                f"PDF support requires pypdf. Skipping {file_path.name}"
                            )
                            continue

                        with open(file_path, "rb") as f:
                            reader = PdfReader(f)
                            content = "".join(
                                (page.extract_text() or "") for page in reader.pages
                            )

                        if content.strip():
                            documents.append((content, file_path.name))
                            logger.info(f"Loaded document: {file_path.name}")
                        else:
                            logger.warning(
                                f"No extractable text found in {file_path.name}"
                            )
                    else:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            documents.append((content, file_path.name))
                            logger.info(f"Loaded document: {file_path.name}")
                except Exception as e:
                    logger.error(f"Error loading {file_path.name}: {str(e)}")
        
        return documents

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to chunk
            
        Returns:
            List of chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            prev_start = start
            # Find end position
            end = start + self.chunk_size
            
            # Try to break at sentence boundary if possible
            if end < len(text):
                # Look for last sentence break within chunk
                last_period = text.rfind(".", start, end)
                last_newline = text.rfind("\n", start, end)
                last_break = max(last_period, last_newline)
                
                if last_break > start:
                    # Only break at boundaries that still allow forward movement
                    # after overlap subtraction.
                    candidate_end = last_break + 1
                    if candidate_end - start > self.chunk_overlap:
                        end = candidate_end
            
            chunks.append(text[start:end].strip())
            
            # Move start position with overlap and guarantee progress.
            start = end - self.chunk_overlap
            if start <= prev_start:
                start = prev_start + 1
        
        return [chunk for chunk in chunks if chunk]  # Remove empty chunks

    def process_documents(self, directory: str) -> List[dict]:
        """
        Load and chunk documents
        
        Args:
            directory: Path to documents directory
            
        Returns:
            List of chunk dictionaries with metadata
        """
        documents = self.load_documents(directory)
        processed_chunks = []
        
        for content, filename in documents:
            chunks = self.chunk_text(content)
            logger.info(f"Created {len(chunks)} chunks from {filename}")
            
            for idx, chunk in enumerate(chunks):
                processed_chunks.append({
                    "content": chunk,
                    "source": filename,
                    "chunk_id": idx,
                    "metadata": {
                        "source": filename,
                        "chunk_index": idx,
                        "total_chunks": len(chunks)
                    }
                })
        
        logger.info(f"Total chunks created: {len(processed_chunks)}")
        return processed_chunks

    def cleanup_documents(self, directory: str) -> bool:
        """
        Remove all uploaded documents from directory after processing
        
        Args:
            directory: Path to documents directory to clean up
            
        Returns:
            True if cleanup successful, False otherwise
        """
        try:
            if not os.path.exists(directory):
                logger.warning(f"Directory {directory} does not exist")
                return False
            
            removed_count = 0
            supported_formats = [".txt", ".pdf", ".md"]
            files_to_remove = []
            
            # First, collect all files to remove
            for file_path in Path(directory).rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in supported_formats:
                    files_to_remove.append(file_path)
            
            if not files_to_remove:
                logger.info(f"No documents found in {directory} to clean up")
                return True  # Still return True as there's nothing to do
            
            # Then remove them
            for file_path in files_to_remove:
                try:
                    # Close any open handles first
                    file_path.unlink()  # Use unlink() which is more reliable than os.remove()
                    removed_count += 1
                    logger.info(f"Removed: {file_path.name}")
                except PermissionError as e:
                    logger.error(f"Permission denied removing {file_path.name}: {str(e)}")
                except Exception as e:
                    logger.error(f"Error removing {file_path.name}: {str(e)}")
            
            logger.info(f"Cleanup complete: Removed {removed_count} documents")
            return True  # Return True if cleanup attempted, regardless of individual file success
        
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            return False
