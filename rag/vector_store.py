"""
FAISS Vector Store Coordinator Module.
Handles local embedding indexing, serialization, and K-Nearest Neighbors retrieval.
"""

import os
import pickle
from typing import List, Dict, Any, Tuple
import faiss
import numpy as np

class FAISSVectorStore:
    """
    A local vector database powered by Facebook AI Similarity Search (FAISS).
    Fast, robust, and running entirely in-memory with disk serialization.
    Stores and indexes chunk representations and matches queries using inner-product (cosine) search.
    """
    
    def __init__(self, dimension: int = 384):
        """
        Initializes the vector store.
        
        Args:
            dimension (int): Dimension of vector embeddings (default 384 for BGE-small).
        """
        self.dimension = dimension
        # Use IndexFlatIP (Inner Product) since normalized vectors dot-product equals Cosine Similarity
        self.index = faiss.IndexFlatIP(self.dimension)
        # Store corresponding chunks metadata mapping
        self.chunks_metadata: List[Dict[str, Any]] = []

    def add_documents(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        """
        Adds text chunks and their corresponding embedding vectors to the index.
        
        Args:
            chunks (List[Dict[str, Any]]): Text chunks from DocumentChunker.
            embeddings (np.ndarray): Embedding matrix of shape (num_chunks, dimension).
        """
        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunks count ({len(chunks)}) does not match embeddings count ({len(embeddings)})")
            
        if len(chunks) == 0:
            return
            
        # Convert embeddings to float32 (FAISS standard)
        embeddings_f32 = embeddings.astype(np.float32)
        
        # Add to FAISS index
        self.index.add(embeddings_f32)
        
        # Append corresponding metadata items
        self.chunks_metadata.extend(chunks)
        print(f"Added {len(chunks)} chunks to FAISS index. Total indexed: {self.index.ntotal}")

    def search(self, query_embedding: np.ndarray, k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        """
        Searches the index for the top-K closest chunks.
        
        Args:
            query_embedding (np.ndarray): 1D query embedding of shape (dimension,) or (1, dimension).
            k (int): Number of nearest neighbors to retrieve.
            
        Returns:
            List[Tuple[Dict[str, Any], float]]: List of tuples containing (chunk_metadata, cosine_similarity_score).
        """
        if self.index.ntotal == 0:
            return []
            
        # Reshape to 2D array if 1D
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        # Ensure correct type
        query_embedding = query_embedding.astype(np.float32)
        
        # Perform retrieval
        # D represents the distances (similarities in case of Inner Product with normalized vectors)
        # I represents indices in the FAISS store
        k_capped = min(k, self.index.ntotal)
        D, I = self.index.search(query_embedding, k_capped)
        
        results = []
        for rank in range(k_capped):
            idx = I[0][rank]
            score = float(D[0][rank])
            
            # Map index back to the metadata
            if idx != -1 and idx < len(self.chunks_metadata):
                results.append((self.chunks_metadata[idx], score))
                
        return results

    def save(self, folder_path: str, filename_prefix: str = "faiss_index"):
        """
        Serializes and saves the FAISS index and metadata to a folder.
        
        Args:
            folder_path (str): Directory where to save the files.
            filename_prefix (str): Prefix name for the index and metadata files.
        """
        os.makedirs(folder_path, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(folder_path, f"{filename_prefix}.index")
        faiss.write_index(self.index, index_path)
        
        # Save metadata mapping via pickle
        meta_path = os.path.join(folder_path, f"{filename_prefix}.meta")
        with open(meta_path, "wb") as f:
            pickle.dump(self.chunks_metadata, f)
            
        print(f"Vector store saved successfully to {folder_path}")

    def load(self, folder_path: str, filename_prefix: str = "faiss_index"):
        """
        Loads the FAISS index and metadata from a folder.
        
        Args:
            folder_path (str): Directory from where to load the files.
            filename_prefix (str): Prefix name for the files.
        """
        index_path = os.path.join(folder_path, f"{filename_prefix}.index")
        meta_path = os.path.join(folder_path, f"{filename_prefix}.meta")
        
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(f"FAISS index files not found in {folder_path}")
            
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load metadata
        with open(meta_path, "rb") as f:
            self.chunks_metadata = pickle.load(f)
            
        self.dimension = self.index.d
        print(f"Vector store loaded successfully. Dimension: {self.dimension}, Total elements: {self.index.ntotal}")
        
    def clear(self):
        """Resets the vector store index and metadata mapping."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks_metadata = []
        print("Vector store cleared.")
