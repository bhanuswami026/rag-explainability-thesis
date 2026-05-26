"""
HuggingFace BGE-small Embedding Wrapper for RAG pipeline.
Uses local SentenceTransformers to generate dense vector embeddings.
"""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

class BGEEmbedder:
    """
    Wrapper around the BAAI/bge-small-en-v1.5 embedding model.
    A state-of-the-art, lightweight model that produces 384-dimensional dense vectors,
    ideal for academic demonstrations and local CPU deployment.
    """
    
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        """
        Initializes the embedding model.
        
        Args:
            model_name (str): HuggingFace model hub ID.
        """
        print(f"Loading embedding model: {model_name}...")
        # Automatically downloads and caches on first initialization
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Embedding model loaded successfully. Dimension: {self.dimension}")

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """
        Generates dense vector embeddings for a list of document chunks.
        
        Args:
            texts (List[str]): List of text chunks.
            
        Returns:
            np.ndarray: Matrix of shape (num_chunks, 384) containing embeddings.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
            
        # BGE models perform best on documents when queries are explicitly formatted.
        # However, for pure similarity, we can embed directly or add search prefixes.
        # Since this is a simple local pipeline, standard sentence encoding is clean.
        embeddings = self.model.encode(
            texts, 
            show_progress_bar=False, 
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalizing to unit length simplifies Cosine Similarity to Dot Product
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generates a dense vector embedding for a query.
        For BGE models, prepending an instruction to the query is highly recommended.
        
        Args:
            query (str): The search query text.
            
        Returns:
            np.ndarray: Vector of shape (384,) containing the normalized query embedding.
        """
        # BGE instruction prefix for query retrieval optimization
        query_prefix = "Represent this sentence for searching relevant passages: "
        formatted_query = query_prefix + query
        
        embedding = self.model.encode(
            formatted_query, 
            show_progress_bar=False, 
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding.astype(np.float32)
