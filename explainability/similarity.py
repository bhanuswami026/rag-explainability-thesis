"""
Similarity Attribution Module.
Calculates, normalizes, and visualizes similarity features between the user query and retrieved context chunks.
"""

from typing import List, Dict, Any
import numpy as np

class SimilarityExplainer:
    """
    Computes vector similarity attributions to explain retrieval choices.
    Provides relative similarity distributions and softmax-normalized relevance scores,
    which simulate a retrieval attention distribution.
    """
    
    def __init__(self):
        pass

    @staticmethod
    def compute_relative_importance(scores: List[float]) -> List[float]:
        """
        Computes the relative similarity weight of each chunk (simple ratio of sum).
        
        Args:
            scores (List[float]): Cosine similarity scores from vector retrieval.
            
        Returns:
            List[float]: Relative percentages of total retrieval similarity.
        """
        if not scores:
            return []
        
        # Avoid negative similarity scores in calculations (shift to positive range if needed)
        min_score = min(scores)
        adjusted_scores = [s - min(0.0, min_score) for s in scores]
        
        total = sum(adjusted_scores)
        if total == 0:
            return [1.0 / len(scores)] * len(scores)
            
        return [float(s / total) for s in adjusted_scores]

    @staticmethod
    def compute_softmax_relevance(scores: List[float], temperature: float = 0.05) -> List[float]:
        """
        Applies a softmax function over vector similarity scores to model a retrieval attention map.
        Low temperature values sharpen the differences, making the top match highly prominent,
        while higher values distribute attention more evenly.
        
        Args:
            scores (List[float]): Vector similarity scores.
            temperature (float): Softmax temperature hyperparameter.
            
        Returns:
            List[float]: Softmax probability distribution.
        """
        if not scores:
            return []
            
        # Convert to numpy array
        arr = np.array(scores, dtype=np.float32)
        
        # Apply temperature scaling
        scaled = arr / temperature
        
        # Softmax with numerical stability (subtract max)
        exp_arr = np.exp(scaled - np.max(scaled))
        softmax_weights = exp_arr / np.sum(exp_arr)
        
        return [float(w) for w in softmax_weights]

    def explain_retrieval(self, retrieved_chunks: List[Dict[str, Any]], scores: List[float]) -> List[Dict[str, Any]]:
        """
        Augments retrieved chunk metadata with comparative explainability metrics.
        
        Args:
            retrieved_chunks (List[Dict[str, Any]]): List of chunk metadata dicts.
            scores (List[float]): Cosine similarity scores.
            
        Returns:
            List[Dict[str, Any]]: Explanatory records combining text, scores, relative shares, and attention maps.
        """
        relative_shares = self.compute_relative_importance(scores)
        attention_shares = self.compute_softmax_relevance(scores)
        
        explanations = []
        for idx, chunk in enumerate(retrieved_chunks):
            explanations.append({
                "chunk_id": chunk.get("chunk_id", f"chunk_{idx}"),
                "source": chunk.get("source_name", "Unknown"),
                "page": chunk.get("page_num", 0),
                "text": chunk.get("text", ""),
                "similarity_score": scores[idx],
                "relative_similarity_share": relative_shares[idx],
                "retrieval_attention_weight": attention_shares[idx]
            })
            
        return explanations
