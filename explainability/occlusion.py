"""
Post-Hoc Chunk Occlusion (Counterfactual Perturbation) Explainer Module.
Systematically removes retrieved chunks, re-generates answers, and measures semantic and lexical shift.
This is a robust causal interpretability method for black-box LLM generators.
"""

import time
from typing import List, Dict, Any, Tuple
import numpy as np

class ChunkOcclusionExplainer:
    """
    Implements a perturbation-based post-hoc explanation of the LLM generator.
    By removing context chunks one-by-one and measuring the change in the generated answer,
    this explainer quantifies the causal attribution/necessity of each chunk.
    """
    
    def __init__(self, embedder):
        """
        Initializes the explainer.
        
        Args:
            embedder: BGEEmbedder instance (used to calculate semantic distance between responses).
        """
        self.embedder = embedder

    def calculate_jaccard_distance(self, text_a: str, text_b: str) -> float:
        """
        Calculates lexical token-level Jaccard distance (1 - Jaccard index) between two texts.
        
        Args:
            text_a (str): Original text.
            text_b (str): Perturbed text.
            
        Returns:
            float: Jaccard distance between 0.0 (identical sets) and 1.0 (disjoint sets).
        """
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        
        if not words_a and not words_b:
            return 0.0
            
        union = words_a.union(words_b)
        intersection = words_a.intersection(words_b)
        
        jaccard_similarity = len(intersection) / len(union)
        return 1.0 - jaccard_similarity

    def calculate_semantic_distance(self, text_a: str, text_b: str) -> float:
        """
        Calculates semantic vector distance (1 - cosine similarity) using the embedding model.
        
        Args:
            text_a (str): Original response.
            text_b (str): Response after context occlusion.
            
        Returns:
            float: Cosine distance between 0.0 (semantically identical) and 1.0 (orthogonal).
        """
        if not text_a.strip() or not text_b.strip():
            return 1.0 if text_a != text_b else 0.0
            
        # Generate embeddings
        emb_a = self.embedder.model.encode(text_a, show_progress_bar=False, normalize_embeddings=True)
        emb_b = self.embedder.model.encode(text_b, show_progress_bar=False, normalize_embeddings=True)
        
        # Cosine similarity is dot-product since vectors are normalized
        similarity = float(np.dot(emb_a, emb_b))
        
        # Clamp to avoid floating point anomalies out of [-1, 1]
        similarity = max(-1.0, min(1.0, similarity))
        
        return 1.0 - similarity

    def explain_generation(
        self, 
        pipeline, 
        query: str, 
        retrieved_chunks: List[Dict[str, Any]], 
        original_response: str
    ) -> List[Dict[str, Any]]:
        """
        Executes causal occlusion analysis. Systematically leaves one chunk out,
        generates a new answer, and measures how far it deviates from the original.
        
        Args:
            pipeline (RAGPipeline): Core pipeline containing configuration and Gemini.
            query (str): Original user query.
            retrieved_chunks (List[Dict[str, Any]]): Currently retrieved context segments.
            original_response (str): The primary generated answer containing all context.
            
        Returns:
            List[Dict[str, Any]]: List of occlusion analysis results for each chunk, containing:
                - "chunk_id": The target chunk identifier.
                - "occluded_chunk_text": The text of the removed chunk.
                - "perturbed_response": The generated response without this chunk.
                - "lexical_shift": Jaccard distance.
                - "semantic_shift": Vector cosine distance.
                - "causal_importance": Aggregated normalized score indicating necessity.
        """
        if len(retrieved_chunks) <= 1:
            # Occlusion requires at least 2 context segments to evaluate relative dependency.
            # If K=1, removing it completely collapses context to empty.
            # In that case, we can still evaluate it against empty context generation!
            pass
            
        occlusion_results = []
        shifts = []
        
        for target_idx, target_chunk in enumerate(retrieved_chunks):
            # Create a context set excluding the current chunk
            perturbed_contexts = [
                c["text"] for idx, c in enumerate(retrieved_chunks) if idx != target_idx
            ]
            
            # Generate response under counterfactual scenario
            # If all are removed (K=1), perturbed_contexts is empty, pipeline generates without context.
            perturbed_response, _, _ = pipeline.generate_answer_with_context(query, perturbed_contexts)
            
            # Calculate deviation distances
            lexical_shift = self.calculate_jaccard_distance(original_response, perturbed_response)
            semantic_shift = self.calculate_semantic_distance(original_response, perturbed_response)
            
            # Save results
            occlusion_results.append({
                "chunk_id": target_chunk["chunk_id"],
                "occluded_chunk_text": target_chunk["text"],
                "page": target_chunk["page_num"],
                "source": target_chunk["source_name"],
                "perturbed_response": perturbed_response,
                "lexical_shift": lexical_shift,
                "semantic_shift": semantic_shift,
            })
            
            # Use semantic shift as the primary metric for causal importance
            shifts.append(semantic_shift)
            
        # Normalize shifts to obtain a percentage causal necessity contribution score
        total_shift = sum(shifts)
        for idx, result in enumerate(occlusion_results):
            if total_shift == 0:
                result["causal_importance"] = 1.0 / len(retrieved_chunks)
            else:
                result["causal_importance"] = shifts[idx] / total_shift
                
        return occlusion_results
