"""
Academic RAG Evaluation Module.
Computes evaluation metrics based on the RAG Triad: Context Relevance, Groundedness (Faithfulness), and Response Relevance.
Also profiles operational latency across the pipeline.
"""

from typing import List, Dict, Any, Tuple
import numpy as np

class RAGEvaluator:
    """
    Evaluates RAG performance and quality post-hoc.
    Computes key metrics for thesis verification without relying on external heavy evaluation suites:
    1. Context Relevance: Average semantic query-chunk match.
    2. Groundedness (Faithfulness): Alignment of response with source text.
    3. Response Relevance: Semantic overlap of response with the original query.
    4. Operational Efficiency: Latency distributions.
    """
    
    def __init__(self, embedder):
        """
        Initializes the evaluator with an embedding model to compute semantic overlaps.
        
        Args:
            embedder: BGEEmbedder instance.
        """
        self.embedder = embedder

    def evaluate_context_relevance(self, similarity_scores: List[float]) -> float:
        """
        Calculates the Context Relevance metric.
        Averages the similarity scores of the retrieved chunks.
        
        Args:
            similarity_scores (List[float]): Cosine similarity scores from vector retrieval.
            
        Returns:
            float: Average context relevance score between 0.0 and 1.0.
        """
        if not similarity_scores:
            return 0.0
        return float(np.mean(similarity_scores))

    def evaluate_groundedness(self, response: str, retrieved_chunks: List[Dict[str, Any]]) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Calculates the Groundedness (Faithfulness) metric.
        Checks if sentences in the generated response are semantically covered by retrieved contexts.
        
        Args:
            response (str): The generated response.
            retrieved_chunks (List[Dict[str, Any]]): Retrieved source chunks.
            
        Returns:
            Tuple[float, List[Dict[str, Any]]]:
                - Groundedness score (0.0 to 1.0).
                - List of response sentences and their maximum context coverage score.
        """
        import re
        if not response.strip() or not retrieved_chunks:
            return 0.0, []
            
        # Compile all context texts into one master block
        full_context = " ".join([c["text"] for c in retrieved_chunks])
        
        # Split generated response into sentences
        response_sentences = re.split(r'(?<=[.!?])\s+', response.strip())
        response_sentences = [s.strip() for s in response_sentences if s.strip()]
        
        if not response_sentences:
            return 0.0, []
            
        # Embed all context paragraphs
        context_texts = [c["text"] for c in retrieved_chunks]
        context_embs = self.embedder.model.encode(context_texts, show_progress_bar=False, normalize_embeddings=True)
        
        sentence_coverages = []
        scores = []
        
        for sent in response_sentences:
            if len(sent.split()) < 3: # Skip very short connective phrases
                continue
                
            sent_emb = self.embedder.model.encode(sent, show_progress_bar=False, normalize_embeddings=True)
            
            # Find best matching context chunk for this sentence
            # Dot product is Cosine Similarity
            similarities = np.dot(context_embs, sent_emb)
            max_sim = float(np.max(similarities))
            
            # Clamp to [0, 1]
            max_sim = max(0.0, min(1.0, max_sim))
            
            sentence_coverages.append({
                "sentence": sent,
                "max_context_overlap": max_sim
            })
            scores.append(max_sim)
            
        if not scores:
            return 1.0, [] # No substantial sentences to evaluate
            
        # Groundedness is the average sentence-level semantic coverage score
        groundedness_score = float(np.mean(scores))
        return groundedness_score, sentence_coverages

    def evaluate_response_relevance(self, query: str, response: str) -> float:
        """
        Calculates the Response Relevance metric.
        Measures the semantic similarity between the user's query and the generated response.
        
        Args:
            query (str): The search query.
            response (str): The generated response.
            
        Returns:
            float: Response relevance score (0.0 to 1.0).
        """
        if not query.strip() or not response.strip():
            return 0.0
            
        # Generate embeddings
        query_emb = self.embedder.model.encode(query, show_progress_bar=False, normalize_embeddings=True)
        resp_emb = self.embedder.model.encode(response, show_progress_bar=False, normalize_embeddings=True)
        
        similarity = float(np.dot(query_emb, resp_emb))
        similarity = max(0.0, min(1.0, similarity))
        
        return similarity

    def get_latency_profile(self, latency_dict: Dict[str, float]) -> Dict[str, Any]:
        """
        Profiles execution efficiency by calculating percentage splits for retrieval vs generation.
        
        Args:
            latency_dict (Dict[str, float]): Timing records from the pipeline.
            
        Returns:
            Dict[str, Any]: Detailed percentage splits.
        """
        ret = latency_dict.get("retrieval", 0.0)
        gen = latency_dict.get("generation", 0.0)
        total = latency_dict.get("total", 1.0)
        
        ret_pct = (ret / total) * 100 if total > 0 else 0.0
        gen_pct = (gen / total) * 100 if total > 0 else 0.0
        
        return {
            "retrieval_seconds": ret,
            "retrieval_percentage": ret_pct,
            "generation_seconds": gen,
            "generation_percentage": gen_pct,
            "total_seconds": total
        }
