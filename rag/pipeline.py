"""
Core RAG Pipeline Integration Module.
Integrates Parser, Chunker, Embedder, FAISS Vector Store, and Google Gemini API.
"""

import time
import os
from typing import List, Dict, Any, Tuple
import google.generativeai as genai
from dotenv import load_dotenv

from rag.parser import PDFParser
from rag.chunker import DocumentChunker
from rag.embedder import BGEEmbedder
from rag.vector_store import FAISSVectorStore

# Load environment variables
load_dotenv()

class RAGPipeline:
    """
    Coordinating pipeline that orchestrates the ingestion of documents,
    retrieval of relevant contexts, prompt formulation, and answer generation using the Google Gemini LLM.
    Also measures operational latency at each step to support performance profile evaluation.
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Initializes the pipeline elements.
        """
        self.parser = PDFParser()
        self.chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedder = BGEEmbedder()
        self.vector_store = FAISSVectorStore(dimension=self.embedder.dimension)
        
        # Initialize Gemini API
        self.api_key = os.getenv("GEMINI_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
            print("Gemini API client configured successfully.")
        else:
            print("WARNING: GEMINI_API_KEY not found in environment variables. Gemini calls will fail.")

    def configure_gemini(self, api_key: str):
        """
        Dynamically configures Gemini API key from UI input.
        """
        self.api_key = api_key
        genai.configure(api_key=api_key)
        print("Gemini API key updated dynamically.")

    def ingest_document_bytes(self, file_bytes: bytes, filename: str) -> int:
        """
        Ingests a PDF document from memory bytes, chunks it, generates embeddings,
        and indexes it into the local FAISS store.
        
        Args:
            file_bytes (bytes): Binary data of the PDF file.
            filename (str): Name of the file.
            
        Returns:
            int: Number of chunks added.
        """
        # 1. Parse text page by page
        pages_data = self.parser.parse_bytes(file_bytes, filename)
        
        # 2. Chunk page contents
        chunks = self.chunker.split_pages(pages_data)
        
        if not chunks:
            return 0
            
        # 3. Generate dense vector representations
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.embed_documents(texts)
        
        # 4. Save to vector index
        self.vector_store.add_documents(chunks, embeddings)
        return len(chunks)

    def retrieve(self, query: str, k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieves the top-K relevant chunks for a given query.
        
        Args:
            query (str): The search query.
            k (int): Number of chunks to retrieve.
            
        Returns:
            List[Tuple[Dict[str, Any], float]]: List of (chunk_metadata, cosine_similarity_score)
        """
        query_embedding = self.embedder.embed_query(query)
        return self.vector_store.search(query_embedding, k=k)

    def generate_answer_with_context(self, query: str, contexts: List[str]) -> Tuple[str, str, float]:
        """
        Calls the Gemini model to synthesize a response using the retrieved contexts.
        
        Args:
            query (str): The search query.
            contexts (List[str]): Extracted texts from retrieved chunks.
            
        Returns:
            Tuple[str, str, float]: (Generated Answer, Formatted Prompt, Latency in seconds)
        """
        if not self.api_key:
            return (
                "Error: Gemini API Key is missing. Please configure it in the sidebar or set GEMINI_API_KEY in the .env file.",
                "",
                0.0
            )
            
        # Structure the system prompt and retrieve-augmented context
        context_str = "\n\n".join([f"--- Context Segment {idx+1} ---\n{ctx}" for idx, ctx in enumerate(contexts)])
        
        prompt = f"""You are an advanced Retrieval-Augmented Generation (RAG) assistant. 
Synthesize a precise, accurate, and completely grounded response to the user's query based ONLY on the provided context segments below.

If the information needed to answer is not present in the context, clearly state that the document does not contain the answer. Do not use outside knowledge.

Retrieved Context Segments:
{context_str}

User Query:
{query}

Grounded Response:"""

        start_time = time.time()
        try:
            # Use gemini-1.5-flash for lightning fast execution
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0  # Crucial for deterministic/explainable generation
                )
            )
            generation_time = time.time() - start_time
            return response.text.strip(), prompt, generation_time
        except Exception as e:
            generation_time = time.time() - start_time
            return f"Error during answer generation: {str(e)}", prompt, generation_time

    def query(self, user_query: str, k: int = 3) -> Dict[str, Any]:
        """
        Executes a complete RAG operation: Retrieve -> Augment -> Generate.
        Tracks operational latencies.
        
        Args:
            user_query (str): The user query string.
            k (int): Number of chunks to retrieve.
            
        Returns:
            Dict[str, Any]: Dictionary containing complete RAG output and execution profiles:
                - "query": original query
                - "answer": synthesized response
                - "retrieved_chunks": list of retrieved chunk dictionaries
                - "similarity_scores": list of cosine similarity scores
                - "prompt": full prompt sent to Gemini
                - "latency": operational timing profile (retrieval, generation, total)
        """
        total_start = time.time()
        
        # 1. RETRIEVE phase
        retrieval_start = time.time()
        retrieved_results = self.retrieve(user_query, k=k)
        retrieval_latency = time.time() - retrieval_start
        
        retrieved_chunks = [item[0] for item in retrieved_results]
        similarity_scores = [item[1] for item in retrieved_results]
        contexts = [c["text"] for c in retrieved_chunks]
        
        # 2. GENERATE phase
        if not retrieved_chunks:
            answer = "No relevant context segments were found in the document database to answer your query."
            prompt = ""
            generation_latency = 0.0
        else:
            answer, prompt, generation_latency = self.generate_answer_with_context(user_query, contexts)
            
        total_latency = time.time() - total_start
        
        return {
            "query": user_query,
            "answer": answer,
            "retrieved_chunks": retrieved_chunks,
            "similarity_scores": similarity_scores,
            "prompt": prompt,
            "latency": {
                "retrieval": retrieval_latency,
                "generation": generation_latency,
                "total": total_latency
            }
        }
