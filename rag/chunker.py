"""
Text Chunking Module for the RAG Explainability Thesis Project.
Splits parsed documents into overlapping text segments while retaining rich metadata.
"""

from typing import List, Dict, Any

class DocumentChunker:
    """
    Splits larger documents or pages into smaller overlapping text segments.
    Ensures that semantic context is preserved across boundaries (overlap)
    and maps each chunk back to its specific source coordinates (page, document) for explainability.
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Initializes the chunker.
        
        Args:
            chunk_size (int): Max character count per chunk.
            chunk_overlap (int): Number of overlapping characters between adjacent chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_pages(self, pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Splits a list of parsed pages into overlapping chunks.
        
        Args:
            pages_data (List[Dict[str, Any]]): Pages containing "text", "page_num", and "source_name".
            
        Returns:
            List[Dict[str, Any]]: List of chunks. Each chunk dictionary contains:
                - "text": The chunk text content.
                - "source_name": Parent file name.
                - "page_num": Page number.
                - "chunk_id": A unique string identifier (e.g. filename_page_chunkidx).
        """
        chunks = []
        
        for page_data in pages_data:
            text = page_data["text"]
            page_num = page_data["page_num"]
            source_name = page_data["source_name"]
            
            if not text:
                continue
                
            text_len = len(text)
            start_idx = 0
            chunk_idx = 0
            
            # Slide window across text
            while start_idx < text_len:
                end_idx = min(start_idx + self.chunk_size, text_len)
                chunk_text = text[start_idx:end_idx].strip()
                
                # Only keep non-empty chunks
                if chunk_text:
                    chunk_id = f"{source_name}_p{page_num}_c{chunk_idx}"
                    chunks.append({
                        "text": chunk_text,
                        "source_name": source_name,
                        "page_num": page_num,
                        "chunk_idx": chunk_idx,
                        "chunk_id": chunk_id
                    })
                    chunk_idx += 1
                
                # Check for completion
                if end_idx == text_len:
                    break
                    
                # Slide window forward (taking overlap into account)
                start_idx += (self.chunk_size - self.chunk_overlap)
                
                # Guard against infinite loops if overlap >= size
                if self.chunk_size <= self.chunk_overlap:
                    start_idx += self.chunk_size
                    
        return chunks
