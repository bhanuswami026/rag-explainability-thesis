"""
PDF Parser Module for the RAG Explainability Thesis Project.
Uses PyMuPDF (fitz) to extract text and structure from PDF documents.
"""

import os
from typing import List, Dict, Any
import fitz  # PyMuPDF

class PDFParser:
    """
    A lightweight, robust parser for PDF documents.
    Extracts text page-by-page and records metadata to ensure source traceability.
    """
    
    def __init__(self):
        pass

    def parse(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parses a PDF file from a local path and extracts text with metadata.
        
        Args:
            file_path (str): Path to the PDF document.
            
        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing pages, each containing:
                - "text": The extracted raw text of the page.
                - "page_num": The page number (1-indexed).
                - "source_name": The filename of the PDF.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
            
        source_name = os.path.basename(file_path)
        pages_data = []
        
        try:
            # Open PDF document
            doc = fitz.open(file_path)
            
            for page_idx, page in enumerate(doc):
                # Extract text preserving layout representation
                text = page.get_text("text").strip()
                
                if text:  # Skip completely empty pages
                    pages_data.append({
                        "text": text,
                        "page_num": page_idx + 1,
                        "source_name": source_name
                    })
            
            doc.close()
        except Exception as e:
            print(f"Error parsing PDF file {file_path}: {str(e)}")
            raise e
            
        return pages_data

    def parse_bytes(self, file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
        """
        Parses PDF text directly from in-memory bytes (useful for Streamlit file uploads).
        
        Args:
            file_bytes (bytes): The raw uploaded file bytes.
            filename (str): The name of the file.
            
        Returns:
            List[Dict[str, Any]]: List of page data dicts.
        """
        pages_data = []
        try:
            # Open PDF directly from bytes stream
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            
            for page_idx, page in enumerate(doc):
                text = page.get_text("text").strip()
                if text:
                    pages_data.append({
                        "text": text,
                        "page_num": page_idx + 1,
                        "source_name": filename
                    })
            doc.close()
        except Exception as e:
            print(f"Error parsing PDF bytes for {filename}: {str(e)}")
            raise e
            
        return pages_data
