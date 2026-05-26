"""
Saliency Mapping and Highlighting Module.
Finds alignments between the generated response and retrieved chunks.
Provides a highlighting generator that marks text overlap for visualization in Streamlit.
"""

import re
from typing import List, Dict, Any, Set

class SaliencyExplainer:
    """
    Computes lexical alignments between the generated answer and retrieved source text.
    Acts as a local post-hoc saliency map showing which exact sentences and keywords
    in the source chunks were reused in the final synthesis.
    """
    
    def __init__(self):
        # Standard English stop words to filter out before keyword matching
        self.stop_words = {
            "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", 
            "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", 
            "below", "between", "both", "but", "by", "can", "can't", "cannot", "could", 
            "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", 
            "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", 
            "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", 
            "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", 
            "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", 
            "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", 
            "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", 
            "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", 
            "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", 
            "than", "that", "that's", "the", "their", "theirs", "them", "themselves", 
            "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", 
            "they've", "this", "those", "through", "to", "too", "under", "until", "up", 
            "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", 
            "weren't", "what", "what's", "when", "when's", "where", "where's", "which", 
            "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", 
            "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", 
            "yourself", "yourselves"
        }

    def clean_text_to_keywords(self, text: str) -> Set[str]:
        """
        Tokenizes text, strips punctuation, downcases, and removes stop words.
        
        Args:
            text (str): The input text.
            
        Returns:
            Set[str]: A set of unique keywords.
        """
        # Remove punctuation and split on whitespace
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        return {w for w in words if w not in self.stop_words}

    def get_sentence_level_saliency(self, response: str, chunk_text: str) -> List[Dict[str, Any]]:
        """
        Splits a retrieved chunk into sentences and scores each sentence's
        relevance based on keyword overlap with the generated response.
        
        Args:
            response (str): The generated response.
            chunk_text (str): The raw text of a retrieved chunk.
            
        Returns:
            List[Dict[str, Any]]: List of sentences with their saliency attributes.
        """
        response_keywords = self.clean_text_to_keywords(response)
        
        # Simple sentence splitter (splitting on punctuation followed by space)
        sentences = re.split(r'(?<=[.!?])\s+', chunk_text.strip())
        
        sentence_records = []
        for idx, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
                
            sent_keywords = self.clean_text_to_keywords(sentence)
            
            # Compute overlap
            matched_keywords = sent_keywords.intersection(response_keywords)
            
            # Score is Jaccard-like or overlap ratio
            score = 0.0
            if sent_keywords:
                score = len(matched_keywords) / len(sent_keywords)
                
            sentence_records.append({
                "sentence_idx": idx,
                "text": sentence,
                "saliency_score": score,
                "matched_keywords": list(matched_keywords)
            })
            
        return sentence_records

    def generate_html_highlights(self, response: str, chunk_text: str) -> str:
        """
        Creates an HTML string of the chunk text with words highlighted based
        on their direct presence in the generated response.
        This provides a stunning micro-visual saliency mapping.
        
        Args:
            response (str): Generated response text.
            chunk_text (str): The retrieved chunk text.
            
        Returns:
            str: Styled HTML string with inline backgrounds.
        """
        response_keywords = self.clean_text_to_keywords(response)
        
        # Use regex to find word tokens and punctuation spaces to preserve layout
        tokens = re.split(r'(\b[a-zA-Z]+\b)', chunk_text)
        
        html_tokens = []
        for token in tokens:
            # Check if this is a word token
            if re.match(r'^[a-zA-Z]+$', token):
                token_lower = token.lower()
                if token_lower in response_keywords:
                    # Highlight word matching
                    # Deep emerald-green color indicating lexical match with glass-like transparency
                    html_tokens.append(
                        f'<span style="background-color: rgba(46, 204, 113, 0.25); '
                        f'border-bottom: 2px solid rgba(46, 204, 113, 0.6); '
                        f'padding: 2px 4px; border-radius: 3px; font-weight: 500; '
                        f'color: #e0f2f1;">{token}</span>'
                    )
                elif token_lower in self.stop_words:
                    # Soft grey background for matched stop words if they reside in the response
                    # This ensures cleaner, less noisy highlights by only focusing on keywords,
                    # but we can leave stop words default.
                    html_tokens.append(token)
                else:
                    html_tokens.append(token)
            else:
                html_tokens.append(token)
                
        return "".join(html_tokens)
