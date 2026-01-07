import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import string

class TextCleaner:
    def __init__(self):
        # Download NLTK data if not already present
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))
    
    def clean_basic(self, text):
        """Basic text cleaning"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\']', '', text)
        
        # Remove multiple periods
        text = re.sub(r'\.{2,}', '.', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def clean_sentences(self, text):
        """Clean text at sentence level"""
        if not text:
            return ""
        
        # Tokenize into sentences
        sentences = sent_tokenize(text)
        
        cleaned_sentences = []
        for sentence in sentences:
            # Clean each sentence
            cleaned = self.clean_basic(sentence)
            
            # Skip very short sentences (likely noise)
            if len(cleaned.split()) > 3:
                cleaned_sentences.append(cleaned)
        
        return ' '.join(cleaned_sentences)
    
    def remove_stopwords(self, text):
        """Remove stopwords (optional, for summary generation)"""
        if not text:
            return ""
        
        words = word_tokenize(text.lower())
        filtered_words = [word for word in words if word not in self.stop_words and word.isalnum()]
        
        return ' '.join(filtered_words)
    
    def fix_encoding_issues(self, text):
        """Fix common encoding issues"""
        if not text:
            return ""
        
        # Fix common encoding issues
        replacements = {
            'â€™': "'",
            'â€œ': '"',
            'â€': '"',
            'â€"': '—',
            'â€"': '–',
            'Â': '',
            'â€¦': '...',
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def clean_for_translation(self, text):
        """Clean text specifically for translation"""
        if not text:
            return ""
        
        # Apply basic cleaning
        text = self.fix_encoding_issues(text)
        text = self.clean_basic(text)
        
        # Split into sentences and clean each
        text = self.clean_sentences(text)
        
        # Ensure proper spacing around punctuation
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        text = re.sub(r'([,.!?;:])(?!\s)', r'\1 ', text)
        
        # Remove excessive line breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'\n', ' ', text)
        
        # Final cleanup
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def chunk_text(self, text, max_chunk_size=4000):
        """Split text into chunks for translation (Google Translate has limits)"""
        if not text:
            return []
        
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence exceeds the limit, start a new chunk
            if len(current_chunk) + len(sentence) > max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    # Single sentence is too long, split it
                    words = sentence.split()
                    temp_chunk = ""
                    for word in words:
                        if len(temp_chunk) + len(word) + 1 > max_chunk_size:
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                                temp_chunk = word
                            else:
                                chunks.append(word)
                        else:
                            temp_chunk += " " + word if temp_chunk else word
                    current_chunk = temp_chunk
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks