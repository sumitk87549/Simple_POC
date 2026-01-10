import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import string
import os
import sys
import argparse
from datetime import datetime
from typing import Optional

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

    # ---------------- Gutenberg detection & stripping ----------------
    def is_gutenberg(self, text: str) -> bool:
        """
        Heuristic checks to decide whether a text is likely from Project Gutenberg.
        Returns True if likely Gutenberg.
        """
        if not text or len(text) < 200:
            return False

        lowered = text.lower()
        # Quick obvious checks
        markers = [
            'project gutenberg', 'project gutenberg-tm', 'gutenberg association',
            'michael s. hart', 'illinois benedictine college', 'this etext is prepared',
            'etext', 'gutenberg etext', 'startthe small print', 'endthe small print',
            'start of this project gutenberg', 'end of this project gutenberg',
            'start of the project gutenberg', 'end of the project gutenberg'
        ]
        hits = sum(1 for m in markers if m in lowered)

        # If multiple markers present, almost certainly Gutenberg
        if hits >= 2:
            return True

        # If explicit start/end markers present
        if re.search(r'\*\*\*\s*start of (this|the) project gutenberg', lowered) \
           or re.search(r'\*\*\*\s*end of (this|the) project gutenberg', lowered):
            return True

        # If legal/trademark "Small Print" block exists
        if 'startthe small print' in lowered or 'endthe small print' in lowered or 'the small print' in lowered:
            return True

        # Otherwise, not confident enough
        return False

    def strip_gutenberg_header_footer(self, text: str) -> str:
        """
        Attempts to remove common Project Gutenberg headers and footers.
        Returns the stripped text. If stripping fails or yields very short output,
        the original text will be returned by the caller.
        """
        if not text or len(text) < 200:
            return text

        original = text

        # Common START markers (try several variants). Use DOTALL for multi-line matching if needed.
        start_patterns = [
            r'\*\*\*\s*start of (this|the) project gutenberg[^\n]*\*\*\*',  # *** START OF THIS PROJECT GUTENBERG EBOOK ...
            r'^\s*project gutenberg?s? (ebook|etext)[\s\S]{0,200}',       # Project Gutenberg Etext header lines
            r'start\s*the\s*small\s*print',                              # STARTTHE SMALL PRINT
            r'\bthis project gutenberg\b',                               # fallback: first "Project Gutenberg" mention
        ]

        # Common END markers
        end_patterns = [
            r'\*\*\*\s*end of (this|the) project gutenberg[^\n]*\*\*\*',
            r'end\s*the\s*small\s*print',                                # ENDTHE SMALL PRINT
            r'\bend of (this|the) project gutenberg\b',
            r'this etext is prepared directly from',                      # some etext footers
        ]

        start_pos: Optional[int] = None
        end_pos: Optional[int] = None

        lowered = text.lower()

        # Find header end: locate the last start-pattern occurrence (so we cut everything up to after it)
        for pat in start_patterns:
            m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
            if m:
                # prefer the end of the matched block
                start_pos = m.end()
                break

        # If header start not found, try to find the book title area:
        if start_pos is None:
            # Heuristic: look for two consecutive newlines followed by a line with alphabets and maybe date
            m = re.search(r'\n{2,}([A-Z][A-Za-z0-9\-\s\']{3,150})\n', text)
            if m:
                # try to place start before this title-like block, but not too early
                start_pos = max(0, m.start())

        # Find footer start (where to cut off)
        for pat in end_patterns:
            m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
            if m:
                end_pos = m.start()
                break

        # If we couldn't find an explicit footer but found a header, attempt to find 'END' markers by searching for common ending tokens near the end.
        if end_pos is None:
            m = re.search(r'\*\*\*\s*end of (this|the) project gutenberg', text, flags=re.IGNORECASE)
            if m:
                end_pos = m.start()

        # If neither start_pos nor end_pos found, be conservative and return original
        if start_pos is None and end_pos is None:
            return original

        # Determine slice indices
        start_index = start_pos if start_pos is not None else 0
        end_index = end_pos if end_pos is not None else len(text)

        # Trim whitespace and any leading/trailing repeated Gutenberg blocks
        stripped = text[start_index:end_index].strip()

        # If stripped looks very small compared to original, reject and return original
        if len(stripped) < max(200, len(original) * 0.1):
            return original

        # Remove leftover common Gutenberg footer lines within the stripped portion
        leftover_patterns = [
            r'(?i)project gutenberg', r'(?i)michael s\. hart', r'(?i)illinois benedictine', r'(?i)the small print',
            r'(?i)permission is granted', r'(?i)you may copy'
        ]
        for lp in leftover_patterns:
            stripped = re.sub(lp, '', stripped, flags=re.IGNORECASE)

        # Normalize whitespace
        stripped = re.sub(r'\s+\n', '\n', stripped)
        stripped = re.sub(r'\n{3,}', '\n\n', stripped)
        stripped = stripped.strip()

        return stripped

    # ---------------- End Gutenberg functions ----------------

    def clean_for_translation(self, text):
        """Clean text specifically for translation"""
        if not text:
            return ""

        # Fix encoding issues first
        text = self.fix_encoding_issues(text)

        # If text appears to be a Project Gutenberg etext, strip header/footer
        try:
            if self.is_gutenberg(text):
                stripped = self.strip_gutenberg_header_footer(text)
                # Only accept stripped if it's meaningfully different and not too short
                if stripped and len(stripped) > 200 and len(stripped) > 0.2 * len(text):
                    text = stripped
        except Exception:
            # In case of any unexpected failure in detection/stripping, fall back to original text
            pass

        # Apply basic cleaning
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

def main():
    """Main function to process PDF/EPUB files directly from command line"""
    parser = argparse.ArgumentParser(
        description='Clean text from PDF/EPUB files and save cleaned text to ./processed/ directory'
    )
    parser.add_argument('file_path', help='Path to the PDF or EPUB file to process')
    parser.add_argument('--output-dir', '-o', default='./processed',
                       help='Output directory for cleaned text files (default: ./processed)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')

    args = parser.parse_args()

    # Check if file exists
    if not os.path.exists(args.file_path):
        print(f"Error: File not found: {args.file_path}")
        sys.exit(1)

    # Check if file is supported format
    file_extension = os.path.splitext(args.file_path)[1].lower()
    if file_extension not in ['.pdf', '.epub']:
        print(f"Error: Unsupported file format: {file_extension}")
        print("Supported formats: PDF, EPUB")
        sys.exit(1)

    # Import extract module for text extraction
    try:
        from extract import EbookExtractor
    except ImportError:
        print("Error: extract.py module not found. Please ensure it's in the same directory.")
        sys.exit(1)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    if args.verbose:
        print(f"Processing file: {args.file_path}")
        print(f"Output directory: {args.output_dir}")

    try:
        # Initialize components
        extractor = EbookExtractor()
        cleaner = TextCleaner()

        # Extract text from the file
        if args.verbose:
            print("Extracting text from file...")

        original_text = extractor.extract_text(args.file_path)

        if not original_text:
            print("Error: Failed to extract text from the file.")
            sys.exit(1)

        if args.verbose:
            print(f"Extracted {len(original_text)} characters of text")
            if cleaner.is_gutenberg(original_text):
                print("Detected Project Gutenberg-style etext. Will attempt to strip header/footer.")

        # Clean the text
        if args.verbose:
            print("Cleaning text...")

        cleaned_text = cleaner.clean_for_translation(original_text)

        if args.verbose:
            print(f"Cleaned text has {len(cleaned_text)} characters")

        # Generate output filename
        book_name = os.path.splitext(os.path.basename(args.file_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{book_name}_cleaned.txt"
        output_path = os.path.join(args.output_dir, output_filename)

        # Save cleaned text
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)

        print(f"✅ Successfully processed: {args.file_path}")
        print(f"📄 Cleaned text saved to: {output_path}")
        print(f"📊 Original text: {len(original_text):,} characters")
        print(f"📊 Cleaned text: {len(cleaned_text):,} characters")

        if args.verbose:
            print(f"🧹 Cleaning reduced text size by {len(original_text) - len(cleaned_text):,} characters")

    except Exception as e:
        print(f"Error processing file: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
