import os
import PyPDF2
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import magic

class EbookExtractor:
    def __init__(self):
        self.supported_formats = ['.pdf', '.epub']
    
    def get_file_type(self, file_path):
        """Get file type using python-magic"""
        try:
            mime = magic.Magic(mime=True)
            file_type = mime.from_file(file_path)
            return file_type
        except Exception as e:
            print(f"Error detecting file type: {e}")
            return None
    
    def extract_text_from_pdf(self, file_path):
        """Extract text from PDF file"""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return None
    
    def extract_text_from_epub(self, file_path):
        """Extract text from EPUB file"""
        try:
            text = ""
            book = epub.read_epub(file_path)
            
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    text += soup.get_text() + "\n"
            
            return text
        except Exception as e:
            print(f"Error extracting text from EPUB: {e}")
            return None
    
    def extract_text(self, file_path):
        """Main method to extract text from ebook"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_extension == '.epub':
            return self.extract_text_from_epub(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def is_supported_format(self, file_path):
        """Check if file format is supported"""
        file_extension = os.path.splitext(file_path)[1].lower()
        return file_extension in self.supported_formats