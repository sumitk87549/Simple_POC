import os
import time
import tempfile
from gtts import gTTS
from pydub import AudioSegment
import re
import warnings
warnings.filterwarnings('ignore')

class EnhancedTTS:
    """
    Simplified Text-to-Speech using gTTS with single-threaded processing
    Focused on stability and compatibility for translation purposes
    """
    
    def __init__(self, use_gpu=True, max_workers=None):
        """
        Initialize TTS with gTTS engine
        
        Args:
            use_gpu (bool): GPU setting (kept for compatibility, gTTS doesn't use GPU)
            max_workers (int): Maximum number of parallel workers (ignored for gTTS)
        """
        # gTTS is always available and stable
        self.engine_name = 'gtts'
        
        print(f"Simplified TTS initialized: Engine={self.engine_name}, Single-threaded mode")
    
    def _create_processed_folder(self, book_name):
        """Create a processed folder with timestamp for the book"""
        # Clean book name for filesystem
        clean_book_name = re.sub(r'[^a-zA-Z0-9_-]', '_', book_name)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        folder_name = f"{clean_book_name}_{timestamp}"
        processed_folder = os.path.join("./processed", folder_name)
        os.makedirs(processed_folder, exist_ok=True)
        return processed_folder
    
    def get_supported_languages(self):
        """Get supported languages for gTTS"""
        return {
            'English': 'en', 'Spanish': 'es', 'French': 'fr', 'German': 'de',
            'Italian': 'it', 'Portuguese': 'pt', 'Russian': 'ru', 'Chinese (Simplified)': 'zh',
            'Japanese': 'ja', 'Korean': 'ko', 'Arabic': 'ar', 'Hindi': 'hi',
            'Dutch': 'nl', 'Polish': 'pl', 'Turkish': 'tr', 'Greek': 'el',
            'Swedish': 'sv', 'Danish': 'da', 'Norwegian': 'no', 'Finnish': 'fi'
        }
    
    def generate_tts(self, text, language, output_path=None, book_name=None, slow=False):
        """
        Generate TTS audio using gTTS with single-threaded processing
        
        Args:
            text (str): Text to convert to speech
            language (str): Target language
            output_path (str): Output audio file path (optional if book_name provided)
            book_name (str): Book name for automatic folder creation (optional)
            slow (bool): Slow speech mode
        """
        if not text or not text.strip():
            return False
        
        try:
            # Create output path if not provided but book_name is given
            if output_path is None and book_name is not None:
                output_folder = self._create_processed_folder(book_name)
                output_path = os.path.join(output_folder, f"{book_name}_audio.mp3")
            elif output_path is None:
                raise ValueError("Either output_path or book_name must be provided")
            
            # Always ensure the file is saved in the processed folder structure
            if not output_path.startswith("./processed/"):
                # Extract filename from the path and create processed folder structure
                filename = os.path.basename(output_path)
                output_path = os.path.join("./processed", filename)
            
            # Create processed directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # For simplicity and reliability, process the entire text at once
            # gTTS handles long texts well and this avoids all threading issues
            return self._generate_single_chunk(text, language, output_path, slow)
            
        except Exception as e:
            print(f"TTS generation error: {e}")
            return False
    
    def _generate_single_chunk(self, text, language, output_path, slow):
        """Generate TTS for a single text chunk using gTTS"""
        try:
            # Always ensure the file is saved in the processed folder structure
            if not output_path.startswith("./processed/"):
                # Extract filename from the path and create processed folder structure
                filename = os.path.basename(output_path)
                output_path = os.path.join("./processed", filename)
            
            # Create processed directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            lang_code = self._get_language_code(language)
            tts = gTTS(text=text, lang=lang_code, slow=slow)
            tts.save(output_path)
            print(f"gTTS audio saved to: {output_path}")
            return True
        except Exception as e:
            print(f"gTTS generation error: {e}")
            return False
    
    def _get_language_code(self, language):
        """Get language code for gTTS"""
        language_map = {
            'English': 'en', 'Spanish': 'es', 'French': 'fr', 'German': 'de',
            'Italian': 'it', 'Portuguese': 'pt', 'Russian': 'ru', 'Chinese (Simplified)': 'zh-cn',
            'Chinese (Traditional)': 'zh-tw', 'Japanese': 'ja', 'Korean': 'ko',
            'Arabic': 'ar', 'Hindi': 'hi', 'Dutch': 'nl', 'Polish': 'pl',
            'Turkish': 'tr', 'Greek': 'el', 'Swedish': 'sv', 'Danish': 'da',
            'Norwegian': 'no', 'Finnish': 'fi'
        }
        return language_map.get(language, 'en')
    
    def get_performance_info(self):
        """Get performance and system information"""
        return {
            'engine': 'gTTS',
            'processing_mode': 'Single-threaded',
            'parallel_processing': False,
            'gpu_acceleration': False  # gTTS doesn't use GPU
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize simplified TTS
    tts = EnhancedTTS()
    
    # Print performance info
    print("Performance Info:", tts.get_performance_info())
    
    # Test TTS generation
    test_text = "Hello, this is a test of the simplified text-to-speech system using gTTS with single-threaded processing."
    
    # Generate TTS
    success = tts.generate_tts(
        text=test_text,
        language="English",
        book_name="test_simplified_tts",
        slow=False
    )
    
    if success:
        print("Simplified TTS generation successful!")
    else:
        print("Simplified TTS generation failed!")
