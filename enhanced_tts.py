import os
import time
import concurrent.futures
import multiprocessing
from threading import Lock
import asyncio
import tempfile
from gtts import gTTS
from pydub import AudioSegment
import re
import warnings
warnings.filterwarnings('ignore')

class EnhancedTTS:
    """
    Simplified Text-to-Speech using gTTS with parallel processing
    Focused on stability and compatibility for translation purposes
    """
    
    def __init__(self, use_gpu=True, max_workers=None):
        """
        Initialize TTS with gTTS engine
        
        Args:
            use_gpu (bool): GPU setting (kept for compatibility, gTTS doesn't use GPU)
            max_workers (int): Maximum number of parallel workers
        """
        # Performance settings
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.cpu_count = multiprocessing.cpu_count()
        
        # Thread safety
        self.tts_lock = Lock()
        
        # gTTS is always available and stable
        self.engine_name = 'gtts'
        
        print(f"Simplified TTS initialized: Engine={self.engine_name}, Max Workers={self.max_workers}")
    
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
        Generate TTS audio using gTTS with simple sequential processing
        
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
            
            # For simplicity and reliability, process the entire text at once
            # gTTS handles long texts well and this avoids all threading issues
            return self._generate_single_chunk(text, language, output_path, slow)
            
        except Exception as e:
            print(f"TTS generation error: {e}")
            return False
    
    def _generate_single_chunk(self, text, language, output_path, slow):
        """Generate TTS for a single text chunk using gTTS"""
        try:
            lang_code = self._get_language_code(language)
            tts = gTTS(text=text, lang=lang_code, slow=slow)
            tts.save(output_path)
            print(f"gTTS audio saved to: {output_path}")
            return True
        except Exception as e:
            print(f"gTTS generation error: {e}")
            return False
    
    def _generate_tts_chunk_worker(self, args):
        """Worker function for parallel TTS generation"""
        text, language, slow, chunk_index, temp_dir = args
        
        try:
            # Create temporary file for this chunk
            temp_file = os.path.join(temp_dir, f"chunk_{chunk_index:04d}.mp3")
            
            success = self._generate_single_chunk(text, language, temp_file, slow)
            
            if success:
                return temp_file, None
            else:
                return None, f"Failed to generate chunk {chunk_index}"
                
        except Exception as e:
            return None, str(e)
    
    def _generate_parallel_chunks(self, chunks, language, output_path, slow):
        """Generate TTS for multiple chunks in parallel"""
        temp_dir = tempfile.mkdtemp(prefix="tts_parallel_")
        
        try:
            # Prepare arguments for parallel processing
            args_list = [
                (chunk, language, slow, i, temp_dir)
                for i, chunk in enumerate(chunks)
            ]
            
            # Use ThreadPoolExecutor for I/O-bound TTS generation (gTTS makes network requests)
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(self.max_workers, len(chunks))
            ) as executor:
                
                # Submit all TTS tasks
                future_to_index = {
                    executor.submit(self._generate_tts_chunk_worker, args): args[3]
                    for args in args_list
                }
                
                # Collect results
                results = {}
                for future in concurrent.futures.as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        temp_file, error = future.result()
                        if error:
                            print(f"TTS generation failed for chunk {index}: {error}")
                        else:
                            results[index] = temp_file
                            print(f"Generated TTS for chunk {index + 1}/{len(chunks)}")
                    except Exception as e:
                        print(f"TTS generation failed for chunk {index}: {e}")
                
                # Sort results by index to maintain order
                sorted_temp_files = [results[i] for i in range(len(chunks)) if i in results]
                
                if sorted_temp_files:
                    # Combine all audio files
                    self._combine_audio_files(sorted_temp_files, output_path)
                    print(f"Combined TTS audio saved to: {output_path}")
                    return True
                else:
                    print("No audio segments were generated")
                    return False
                    
        finally:
            # Clean up temporary directory
            try:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as e:
                print(f"Error cleaning up temp directory: {e}")
    
    def _combine_audio_files(self, audio_files, output_path):
        """Combine multiple audio files into one"""
        combined_audio = None
        
        for audio_file in audio_files:
            try:
                audio = AudioSegment.from_mp3(audio_file)
                if combined_audio is None:
                    combined_audio = audio
                else:
                    combined_audio += audio
            except Exception as e:
                print(f"Error loading audio file {audio_file}: {e}")
        
        if combined_audio:
            combined_audio.export(output_path, format="mp3")
        else:
            raise RuntimeError("No valid audio segments to combine")
    
    def _split_text_for_tts(self, text, max_length=500):
        """Split text into optimal chunks for TTS processing"""
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        sentences = text.split('. ')
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_length:
                current_chunk += sentence + '. '
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + '. '
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
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
    
    async def generate_tts_async(self, text, language, output_path=None, book_name=None, slow=False):
        """Async version of TTS generation"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.generate_tts,
            text,
            language,
            output_path,
            book_name,
            slow
        )
    
    def get_performance_info(self):
        """Get performance and system information"""
        return {
            'engine': 'gTTS',
            'cpu_count': self.cpu_count,
            'max_workers': self.max_workers,
            'parallel_processing': True,
            'gpu_acceleration': False  # gTTS doesn't use GPU
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize simplified TTS
    tts = EnhancedTTS()
    
    # Print performance info
    print("Performance Info:", tts.get_performance_info())
    
    # Test TTS generation
    test_text = "Hello, this is a test of the simplified text-to-speech system using gTTS with parallel processing."
    
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
