import os
import time
import concurrent.futures
import multiprocessing
from threading import Lock
import asyncio
import tempfile
import torch
import numpy as np
from gtts import gTTS
from pydub import AudioSegment
from TTS.api import TTS
from TTS.tts.configs.xtts_v2_config import XttsV2Config
from TTS.tts.models.xtts_v2 import Xtts_v2
import re
import warnings
warnings.filterwarnings('ignore')

class EnhancedTTS:
    """
    Enhanced Text-to-Speech with GPU acceleration and parallel processing
    Supports multiple TTS engines for optimal performance
    """
    
    def __init__(self, use_gpu=True, max_workers=None, preferred_engine="auto"):
        """
        Initialize Enhanced TTS with GPU support and parallel processing
        
        Args:
            use_gpu (bool): Enable GPU acceleration if available
            max_workers (int): Maximum number of parallel workers
            preferred_engine (str): Preferred TTS engine ("auto", "gtts", "coqui", "xtts")
        """
        # Performance settings
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.cpu_count = multiprocessing.cpu_count()
        
        # Thread safety
        self.tts_lock = Lock()
        
        # Available TTS engines
        self.engines = {
            'gtts': self._init_gtts,
            'coqui': self._init_coqui_tts,
            'xtts': self._init_xtts
        }
        
        # Initialize preferred engine
        self.current_engine = None
        self.engine_name = preferred_engine
        self._initialize_engine(preferred_engine)
        
        print(f"Enhanced TTS initialized: GPU={self.use_gpu}, Engine={self.engine_name}, Max Workers={self.max_workers}")
    
    def _create_processed_folder(self, book_name):
        """Create a processed folder with timestamp for the book"""
        # Clean book name for filesystem
        clean_book_name = re.sub(r'[^a-zA-Z0-9_-]', '_', book_name)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        folder_name = f"{clean_book_name}_{timestamp}"
        processed_folder = os.path.join("./processed", folder_name)
        os.makedirs(processed_folder, exist_ok=True)
        return processed_folder
    
    def _initialize_engine(self, preferred_engine):
        """Initialize the preferred TTS engine with fallback options"""
        if preferred_engine == "auto":
            # Try engines in order of preference
            engine_order = ['xtts', 'coqui', 'gtts']
        else:
            engine_order = [preferred_engine] + [e for e in ['xtts', 'coqui', 'gtts'] if e != preferred_engine]
        
        for engine in engine_order:
            try:
                self.current_engine = self.engines[engine]()
                self.engine_name = engine
                print(f"Successfully initialized {engine} TTS engine")
                return
            except Exception as e:
                print(f"Failed to initialize {engine} engine: {e}")
                continue
        
        raise RuntimeError("Failed to initialize any TTS engine")
    
    def _init_gtts(self):
        """Initialize Google TTS (always available as fallback)"""
        return {'type': 'gtts', 'model': None}
    
    def _init_coqui_tts(self):
        """Initialize Coqui TTS with GPU support"""
        try:
            # Try to load a multilingual model
            model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
            
            # Initialize TTS
            tts = TTS(model_name)
            
            if self.use_gpu:
                tts = tts.to("cuda")
            
            return {'type': 'coqui', 'model': tts}
        except Exception as e:
            print(f"Coqui TTS initialization failed: {e}")
            raise
    
    def _init_xtts(self):
        """Initialize XTTS v2 for high-quality multilingual TTS"""
        try:
            config = XttsV2Config()
            model = Xtts_v2.init_from_config(config)
            
            if self.use_gpu:
                model = model.cuda()
            
            return {'type': 'xtts', 'model': model}
        except Exception as e:
            print(f"XTTS initialization failed: {e}")
            raise
    
    def get_supported_languages(self):
        """Get supported languages for current engine"""
        if self.engine_name == 'gtts':
            return {
                'English': 'en', 'Spanish': 'es', 'French': 'fr', 'German': 'de',
                'Italian': 'it', 'Portuguese': 'pt', 'Russian': 'ru', 'Chinese': 'zh',
                'Japanese': 'ja', 'Korean': 'ko', 'Arabic': 'ar', 'Hindi': 'hi'
            }
        elif self.engine_name in ['coqui', 'xtts']:
            # These engines support many more languages
            return {
                'English': 'en', 'Spanish': 'es', 'French': 'fr', 'German': 'de',
                'Italian': 'it', 'Portuguese': 'pt', 'Russian': 'ru', 'Chinese': 'zh-cn',
                'Japanese': 'ja', 'Korean': 'ko', 'Arabic': 'ar', 'Hindi': 'hi',
                'Dutch': 'nl', 'Polish': 'pl', 'Turkish': 'tr', 'Greek': 'el'
            }
        else:
            return {}
    
    def generate_tts(self, text, language, output_path=None, book_name=None, slow=False, voice_preset=None):
        """
        Generate TTS audio with parallel processing and GPU acceleration
        
        Args:
            text (str): Text to convert to speech
            language (str): Target language
            output_path (str): Output audio file path (optional if book_name provided)
            book_name (str): Book name for automatic folder creation (optional)
            slow (bool): Slow speech mode
            voice_preset (str): Voice preset for advanced engines
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
            
            # Split text into chunks for parallel processing
            chunks = self._split_text_for_tts(text)
            
            if len(chunks) == 1:
                # Single chunk - direct processing
                return self._generate_single_chunk(
                    chunks[0], language, output_path, slow, voice_preset
                )
            
            # Multiple chunks - parallel processing
            return self._generate_parallel_chunks(
                chunks, language, output_path, slow, voice_preset
            )
            
        except Exception as e:
            print(f"TTS generation error: {e}")
            return False
    
    def _generate_single_chunk(self, text, language, output_path, slow, voice_preset):
        """Generate TTS for a single text chunk"""
        try:
            if self.engine_name == 'gtts':
                return self._generate_gtts(text, language, output_path, slow)
            elif self.engine_name in ['coqui', 'xtts']:
                return self._generate_advanced_tts(text, language, output_path, voice_preset)
            else:
                raise ValueError(f"Unsupported engine: {self.engine_name}")
        except Exception as e:
            print(f"Error generating single chunk TTS: {e}")
            return False
    
    def _generate_gtts(self, text, language, output_path, slow):
        """Generate TTS using Google TTS"""
        try:
            lang_code = self._get_language_code(language)
            tts = gTTS(text=text, lang=lang_code, slow=slow)
            tts.save(output_path)
            print(f"gTTS audio saved to: {output_path}")
            return True
        except Exception as e:
            print(f"gTTS generation error: {e}")
            return False
    
    def _generate_advanced_tts(self, text, language, output_path, voice_preset):
        """Generate TTS using advanced engines (Coqui/XTTS)"""
        try:
            with self.tts_lock:
                if self.engine_name == 'coqui':
                    model = self.current_engine['model']
                    lang_code = self._get_language_code(language)
                    
                    # Generate speech
                    wav = model.tts(text, speaker=voice_preset, language=lang_code)
                    
                    # Save to file
                    model.save_wav(wav, output_path)
                    
                elif self.engine_name == 'xtts':
                    model = self.current_engine['model']
                    lang_code = self._get_language_code(language)
                    
                    # Generate speech with XTTS
                    wav = model.synthesize(text, speaker=voice_preset, language=lang_code)
                    
                    # Save to file
                    import soundfile as sf
                    sf.write(output_path, wav, 22050)
            
            print(f"{self.engine_name} audio saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"{self.engine_name} generation error: {e}")
            return False
    
    def _generate_tts_chunk_worker(self, args):
        """Worker function for parallel TTS generation"""
        text, language, slow, voice_preset, chunk_index, temp_dir = args
        
        try:
            # Create temporary file for this chunk
            temp_file = os.path.join(temp_dir, f"chunk_{chunk_index:04d}.mp3")
            
            success = self._generate_single_chunk(text, language, temp_file, slow, voice_preset)
            
            if success:
                return temp_file, None
            else:
                return None, f"Failed to generate chunk {chunk_index}"
                
        except Exception as e:
            return None, str(e)
    
    def _generate_parallel_chunks(self, chunks, language, output_path, slow, voice_preset):
        """Generate TTS for multiple chunks in parallel"""
        temp_dir = tempfile.mkdtemp(prefix="tts_parallel_")
        
        try:
            # Prepare arguments for parallel processing
            args_list = [
                (chunk, language, slow, voice_preset, i, temp_dir)
                for i, chunk in enumerate(chunks)
            ]
            
            # Use ProcessPoolExecutor for CPU-intensive TTS generation
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=min(self.cpu_count, len(chunks))
            ) as executor:
                
                # Submit all TTS tasks
                future_to_index = {
                    executor.submit(self._generate_tts_chunk_worker, args): args[4]
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
                    # Combine all audio segments
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
        """Get language code for current TTS engine"""
        language_map = {
            'English': 'en', 'Spanish': 'es', 'French': 'fr', 'German': 'de',
            'Italian': 'it', 'Portuguese': 'pt', 'Russian': 'ru', 'Chinese (Simplified)': 'zh-cn',
            'Chinese (Traditional)': 'zh-tw', 'Japanese': 'ja', 'Korean': 'ko',
            'Arabic': 'ar', 'Hindi': 'hi', 'Dutch': 'nl', 'Polish': 'pl',
            'Turkish': 'tr', 'Greek': 'el', 'Swedish': 'sv', 'Danish': 'da',
            'Norwegian': 'no', 'Finnish': 'fi'
        }
        return language_map.get(language, 'en')
    
    async def generate_tts_async(self, text, language, output_path=None, book_name=None, slow=False, voice_preset=None):
        """Async version of TTS generation"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.generate_tts,
            text,
            language,
            output_path,
            book_name,
            slow,
            voice_preset
        )
    
    def get_performance_info(self):
        """Get performance and system information"""
        return {
            'gpu_available': torch.cuda.is_available(),
            'gpu_used': self.use_gpu,
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'cpu_count': self.cpu_count,
            'max_workers': self.max_workers,
            'current_engine': self.engine_name,
            'torch_version': torch.__version__,
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize enhanced TTS
    tts = EnhancedTTS(use_gpu=True, preferred_engine="auto")
    
    # Print performance info
    print("Performance Info:", tts.get_performance_info())
    
    # Test TTS generation
    test_text = "Hello, this is a test of the enhanced text-to-speech system with parallel processing and GPU acceleration."
    
    # Generate TTS
    success = tts.generate_tts(
        text=test_text,
        language="English",
        book_name="test_enhanced_tts",
        slow=False
    )
    
    if success:
        print("Enhanced TTS generation successful!")
    else:
        print("Enhanced TTS generation failed!")