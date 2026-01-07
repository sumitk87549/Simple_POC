import os
import time
import concurrent.futures
import multiprocessing
from threading import Lock
from functools import partial
import asyncio
import aiohttp
from deep_translator import GoogleTranslator
from gtts import gTTS
import pygame
from pydub import AudioSegment
from pydub.silence import split_on_silence
import tempfile
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import warnings
warnings.filterwarnings('ignore')

class TextTranslatorAndTTS:
    def __init__(self, use_gpu=True, max_workers=None):
        self.supported_languages = {
            'English': 'en',
            'Spanish': 'es',
            'French': 'fr',
            'German': 'de',
            'Italian': 'it',
            'Portuguese': 'pt',
            'Russian': 'ru',
            'Chinese (Simplified)': 'zh-cn',
            'Chinese (Traditional)': 'zh-tw',
            'Japanese': 'ja',
            'Korean': 'ko',
            'Arabic': 'ar',
            'Hindi': 'hi',
            'Dutch': 'nl',
            'Swedish': 'sv',
            'Danish': 'da',
            'Norwegian': 'no',
            'Finnish': 'fi',
            'Polish': 'pl',
            'Turkish': 'tr',
            'Greek': 'el',
            'Hebrew': 'he',
            'Thai': 'th',
            'Vietnamese': 'vi'
        }
        
        # Performance settings
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.cpu_count = multiprocessing.cpu_count()
        
        # Thread safety
        self.translation_lock = Lock()
        self.tts_lock = Lock()
        
        # GPU-based translation model (fallback when Google Translate is slow)
        self.gpu_translator = None
        if self.use_gpu:
            try:
                print("Initializing GPU translation model...")
                model_name = "Helsinki-NLP/opus-mt-en-ROMANCE"  # Multi-lingual model
                self.gpu_translator = pipeline(
                    "translation",
                    model=model_name,
                    device=0 if self.use_gpu else -1,
                    torch_dtype=torch.float16 if self.use_gpu else torch.float32
                )
                print("GPU translation model loaded successfully!")
            except Exception as e:
                print(f"GPU translation model failed to load: {e}")
                self.use_gpu = False
        
        print(f"Translation initialized: GPU={self.use_gpu}, Max Workers={self.max_workers}")
    
    def get_language_code(self, language_name):
        """Get language code from language name"""
        return self.supported_languages.get(language_name, 'en')
    
    def get_supported_languages(self):
        """Get list of supported languages"""
        return list(self.supported_languages.keys())
    
    def translate_text(self, text, target_language, source_language='auto'):
        """Translate text to target language using parallel processing"""
        if not text or not text.strip():
            return ""
        
        try:
            target_lang_code = self.get_language_code(target_language)
            
            # Split text into smaller chunks for parallel processing
            chunks = self._split_text_for_translation(text)
            
            if len(chunks) == 1:
                # For single chunk, use direct translation
                return self._translate_single_chunk(chunks[0], target_lang_code, source_language)
            
            # Use multi-threading for multiple chunks
            return self._translate_chunks_parallel(chunks, target_lang_code, source_language)
            
        except Exception as e:
            print(f"Translation error: {e}")
            return text  # Return original text if translation fails
    
    def _translate_single_chunk(self, chunk, target_lang_code, source_language):
        """Translate a single chunk using GPU if available, otherwise Google Translate"""
        try:
            # Try GPU translation first if available and text is not too long
            if self.use_gpu and self.gpu_translator and len(chunk) < 1000:
                with self.translation_lock:
                    result = self.gpu_translator(chunk)
                    return result[0]['translation_text']
            else:
                # Fall back to Google Translate
                translator = GoogleTranslator(source=source_language, target=target_lang_code)
                result = translator.translate(chunk)
                time.sleep(0.05)  # Reduced delay for faster processing
                return result
        except Exception as e:
            print(f"Error translating chunk: {e}")
            return chunk  # Return original chunk if translation fails
    
    def _translate_chunks_parallel(self, chunks, target_lang_code, source_language):
        """Translate multiple chunks in parallel using ThreadPoolExecutor"""
        translated_chunks = []
        
        # Use ThreadPoolExecutor for I/O-bound translation tasks
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.max_workers, len(chunks))) as executor:
            # Submit all translation tasks
            future_to_chunk = {
                executor.submit(self._translate_single_chunk, chunk, target_lang_code, source_language): chunk 
                for chunk in chunks
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_chunk):
                try:
                    result = future.result()
                    translated_chunks.append(result)
                except Exception as e:
                    chunk = future_to_chunk[future]
                    print(f"Translation failed for chunk: {e}")
                    translated_chunks.append(chunk)  # Keep original chunk
        
        # Maintain original order
        ordered_results = []
        for original_chunk in chunks:
            # Find the corresponding translated chunk
            for i, chunk in enumerate(chunks):
                if chunk == original_chunk and i < len(translated_chunks):
                    ordered_results.append(translated_chunks[i])
                    break
        
        return ' '.join(ordered_results)
    
    def _split_text_for_translation(self, text, max_length=4000):
        """Split text into smaller chunks for translation"""
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
    
    def generate_tts(self, text, language, output_path, slow=False):
        """Generate text-to-speech audio using parallel processing"""
        if not text or not text.strip():
            return False
        
        try:
            lang_code = self.get_language_code(language)
            
            # Split text into smaller chunks for parallel TTS generation
            chunks = self._split_text_for_tts(text)
            
            if len(chunks) == 1:
                # For single chunk, use direct TTS generation
                return self._generate_single_tts(chunks[0], lang_code, output_path, slow)
            
            # Use multi-threading for multiple chunks
            return self._generate_tts_parallel(chunks, lang_code, output_path, slow)
                
        except Exception as e:
            print(f"TTS generation error: {e}")
            return False
    
    def _generate_single_tts(self, text, lang_code, output_path, slow):
        """Generate TTS for a single chunk"""
        try:
            tts = gTTS(text=text, lang=lang_code, slow=slow)
            tts.save(output_path)
            print(f"TTS audio saved to: {output_path}")
            return True
        except Exception as e:
            print(f"Error generating TTS: {e}")
            return False
    
    def _generate_tts_chunk(self, args):
        """Generate TTS for a single chunk (for multiprocessing)"""
        chunk, lang_code, slow, chunk_index = args
        try:
            tts = gTTS(text=chunk, lang=lang_code, slow=slow)
            
            # Create temporary file for this chunk
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'_{chunk_index}.mp3') as temp_file:
                tts.save(temp_file.name)
                return temp_file.name, None
        except Exception as e:
            return None, str(e)
    
    def _generate_tts_parallel(self, chunks, lang_code, output_path, slow):
        """Generate TTS for multiple chunks in parallel"""
        temp_files = []
        
        # Prepare arguments for multiprocessing
        args_list = [(chunk, lang_code, slow, i) for i, chunk in enumerate(chunks)]
        
        # Use ProcessPoolExecutor for CPU-intensive TTS generation
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(self.cpu_count, len(chunks))) as executor:
            # Submit all TTS tasks
            future_to_index = {
                executor.submit(self._generate_tts_chunk, args): args[3] 
                for args in args_list
            }
            
            # Collect results as they complete
            results = {}
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    temp_file, error = future.result()
                    if error:
                        print(f"Error generating TTS for chunk {index}: {error}")
                    else:
                        results[index] = temp_file
                        print(f"Generated TTS for chunk {index + 1}/{len(chunks)}")
                except Exception as e:
                    print(f"TTS generation failed for chunk {index}: {e}")
            
            # Sort results by index to maintain order
            sorted_temp_files = [results[i] for i in range(len(chunks)) if i in results]
            
            if sorted_temp_files:
                # Combine all audio segments
                audio_segments = []
                for temp_file in sorted_temp_files:
                    try:
                        audio = AudioSegment.from_mp3(temp_file)
                        audio_segments.append(audio)
                        os.unlink(temp_file)  # Clean up temp file
                    except Exception as e:
                        print(f"Error loading audio segment: {e}")
                
                if audio_segments:
                    # Combine all audio segments
                    final_audio = sum(audio_segments)
                    
                    # Export final audio
                    final_audio.export(output_path, format="mp3")
                    print(f"TTS audio saved to: {output_path}")
                    return True
                else:
                    print("No valid audio segments were generated")
                    return False
            else:
                print("No audio segments were generated")
                return False
    
    def _split_text_for_tts(self, text, max_length=500):
        """Split text into smaller chunks for TTS"""
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
    
    async def process_book_translation_async(self, original_text, target_language, book_name, output_folder):
        """Async version of complete book translation and TTS"""
        try:
            # Create output folder if it doesn't exist
            os.makedirs(output_folder, exist_ok=True)
            
            # Translate text asynchronously
            print(f"Translating to {target_language}...")
            
            # Run translation in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            translated_text = await loop.run_in_executor(
                None, 
                self.translate_text, 
                original_text, 
                target_language
            )
            
            # Save translated text
            translated_file_path = os.path.join(output_folder, f"{book_name}_translated.txt")
            await self._save_text_async(translated_file_path, translated_text)
            print(f"Translated text saved to: {translated_file_path}")
            
            # Generate TTS audio asynchronously
            print("Generating audio...")
            audio_file_path = os.path.join(output_folder, f"{book_name}_audio.mp3")
            
            # Run TTS generation in thread pool
            success = await loop.run_in_executor(
                None,
                self.generate_tts,
                translated_text,
                target_language,
                audio_file_path
            )
            
            if success:
                print(f"Audio saved to: {audio_file_path}")
                return {
                    'success': True,
                    'translated_file': translated_file_path,
                    'audio_file': audio_file_path,
                    'translated_text': translated_text
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to generate audio',
                    'translated_file': translated_file_path,
                    'translated_text': translated_text
                }
                
        except Exception as e:
            print(f"Error processing book translation: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _save_text_async(self, file_path, text):
        """Save text to file asynchronously"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._save_text_sync,
            file_path,
            text
        )
    
    def _save_text_sync(self, file_path, text):
        """Synchronous text saving"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)
    
    def process_book_translation(self, original_text, target_language, book_name, output_folder):
        """Process complete book translation and TTS (sync wrapper)"""
        try:
            # Try to run async version if available
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self.process_book_translation_async(
                        original_text, target_language, book_name, output_folder
                    )
                )
                return result
            finally:
                loop.close()
        except Exception as e:
            # Fall back to synchronous processing
            print(f"Async processing failed, falling back to sync: {e}")
            return self._process_book_translation_sync(
                original_text, target_language, book_name, output_folder
            )
    
    def _process_book_translation_sync(self, original_text, target_language, book_name, output_folder):
        """Synchronous fallback for book processing"""
        try:
            # Create output folder if it doesn't exist
            os.makedirs(output_folder, exist_ok=True)
            
            # Translate text
            print(f"Translating to {target_language}...")
            translated_text = self.translate_text(original_text, target_language)
            
            # Save translated text
            translated_file_path = os.path.join(output_folder, f"{book_name}_translated.txt")
            with open(translated_file_path, 'w', encoding='utf-8') as f:
                f.write(translated_text)
            print(f"Translated text saved to: {translated_file_path}")
            
            # Generate TTS audio
            print("Generating audio...")
            audio_file_path = os.path.join(output_folder, f"{book_name}_audio.mp3")
            success = self.generate_tts(translated_text, target_language, audio_file_path)
            
            if success:
                print(f"Audio saved to: {audio_file_path}")
                return {
                    'success': True,
                    'translated_file': translated_file_path,
                    'audio_file': audio_file_path,
                    'translated_text': translated_text
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to generate audio',
                    'translated_file': translated_file_path,
                    'translated_text': translated_text
                }
                
        except Exception as e:
            print(f"Error processing book translation: {e}")
            return {
                'success': False,
                'error': str(e)
            }