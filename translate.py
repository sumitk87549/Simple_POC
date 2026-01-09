import os
import time
import asyncio
import aiohttp
from deep_translator import GoogleTranslator
from gtts import gTTS
import pygame
from pydub import AudioSegment
from pydub.silence import split_on_silence
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import re
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
        self.cpu_count = os.cpu_count()
        
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
    
    def _create_processed_folder(self, book_name):
        """Create a processed folder with timestamp for the book"""
        # Clean book name for filesystem
        clean_book_name = re.sub(r'[^a-zA-Z0-9_-]', '_', book_name)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        folder_name = f"{clean_book_name}_{timestamp}"
        processed_folder = os.path.join("./processed", folder_name)
        os.makedirs(processed_folder, exist_ok=True)
        return processed_folder
    
    def get_language_code(self, language_name):
        """Get language code from language name"""
        return self.supported_languages.get(language_name, 'en')
    
    def get_supported_languages(self):
        """Get list of supported languages"""
        return list(self.supported_languages.keys())
    
    def translate_text(self, text, target_language, source_language='auto'):
        """Translate text to target language"""
        if not text or not text.strip():
            return ""
        
        try:
            target_lang_code = self.get_language_code(target_language)
            
            # For simplicity, process text directly without complex chunking
            return self._translate_single_chunk(text, target_lang_code, source_language)
            
        except Exception as e:
            print(f"Translation error: {e}")
            return text  # Return original text if translation fails
    
    def _translate_single_chunk(self, chunk, target_lang_code, source_language):
        """Translate a single chunk using GPU if available, otherwise Google Translate"""
        try:
            # Try GPU translation first if available and text is not too long
            if self.use_gpu and self.gpu_translator and len(chunk) < 1000:
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
        """Generate text-to-speech audio using single-threaded processing"""
        if not text or not text.strip():
            return False
        
        try:
            lang_code = self.get_language_code(language)
            
            # Always ensure the file is saved in the processed folder structure
            if not output_path.startswith("./processed/"):
                # Extract filename from the path and create processed folder structure
                filename = os.path.basename(output_path)
                output_path = os.path.join("./processed", filename)
            
            # Create processed directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # For simplicity and reliability, process the entire text at once
            # gTTS handles long texts well and this avoids all threading issues
            return self._generate_single_tts(text, lang_code, output_path, slow)
                
        except Exception as e:
            print(f"TTS generation error: {e}")
            return False
    
    def _generate_single_tts(self, text, lang_code, output_path, slow):
        """Generate TTS for a single chunk"""
        try:
            # Always ensure the file is saved in the processed folder structure
            if not output_path.startswith("./processed/"):
                # Extract filename from the path and create processed folder structure
                filename = os.path.basename(output_path)
                output_path = os.path.join("./processed", filename)
            
            # Create processed directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            tts = gTTS(text=text, lang=lang_code, slow=slow)
            tts.save(output_path)
            print(f"TTS audio saved to: {output_path}")
            return True
        except Exception as e:
            print(f"Error generating TTS: {e}")
            return False
    
    async def process_book_translation_async(self, original_text, target_language, book_name, output_folder=None):
        """Async version of complete book translation and TTS"""
        try:
            # Create processed folder if output_folder not specified
            if output_folder is None:
                output_folder = self._create_processed_folder(book_name)
            else:
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
    
    def process_book_translation(self, original_text, target_language, book_name, output_folder=None):
        """Process complete book translation and TTS (sync wrapper)"""
        try:
            # Create processed folder if output_folder not specified
            if output_folder is None:
                output_folder = self._create_processed_folder(book_name)
            
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
            # Create processed folder if output_folder not specified
            if output_folder is None:
                output_folder = self._create_processed_folder(book_name)
            else:
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
