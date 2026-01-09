import os
import time
import concurrent.futures
import multiprocessing
from threading import Lock
import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    MBart50TokenizerFast,
    MBartForConditionalGeneration,
    M2M100ForConditionalGeneration,
    M2M100Tokenizer
)
import re
import warnings
from typing import List, Dict, Optional, Tuple

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pygame')
warnings.filterwarnings('ignore', message='.*torch.classes.*')
warnings.filterwarnings('ignore', message='.*pkg_resources.*')
warnings.filterwarnings('ignore', message='.*meta tensor.*')
warnings.filterwarnings('ignore', message='.*Cannot copy out of meta tensor.*')

# Set environment variable to prevent tokenizer parallelism issues
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class DeepTranslator:
    """
    Advanced AI Translation using Hugging Face Transformers models
    Supports mBART-50, M2M100, and MarianMT models with configurable parameters
    """

    def __init__(self, model_name: str = "facebook/mbart-large-50-many-to-many-mmt",
                 chunk_size: int = 4000, max_threads: int = 4, use_gpu: bool = True):
        """
        Initialize DeepTranslator with specified model and parameters

        Args:
            model_name: Hugging Face model identifier
            chunk_size: Maximum characters per translation chunk
            max_threads: Number of parallel threads for processing
            use_gpu: Whether to use GPU acceleration
        """
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.max_threads = max_threads
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"
        self.translation_lock = Lock()

        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.supported_models = self._get_supported_models()
        self._initialize_model()

        print(f"DeepTranslator initialized: Model={model_name}, GPU={self.use_gpu}, "
              f"Threads={max_threads}, Chunk Size={chunk_size}")

    def _get_supported_models(self) -> Dict[str, str]:
        """Get dictionary of supported models with display names"""
        return {
            "mBART-50 (Facebook)": "facebook/mbart-large-50-many-to-many-mmt",
            "M2M100 (Facebook)": "facebook/m2m100_418M",
            "MarianMT (OPUS)": "Helsinki-NLP/opus-mt-en-ROMANCE",
            "NLLB (Meta)": "facebook/nllb-200-distilled-600M"
        }
    
    def _create_processed_folder(self, book_name):
        """Create a processed folder with timestamp for the book"""
        # Clean book name for filesystem
        clean_book_name = re.sub(r'[^a-zA-Z0-9_-]', '_', book_name)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        folder_name = f"{clean_book_name}_{timestamp}"
        processed_folder = os.path.join("./processed", folder_name)
        os.makedirs(processed_folder, exist_ok=True)
        return processed_folder
    
    def save_translated_text(self, text: str, book_name: str, target_language: str) -> str:
        """
        Save translated text to a file in the processed folder
        
        Args:
            text: Translated text to save
            book_name: Name of the book for file naming
            target_language: Target language for the translation
            
        Returns:
            Path to the saved file
        """
        try:
            # Create processed folder structure
            output_folder = self._create_processed_folder(book_name)
            
            # Create filename with language information
            clean_book_name = re.sub(r'[^a-zA-Z0-9_-]', '_', book_name)
            filename = f"{clean_book_name}_translated_{target_language.replace(' ', '_')}.txt"
            file_path = os.path.join(output_folder, filename)
            
            # Save the translated text
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            print(f"Translated text saved to: {file_path}")
            return file_path
            
        except Exception as e:
            print(f"Error saving translated text: {e}")
            return ""

    def get_available_models(self) -> List[str]:
        """Get list of available model display names"""
        return list(self.supported_models.keys())

    def _initialize_model(self) -> bool:
        """Initialize the selected Hugging Face model"""
        try:
            print(f"Loading model: {self.model_name}")

            # Model-specific initialization
            if "mbart" in self.model_name.lower():
                self.tokenizer = MBart50TokenizerFast.from_pretrained(self.model_name, src_lang="en_XX")
                self.model = MBartForConditionalGeneration.from_pretrained(self.model_name)
            elif "m2m" in self.model_name.lower():
                self.tokenizer = M2M100Tokenizer.from_pretrained(self.model_name, src_lang="en")
                self.model = M2M100ForConditionalGeneration.from_pretrained(self.model_name)
            else:
                # Generic pipeline for other models
                self.pipeline = pipeline(
                    "translation",
                    model=self.model_name,
                    device=self.device,
                    torch_dtype=torch.float16 if self.use_gpu else torch.float32
                )
                return True

            # Move model to appropriate device with proper handling for meta tensors
            if self.model:
                try:
                    # First try the standard approach
                    if self.use_gpu:
                        self.model = self.model.to("cuda")
                        self.model = self.model.half()  # Use FP16 for GPU
                    else:
                        self.model = self.model.to("cpu")
                except Exception as e:
                    if "meta tensor" in str(e).lower():
                        # Handle meta tensor case by using to_empty() first
                        print("Detected meta tensor, using to_empty() approach...")
                        if self.use_gpu:
                            self.model = self.model.to_empty(device="cuda")
                            self.model = self.model.half()  # Use FP16 for GPU
                        else:
                            self.model = self.model.to_empty(device="cpu")
                    else:
                        raise e

            print("Model loaded successfully!")
            return True

        except Exception as e:
            print(f"Failed to load model {self.model_name}: {e}")
            return False

    def _get_target_language_code(self, target_language: str) -> str:
        """Convert target language name to model-specific language code"""
        # Common language mappings
        language_codes = {
            'English': 'en',
            'Spanish': 'es',
            'French': 'fr',
            'German': 'de',
            'Italian': 'it',
            'Portuguese': 'pt',
            'Russian': 'ru',
            'Chinese (Simplified)': 'zh',
            'Japanese': 'ja',
            'Korean': 'ko',
            'Arabic': 'ar',
            'Hindi': 'hi'
        }

        # mBART-50 specific codes
        mbart_codes = {
            'English': 'en_XX',
            'Spanish': 'es_XX',
            'French': 'fr_XX',
            'German': 'de_DE',
            'Italian': 'it_IT',
            'Portuguese': 'pt_XX',
            'Russian': 'ru_RU',
            'Chinese (Simplified)': 'zh_CN',
            'Japanese': 'ja_XX',
            'Korean': 'ko_KR',
            'Arabic': 'ar_AR',
            'Hindi': 'hi_IN'
        }

        # M2M100 specific codes
        m2m_codes = {
            'English': 'en',
            'Spanish': 'es',
            'French': 'fr',
            'German': 'de',
            'Italian': 'it',
            'Portuguese': 'pt',
            'Russian': 'ru',
            'Chinese (Simplified)': 'zh',
            'Japanese': 'ja',
            'Korean': 'ko',
            'Arabic': 'ar',
            'Hindi': 'hi'
        }

        if "mbart" in self.model_name.lower():
            return mbart_codes.get(target_language, 'en_XX')
        elif "m2m" in self.model_name.lower():
            return m2m_codes.get(target_language, 'en')
        else:
            return language_codes.get(target_language, 'en')

    def _split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into chunks based on chunk_size parameter"""
        if not text or len(text) <= self.chunk_size:
            return [text]

        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += (" " + sentence if current_chunk else sentence)
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _translate_chunk(self, chunk: str, target_language: str) -> str:
        """Translate a single text chunk using the loaded model"""
        try:
            if not chunk or not chunk.strip():
                return ""

            target_lang_code = self._get_target_language_code(target_language)

            # Model-specific translation
            if "mbart" in self.model_name.lower():
                # mBART-50 translation
                self.tokenizer.src_lang = "en_XX"
                inputs = self.tokenizer(chunk, return_tensors="pt", truncation=True, max_length=1024)

                # Move inputs to the correct device
                try:
                    if self.use_gpu:
                        inputs = inputs.to("cuda")
                    else:
                        inputs = inputs.to("cpu")
                except Exception as e:
                    if "meta tensor" in str(e).lower():
                        # Handle meta tensor case
                        if self.use_gpu:
                            inputs = inputs.to("cuda")
                        else:
                            inputs = inputs.to("cpu")
                    else:
                        raise e

                # Generate translation with error handling for meta tensors
                try:
                    generated_tokens = self.model.generate(
                        **inputs,
                        forced_bos_token_id=self.tokenizer.lang_code_to_id[target_lang_code],
                        max_length=1500,
                        num_beams=4,
                        early_stopping=True
                    )
                    return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                except Exception as e:
                    if "meta tensor" in str(e).lower():
                        # Fallback to CPU if GPU fails with meta tensor
                        print("Meta tensor issue detected, falling back to CPU for this chunk")
                        inputs = inputs.to("cpu")
                        if hasattr(self.model, 'to'):
                            self.model.to("cpu")
                        generated_tokens = self.model.generate(
                            **inputs,
                            forced_bos_token_id=self.tokenizer.lang_code_to_id[target_lang_code],
                            max_length=1500,
                            num_beams=4,
                            early_stopping=True
                        )
                        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                    else:
                        raise e

            elif "m2m" in self.model_name.lower():
                # M2M100 translation
                self.tokenizer.src_lang = "en"
                inputs = self.tokenizer(chunk, return_tensors="pt", truncation=True, max_length=1024)

                # Move inputs to the correct device
                try:
                    if self.use_gpu:
                        inputs = inputs.to("cuda")
                    else:
                        inputs = inputs.to("cpu")
                except Exception as e:
                    if "meta tensor" in str(e).lower():
                        # Handle meta tensor case
                        if self.use_gpu:
                            inputs = inputs.to("cuda")
                        else:
                            inputs = inputs.to("cpu")
                    else:
                        raise e

                # Generate translation with error handling for meta tensors
                try:
                    generated_tokens = self.model.generate(
                        **inputs,
                        forced_bos_token_id=self.tokenizer.get_lang_id(target_lang_code),
                        max_length=1500,
                        num_beams=4,
                        early_stopping=True
                    )
                    return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                except Exception as e:
                    if "meta tensor" in str(e).lower():
                        # Fallback to CPU if GPU fails with meta tensor
                        print("Meta tensor issue detected, falling back to CPU for this chunk")
                        inputs = inputs.to("cpu")
                        if hasattr(self.model, 'to'):
                            self.model.to("cpu")
                        generated_tokens = self.model.generate(
                            **inputs,
                            forced_bos_token_id=self.tokenizer.get_lang_id(target_lang_code),
                            max_length=1500,
                            num_beams=4,
                            early_stopping=True
                        )
                        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                    else:
                        raise e

            elif self.pipeline:
                # Generic pipeline translation
                result = self.pipeline(chunk, max_length=1500, num_beams=4)
                return result[0]['translation_text']

            else:
                raise ValueError("No valid translation model available")

        except Exception as e:
            print(f"Error translating chunk: {e}")
            return chunk  # Return original chunk on failure

    def _translate_chunk_wrapper(self, args: Tuple[str, str]) -> str:
        """Wrapper for parallel translation with thread safety"""
        chunk, target_language = args
        with self.translation_lock:
            return self._translate_chunk(chunk, target_language)

    def translate_text(self, text: str, target_language: str) -> str:
        """
        Translate text using the selected AI model with parallel processing

        Args:
            text: Text to translate
            target_language: Target language name

        Returns:
            Translated text
        """
        if not text or not text.strip():
            return ""

        try:
            # Split text into chunks
            chunks = self._split_text_into_chunks(text)
            print(f"Split text into {len(chunks)} chunks for translation")

            if len(chunks) == 1:
                # Single chunk - direct translation
                return self._translate_chunk(chunks[0], target_language)
            else:
                # Multiple chunks - parallel processing
                return self._translate_chunks_parallel(chunks, target_language)

        except Exception as e:
            print(f"Translation error: {e}")
            return text  # Return original text on failure

    def _translate_chunks_parallel(self, chunks: List[str], target_language: str) -> str:
        """Translate multiple chunks in parallel using ThreadPoolExecutor"""
        translated_chunks = []

        try:
            # Prepare arguments for parallel processing
            chunk_args = [(chunk, target_language) for chunk in chunks]

            # Use ThreadPoolExecutor for parallel translation
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                # Submit all translation tasks
                futures = [executor.submit(self._translate_chunk_wrapper, args) for args in chunk_args]

                # Collect results as they complete
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    try:
                        result = future.result()
                        translated_chunks.append(result)
                        print(f"Translated chunk {i+1}/{len(chunks)}")
                    except Exception as e:
                        print(f"Translation failed for chunk {i+1}: {e}")
                        translated_chunks.append(chunks[i])  # Keep original chunk

            # Reconstruct text in original order
            return ' '.join(translated_chunks)

        except Exception as e:
            print(f"Parallel translation error: {e}")
            # Fallback to sequential translation
            return ' '.join([self._translate_chunk(chunk, target_language) for chunk in chunks])

    def get_performance_info(self) -> Dict[str, str]:
        """Get performance and configuration information"""
        return {
            'model': self.model_name,
            'gpu_acceleration': "✅" if self.use_gpu else "❌",
            'max_threads': self.max_threads,
            'chunk_size': self.chunk_size,
            'device': "CUDA" if self.use_gpu else "CPU",
            'model_loaded': "✅" if (self.model or self.pipeline) else "❌"
        }

    def validate_model(self) -> bool:
        """Validate that the model is properly loaded"""
        return self.model is not None or self.pipeline is not None

# Example usage
if __name__ == "__main__":
    # Test the DeepTranslator
    translator = DeepTranslator(
        model_name="facebook/mbart-large-50-many-to-many-mmt",
        chunk_size=2000,
        max_threads=2,
        use_gpu=False
    )

    test_text = "Hello, this is a test of the AI translation system. " * 10
    result = translator.translate_text(test_text, "Spanish")
    print(f"Translation result: {result[:100]}...")

    # Test saving translated text
    saved_path = translator.save_translated_text(result, "test_book", "Spanish")
    print(f"Text saved to: {saved_path}")

    print("Performance info:", translator.get_performance_info())
