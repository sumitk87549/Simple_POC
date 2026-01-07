# Enhanced Ebook Translator & TTS

High-performance ebook translation and text-to-speech application with GPU acceleration and parallel processing.

## Features

### 🚀 Performance Optimizations
- **Multi-threading Translation**: Parallel processing of text chunks for 3-5x speed improvement
- **GPU Acceleration**: CUDA support for translation (if GPU available)
- **Multiprocessing TTS**: Parallel audio generation for 2-4x faster processing
- **Async Operations**: Non-blocking I/O for better throughput
- **Smart Resource Management**: Automatic CPU/GPU resource allocation

### 📚 Core Functionality
- **Multi-format Support**: PDF and EPUB ebook processing
- **36 Languages**: Comprehensive translation support
- **Dual Audio Output**: Both standard and enhanced audio generation
- **Text Cleaning**: Intelligent text preprocessing
- **Batch Processing**: Handle large books efficiently

## Python Version Compatibility

Optimized for **Python 3.13** with automatic fallback mechanisms.

## Quick Installation

1. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run the Application:**
```bash
streamlit run app.py
```

## Performance Settings

Configure in the application sidebar:
- 🚀 **GPU Acceleration**: Enable/disable GPU processing
- 🔄 **Parallel Processing**: Enable/disable multi-threading
- 📊 **Performance Info**: View system capabilities

## Optional GPU Support

For maximum performance, install CUDA-enabled PyTorch:

**For CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Expected Performance Gains

With optimizations enabled:
- **Translation Speed**: 3-5x faster with parallel processing
- **TTS Generation**: 2-4x faster with multiprocessing
- **Large Books**: Significantly reduced processing time

## File Structure

```
Simple_POC/
├── app.py              # Main Streamlit application
├── translate.py         # Enhanced translation with GPU/parallel support
├── TTS.py              # Enhanced TTS with parallel processing
├── extract.py          # PDF/EPUB text extraction
├── cleaning.py         # Text cleaning and preprocessing
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Usage

1. Run `streamlit run app.py`
2. Upload a PDF or EPUB file
3. Select target language
4. Configure performance settings
5. Click "Process Ebook"
6. Download translated text and audio files

## Troubleshooting

### Python 3.13 Compatibility
Some advanced TTS engines (Coqui, XTTS) have compatibility issues with Python 3.13. The system automatically falls back to gTTS, which still benefits from parallel processing optimizations.

### If you encounter dependency issues:
```bash
# Update pip
pip install --upgrade pip

# Force reinstall problematic packages
pip install --force-reinstall --no-deps torch transformers
```

### Check GPU availability:
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## Dependencies

- **Core**: streamlit, PyPDF2, ebooklib, beautifulsoup4, deep-translator, gTTS
- **Performance**: torch, transformers, accelerate, datasets
- **Audio**: pydub, librosa, soundfile
- **Processing**: aiohttp, numpy, scipy, pillow
