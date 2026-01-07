# 📚 Ebook Translator & Text-to-Speech App - Setup Guide

A simple Streamlit application that can ingest ebooks (PDF or EPUB), extract text, clean it, translate it using Google Translate, and generate audio using Google Text-to-Speech.

## ✨ Features

- 📄 **Text Extraction**: Extract text from PDF and EPUB files
- 🧹 **Text Cleaning**: Clean and format text for better translation
- 🌍 **Translation**: Translate text to 25+ languages using Google Translate
- 🔊 **Audio Generation**: Convert translated text to speech using Google TTS
- 💾 **File Organization**: Save outputs in timestamped folders
- 🎨 **User-Friendly Interface**: Simple and intuitive Streamlit GUI

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
streamlit run app.py
```

### 3. Open in Browser

The app will automatically open in your web browser at `http://localhost:8501`

## 📋 Requirements

The following Python packages are required (all listed in `requirements.txt`):

```
streamlit==1.29.0
PyPDF2==3.0.1
ebooklib==0.18
beautifulsoup4==4.12.2
googletrans==4.0.0rc1
gTTS==2.4.0
pydub==0.25.1
python-magic==0.4.27
nltk==3.8.1
```

## 🔧 System Dependencies

### For Linux Users:

You may need to install additional system dependencies:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3-magic libmagic1

# For audio processing (optional)
sudo apt-get install ffmpeg

# For PDF processing (if needed)
sudo apt-get install poppler-utils
```

### For macOS Users:

```bash
brew install libmagic
```

### For Windows Users:

Download and install `python-magic-bin` from:
https://pypi.org/project/python-magic-bin/

## 🌐 Supported Languages

The app supports translation to 25+ languages including:

- English, Spanish, French, German, Italian
- Portuguese, Russian, Chinese (Simplified/Traditional)
- Japanese, Korean, Arabic, Hindi
- Dutch, Swedish, Danish, Norwegian, Finnish
- Polish, Turkish, Greek, Hebrew, Thai, Vietnamese

## 📖 How to Use

1. **Upload an Ebook**: Click "Browse files" and select a PDF or EPUB file
2. **Select Language**: Choose your target language from the sidebar dropdown
3. **Process**: Click "Process Ebook" to start the translation
4. **Wait**: The app will:
   - Extract text from your ebook
   - Clean and format the text
   - Translate to your selected language
   - Generate audio from the translated text
5. **Download**: Download the translated text and audio files

## 📁 Output Structure

Processed files are saved in folders named: `{book_name}_{timestamp}/`

```
my_book_20240107_143022/
├── my_book_translated.txt    # Translated text
└── my_book_audio.mp3         # Generated audio
```

## ⚠️ Important Notes

### No API Keys Required!
This app uses free services:
- **Google Translate** (via googletrans library)
- **Google Text-to-Speech** (via gTTS library)

### Limitations:
- **Large files** may take several minutes to process
- **Translation quality** depends on Google Translate
- **Audio generation** may be slow for long texts
- **Internet connection** is required for translation and TTS
- **Rate limiting** may occur with very large texts

### File Size Recommendations:
- **PDF**: Up to 50MB for optimal performance
- **EPUB**: Up to 20MB for optimal performance
- **Text content**: Up to 100,000 characters per session

## 🔍 Troubleshooting

### Common Issues:

1. **"File not supported" error**
   - Ensure your file is a valid PDF or EPUB
   - Check file extension (.pdf, .epub)

2. **"Failed to extract text" error**
   - Try a different ebook file
   - Some PDFs may be scanned images (need OCR)

3. **Translation errors**
   - Check your internet connection
   - Try again after a few minutes (rate limiting)

4. **Audio generation fails**
   - Text may be too long - try smaller files
   - Check if the target language supports TTS

5. **python-magic errors**
   - Install system dependencies (see above)
   - For Windows, use python-magic-bin

### Performance Tips:
- Start with smaller files to test
- Close other browser tabs for better performance
- Ensure stable internet connection

## 🛠️ Development

### Project Structure:
```
Simple_POC/
├── app.py              # Main Streamlit application
├── extract.py          # Text extraction functionality
├── cleaning.py         # Text cleaning and preprocessing
├── translate.py        # Translation and TTS functionality
├── requirements.txt    # Python dependencies
└── SETUP.md           # This file
```

### Key Components:

- **EbookExtractor**: Handles PDF and EPUB text extraction
- **TextCleaner**: Cleans and formats text for translation
- **TextTranslatorAndTTS**: Manages translation and audio generation
- **Streamlit GUI**: User interface for file upload and processing

## 📄 License

This project is for educational purposes. Please respect the terms of service of Google Translate and Google TTS.

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📞 Support

If you encounter any issues:
1. Check the troubleshooting section above
2. Ensure all dependencies are properly installed
3. Verify your internet connection
4. Try with smaller files first
