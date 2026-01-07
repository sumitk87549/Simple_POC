import streamlit as st
import os
import time
from datetime import datetime
import torch
from extract import EbookExtractor
from cleaning import TextCleaner
from translate import TextTranslatorAndTTS
from TTS import EnhancedTTS

# Configure Streamlit page
st.set_page_config(
    page_title="Ebook Translator & TTS",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .error-message {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .info-message {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">📚 Ebook Translator & Text-to-Speech</h1>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    st.sidebar.title("⚙️ Configuration")
    
    # Language selection
    st.sidebar.subheader("🌍 Translation Settings")
    
    # Performance settings
    st.sidebar.subheader("⚡ Performance Settings")
    use_gpu = st.sidebar.checkbox("🚀 Use GPU Acceleration", value=True)
    use_parallel = st.sidebar.checkbox("🔄 Use Parallel Processing", value=True)
    
    # Display performance info
    if st.sidebar.button("📊 Show Performance Info"):
        trans_info = {
            'GPU Available': "✅" if torch.cuda.is_available() else "❌",
            'CPU Cores': os.cpu_count(),
            'Max Workers': min(32, (os.cpu_count() or 1) + 4)
        }
        
        tts_info = enhanced_tts.get_performance_info()
        
        st.sidebar.markdown("**Translation Performance:**")
        for key, value in trans_info.items():
            st.sidebar.write(f"- {key}: {value}")
        
        st.sidebar.markdown("**TTS Performance:**")
        for key, value in tts_info.items():
            st.sidebar.write(f"- {key}: {value}")
    
    # Initialize components with performance optimizations
    extractor = EbookExtractor()
    cleaner = TextCleaner()
    translator = TextTranslatorAndTTS(use_gpu=use_gpu, max_workers=None if use_parallel else 1)
    enhanced_tts = EnhancedTTS(use_gpu=use_gpu, preferred_engine="auto")
    
    # Get supported languages
    supported_languages = translator.get_supported_languages()
    
    # Language selection dropdown
    target_language = st.sidebar.selectbox(
        "Select Target Language:",
        options=supported_languages,
        index=0
    )
    
    # File upload section
    st.header("📁 Upload Ebook")
    
    uploaded_file = st.file_uploader(
        "Choose an ebook file (PDF or EPUB)",
        type=['pdf', 'epub'],
        help="Supported formats: PDF, EPUB"
    )
    
    if uploaded_file is not None:
        # Display file info
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024:.2f} KB",
            "File type": uploaded_file.type
        }
        
        st.write("### 📋 File Details:")
        for key, value in file_details.items():
            st.write(f"- **{key}:** {value}")
        
        # Check if format is supported
        if extractor.is_supported_format(uploaded_file.name):
            st.success("✅ File format is supported!")
            
            # Process button
            if st.button("🚀 Process Ebook", type="primary"):
                process_ebook(uploaded_file, extractor, cleaner, translator, enhanced_tts, target_language, use_gpu, use_parallel)
        else:
            st.error("❌ Unsupported file format. Please upload a PDF or EPUB file.")
    
    # Instructions section
    with st.expander("📖 How to Use"):
        st.markdown("""
        ### Steps to use this app:
        
        1. **Upload an Ebook**: Click "Browse files" and select a PDF or EPUB file
        2. **Select Language**: Choose your target language from the sidebar
        3. **Process**: Click "Process Ebook" to start the translation
        4. **Wait**: The app will extract text, clean it, translate it, and generate audio
        5. **Download**: Download the translated text and audio files
        
        ### What happens during processing:
        - 📄 **Text Extraction**: Text is extracted from your ebook
        - 🧹 **Text Cleaning**: The text is cleaned and formatted
        - 🌍 **Translation**: Text is translated to your selected language
        - 🔊 **Audio Generation**: Audio is created from the translated text
        - 💾 **File Saving**: Files are saved in a timestamped folder
        """)
    
    # System requirements section
    with st.expander("⚙️ System Requirements"):
        st.markdown("""
        ### Required Dependencies:
        - Python 3.7+
        - Internet connection for translation and TTS
        
        ### No API Keys Required!
        This app uses free services:
        - **Google Translate** (via googletrans library)
        - **Google Text-to-Speech** (via gTTS library)
        
        ### Note:
        - Large files may take several minutes to process
        - Translation quality depends on Google Translate
        - Audio generation may be slow for long texts
        """)

def process_ebook(uploaded_file, extractor, cleaner, translator, enhanced_tts, target_language, use_gpu, use_parallel):
    """Process the uploaded ebook file"""
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Save uploaded file temporarily
        status_text.text("📁 Saving uploaded file...")
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        progress_bar.progress(10)
        
        # Step 2: Extract text
        status_text.text("📄 Extracting text from ebook...")
        original_text = extractor.extract_text(temp_file_path)
        
        if not original_text:
            st.error("❌ Failed to extract text from the ebook. Please try another file.")
            return
        
        progress_bar.progress(30)
        
        # Step 3: Clean text
        status_text.text("🧹 Cleaning and formatting text...")
        cleaned_text = cleaner.clean_for_translation(original_text)
        
        progress_bar.progress(50)
        
        # Step 4: Create output folder
        book_name = os.path.splitext(uploaded_file.name)[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder = f"{book_name}_{timestamp}"
        
        progress_bar.progress(60)
        
        # Step 5: Translate and generate TTS
        status_text.text(f"🌍 Translating to {target_language}...")
        
        result = translator.process_book_translation(
            cleaned_text, 
            target_language, 
            book_name, 
            output_folder
        )
        
        progress_bar.progress(90)
        
        # Step 6: Generate enhanced TTS if translation succeeded
        if result['success'] and 'translated_text' in result:
            status_text.text("🔊 Generating enhanced audio...")
            audio_file_path = os.path.join(output_folder, f"{book_name}_enhanced_audio.mp3")
            
            # Use enhanced TTS for better quality
            tts_success = enhanced_tts.generate_tts(
                result['translated_text'],
                target_language,
                audio_file_path
            )
            
            if tts_success:
                result['enhanced_audio_file'] = audio_file_path
                print(f"Enhanced audio saved to: {audio_file_path}")
        
        # Step 6: Display results
        if result['success']:
            progress_bar.progress(100)
            status_text.text("✅ Processing completed successfully!")
            
            # Success message
            st.markdown(f"""
            <div class="success-message">
                <h3>🎉 Processing Completed Successfully!</h3>
                <p>Your ebook has been processed and files are saved in: <strong>{output_folder}/</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display file information
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📄 Translated Text")
                st.info(f"File: {result['translated_file']}")
                
                # Show preview of translated text
                if result['translated_text']:
                    with st.expander("Preview Translated Text"):
                        st.text_area("Translated Content:", result['translated_text'][:1000] + "..." if len(result['translated_text']) > 1000 else result['translated_text'], height=200)
            
            with col2:
                st.subheader("🔊 Audio Files")
                
                # Standard audio
                if 'audio_file' in result:
                    st.info(f"Standard Audio: {result['audio_file']}")
                    
                    # Provide audio player if file exists
                    if os.path.exists(result['audio_file']):
                        st.audio(result['audio_file'], format='audio/mp3')
                
                # Enhanced audio
                if 'enhanced_audio_file' in result:
                    st.success(f"Enhanced Audio: {result['enhanced_audio_file']}")
                    
                    # Provide audio player if file exists
                    if os.path.exists(result['enhanced_audio_file']):
                        st.audio(result['enhanced_audio_file'], format='audio/mp3')
                elif 'audio_file' not in result:
                    st.warning("No audio files were generated")
            
            # Download buttons
            st.subheader("💾 Download Files")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if os.path.exists(result['translated_file']):
                    with open(result['translated_file'], 'r', encoding='utf-8') as f:
                        st.download_button(
                            label="📄 Download Translated Text",
                            data=f.read(),
                            file_name=os.path.basename(result['translated_file']),
                            mime="text/plain"
                        )
            
            with col2:
                if 'audio_file' in result and os.path.exists(result['audio_file']):
                    with open(result['audio_file'], 'rb') as f:
                        st.download_button(
                            label="🔊 Download Standard Audio",
                            data=f.read(),
                            file_name=os.path.basename(result['audio_file']),
                            mime="audio/mpeg"
                        )
            
            with col3:
                if 'enhanced_audio_file' in result and os.path.exists(result['enhanced_audio_file']):
                    with open(result['enhanced_audio_file'], 'rb') as f:
                        st.download_button(
                            label="🚀 Download Enhanced Audio",
                            data=f.read(),
                            file_name=os.path.basename(result['enhanced_audio_file']),
                            mime="audio/mpeg"
                        )
            
        else:
            st.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
            
            # Show partial results if available
            if 'translated_file' in result and os.path.exists(result['translated_file']):
                st.warning("⚠️ Translation was completed but audio generation failed.")
                st.info(f"Translated text is available at: {result['translated_file']}")
        
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
    except Exception as e:
        st.error(f"❌ An error occurred during processing: {str(e)}")
        progress_bar.progress(0)
        status_text.text("❌ Processing failed")

if __name__ == "__main__":
    main()