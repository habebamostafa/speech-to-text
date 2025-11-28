# -*- coding: utf-8 -*-
import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import tempfile
import os
import io
from scipy.io import wavfile

# Page settings
st.set_page_config(
    page_title="Speech Recognition System",
    page_icon="🎤",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-right: 5px solid #1f77b4;
        margin: 10px 0;
    }
    .upload-box {
        background-color: #e8f4fd;
        padding: 20px;
        border-radius: 10px;
        border: 2px dashed #1f77b4;
        margin: 20px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">🎤 Speech Recognition System</h1>', unsafe_allow_html=True)

# Load model automatically
@st.cache_resource
def load_model():
    """Load model from my_model.h5"""
    try:
        # Try different possible model names
        model_files = ['my_model.h5', 'my_model (1).h5', 'model.h5']
        model_path = None
        
        for file in model_files:
            if os.path.exists(file):
                model_path = file
                break
        
        if model_path is None:
            st.error("❌ Model file not found. Please make sure my_model.h5 exists in the current directory.")
            return None
            
        model = keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# Load model
if 'model' not in st.session_state:
    with st.spinner("🔄 Loading model..."):
        model = load_model()
        if model is not None:
            st.session_state.model = model
            st.success("✅ Model loaded successfully!")
        else:
            st.stop()

# Constants (same as training)
frame_length = 256
frame_step = 160
fft_length = 384

# English vocabulary (same as training)
characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=characters, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

# Multiple audio conversion methods
def convert_audio_to_wav(input_path, output_path, file_extension):
    """Convert any audio format to WAV using multiple methods"""
    
    # Method 1: Try pydub first (most reliable for M4A)
    try:
        from pydub import AudioSegment
        st.info("🔄 Trying pydub for conversion...")
        
        # Load audio file
        audio = AudioSegment.from_file(input_path, format=file_extension)
        
        # Convert to mono and set sample rate to 16000 Hz
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)
        
        # Set appropriate parameters
        audio = audio.set_sample_width(2)  # 16-bit
        
        # Export as WAV
        audio.export(output_path, format="wav", parameters=["-ac", "1", "-ar", "16000"])
        st.success("✅ Conversion successful with pydub!")
        return True
        
    except Exception as e:
        st.warning(f"⚠️ Pydub failed: {e}")
    
    # Method 2: Try librosa with different backends
    try:
        import librosa
        import soundfile as sf
        st.info("🔄 Trying librosa for conversion...")
        
        # Try different backends
        try:
            audio, sr = librosa.load(input_path, sr=16000, mono=True)
        except:
            # If default backend fails, try audioread
            audio, sr = librosa.load(input_path, sr=16000, mono=True, res_type='kaiser_fast')
        
        # Save as WAV
        sf.write(output_path, audio, 16000, subtype='PCM_16')
        st.success("✅ Conversion successful with librosa!")
        return True
        
    except Exception as e:
        st.warning(f"⚠️ Librosa failed: {e}")
    
    # Method 3: Try using subprocess with ffmpeg if available
    try:
        import subprocess
        st.info("🔄 Trying ffmpeg for conversion...")
        
        result = subprocess.run([
            'ffmpeg', '-i', input_path, '-ac', '1', '-ar', '16000', 
            '-acodec', 'pcm_s16le', output_path, '-y'
        ], capture_output=True, text=True)
        
        if result.returncode == 0 and os.path.exists(output_path):
            st.success("✅ Conversion successful with ffmpeg!")
            return True
        else:
            st.warning(f"⚠️ FFmpeg failed: {result.stderr}")
            
    except Exception as e:
        st.warning(f"⚠️ FFmpeg method failed: {e}")
    
    return False

# Audio processing functions
def process_audio_file(audio_path):
    """Process audio file - same as training function"""
    try:
        # Read file
        audio = tf.io.read_file(audio_path)
        audio, sample_rate = tf.audio.decode_wav(audio)
        audio = tf.squeeze(audio, axis=-1)
        audio = tf.cast(audio, tf.float32)

        # Extract spectrogram
        spectrogram = tf.signal.stft(
            audio,
            frame_length=frame_length,
            frame_step=frame_step,
            fft_length=fft_length
        )
        spectrogram = tf.abs(spectrogram)
        spectrogram = tf.math.pow(spectrogram, 0.5)

        # Normalize
        means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
        stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
        spectrogram = (spectrogram - means) / (stddevs + 1e-10)

        return spectrogram
    
    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
        return None

def decode_prediction(pred):
    """Decode prediction - same as training function"""
    try:
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
        
        output_text = []
        for result in results:
            chars = num_to_char(result)
            text = tf.strings.reduce_join(chars).numpy().decode("utf-8")
            output_text.append(text)
        
        return output_text[0] if output_text else ""
    
    except Exception as e:
        return ""

def predict_from_audio(audio_path):
    """Predict text from audio file"""
    try:
        spectrogram = process_audio_file(audio_path)
        if spectrogram is None:
            return None
        
        spectrogram = tf.expand_dims(spectrogram, axis=0)
        prediction = st.session_state.model(spectrogram, training=False)
        text = decode_prediction(prediction)
        
        return text
    
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        return None

# Main interface
st.markdown('<div class="upload-box">', unsafe_allow_html=True)
st.header("📁 Upload Audio File")

# Supported formats
supported_formats = ['wav', 'mp3', 'm4a', 'flac', 'ogg', 'aac', 'wma']

uploaded_audio = st.file_uploader(
    f"Choose audio file ({', '.join(supported_formats)})", 
    type=supported_formats
)

st.markdown('</div>', unsafe_allow_html=True)

if uploaded_audio is not None:
    # Display file info
    file_extension = uploaded_audio.name.split('.')[-1].lower()
    file_size = len(uploaded_audio.getvalue()) / 1024
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("File Size", f"{file_size:.1f} KB")
    with col2:
        st.metric("Format", file_extension.upper())
    with col3:
        st.metric("Status", "✅ Ready")
    
    # Play audio
    st.audio(uploaded_audio, format=f'audio/{file_extension}')
    
    if st.button("🔍 Transcribe Audio", use_container_width=True):
        with st.spinner("Processing audio file..."):
            try:
                # Save uploaded file temporarily
                input_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}')
                input_temp_file.write(uploaded_audio.getvalue())
                input_temp_path = input_temp_file.name
                input_temp_file.close()
                
                # If file is already WAV, use directly
                if file_extension == 'wav':
                    wav_path = input_temp_path
                    conversion_success = True
                    st.success("✅ WAV file - no conversion needed")
                else:
                    # Convert to WAV
                    st.info(f"🔄 Converting {file_extension.upper()} to WAV format...")
                    wav_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                    wav_path = wav_temp_file.name
                    wav_temp_file.close()
                    
                    conversion_success = convert_audio_to_wav(input_temp_path, wav_path, file_extension)
                
                if conversion_success:
                    # Verify the converted file
                    if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
                        # Process the WAV file
                        prediction = predict_from_audio(wav_path)
                        
                        if prediction:
                            st.session_state.last_prediction = prediction
                            st.success("✅ Transcription complete!")
                            
                            st.markdown('<div class="result-box">', unsafe_allow_html=True)
                            st.subheader("📝 Transcribed Text")
                            st.code(prediction, language='text')
                            
                            # Text statistics
                            text_length = len(prediction)
                            word_count = len(prediction.split())
                            
                            col_stat1, col_stat2 = st.columns(2)
                            with col_stat1:
                                st.metric("Characters", text_length)
                            with col_stat2:
                                st.metric("Words", word_count)
                                
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.error("❌ Failed to transcribe audio")
                    else:
                        st.error("❌ Converted file is empty or doesn't exist")
                else:
                    st.error("❌ All conversion methods failed")
                    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                    st.warning("""
                    **Conversion failed! Possible solutions:**
                    1. **Install FFmpeg** - Download from https://ffmpeg.org/
                    2. **Convert manually** to WAV format first
                    3. **Try a different audio format** like MP3
                    """)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Clean up temporary files
                try:
                    if os.path.exists(input_temp_path):
                        os.unlink(input_temp_path)
                    if file_extension != 'wav' and os.path.exists(wav_path):
                        os.unlink(wav_path)
                except Exception as e:
                    st.warning(f"⚠️ Could not clean up temp files: {e}")
                    
            except Exception as e:
                st.error(f"❌ Error processing file: {e}")

# Show previous results
if st.session_state.get('last_prediction'):
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.subheader("📝 Last Transcription")
    st.code(st.session_state.last_prediction, language='text')
    
    # Clear button
    if st.button("🗑️ Clear Result"):
        st.session_state.last_prediction = None
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# Installation instructions
with st.expander("🔧 Installation & Troubleshooting"):
    st.markdown("""
    ### Required Packages:
    ```bash
    pip install streamlit tensorflow numpy pydub librosa soundfile scipy
    ```
    
    ### For M4A Support (Required):
    **Option 1: Install FFmpeg (Recommended)**
    - **Windows:** Download from https://ffmpeg.org/download.html
    - **Mac:** `brew install ffmpeg`
    - **Linux:** `sudo apt install ffmpeg`
    
    **Option 2: Use Online Converter**
    Convert M4A to WAV first using: https://online-audio-converter.com/
    
    **Option 3: Manual Conversion**
    Use VLC media player or Audacity to convert to WAV format
    """)

# Footer
st.markdown("---")
st.markdown("🎤 Speech Recognition System - Supports M4A, MP3, WAV & more!")