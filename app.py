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
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #28a745;
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
TARGET_SAMPLE_RATE = 16000

# English vocabulary (same as training)
characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=characters, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

# Audio conversion using librosa only
def convert_to_mono_16k(audio_path, output_path):
    """Convert any audio to mono 16kHz WAV format using librosa"""
    try:
        import librosa
        import soundfile as sf
        
        # Load audio with librosa - automatically converts to mono and 16kHz
        audio, sr = librosa.load(audio_path, sr=TARGET_SAMPLE_RATE, mono=True)
        
        # Save as WAV file
        sf.write(output_path, audio, TARGET_SAMPLE_RATE, subtype='PCM_16')
        return True
        
    except ImportError:
        st.error("❌ Required libraries not installed. Please install: pip install librosa soundfile")
        return False
    except Exception as e:
        st.error(f"❌ Conversion error: {e}")
        return False

def process_audio_directly(audio_bytes, file_extension):
    """Process audio directly from bytes without saving temporary files"""
    try:
        import librosa
        import io
        
        # Load audio from bytes
        if file_extension == 'wav':
            # For WAV files, use tensorflow directly
            audio_tensor, original_sr = tf.audio.decode_wav(audio_bytes)
            
            # Convert stereo to mono if needed
            if len(audio_tensor.shape) > 1 and audio_tensor.shape[1] == 2:
                audio_mono = tf.reduce_mean(audio_tensor, axis=1)
            else:
                audio_mono = tf.squeeze(audio_tensor, axis=-1)
            
            # Resample if needed
            if original_sr != TARGET_SAMPLE_RATE:
                # Simple resampling
                ratio = original_sr / TARGET_SAMPLE_RATE
                indices = tf.range(0, tf.shape(audio_mono)[0], ratio, dtype=tf.int32)
                audio_mono = tf.gather(audio_mono, indices)
        else:
            # For other formats, use librosa
            audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=TARGET_SAMPLE_RATE, mono=True)
            audio_mono = tf.convert_to_tensor(audio, dtype=tf.float32)
        
        audio_mono = tf.cast(audio_mono, tf.float32)

        # Extract spectrogram
        spectrogram = tf.signal.stft(
            audio_mono,
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
        st.error(f"❌ Error processing audio: {e}")
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

def predict_from_audio_bytes(audio_bytes, file_extension):
    """Predict text from audio bytes"""
    try:
        spectrogram = process_audio_directly(audio_bytes, file_extension)
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
                # Get audio bytes
                audio_bytes = uploaded_audio.getvalue()
                
                # Process directly from bytes
                st.info("🔄 Processing audio...")
                
                prediction = predict_from_audio_bytes(audio_bytes, file_extension)
                
                if prediction and prediction.strip():
                    st.session_state.last_prediction = prediction
                    st.markdown('<div class="success-box">', unsafe_allow_html=True)
                    st.success("✅ Transcription complete!")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
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
                    st.error("❌ No speech detected or transcription failed")
                    
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

# Simple WAV-only fallback
with st.expander("🎯 Simple WAV Upload (Fallback)"):
    st.markdown("""
    **If you're having issues with other formats, use this simple WAV upload:**
    """)
    
    wav_upload = st.file_uploader("Upload WAV file only", type=['wav'], key="wav_upload")
    
    if wav_upload and st.session_state.get('model'):
        st.audio(wav_upload, format='audio/wav')
        
        if st.button("🔍 Transcribe WAV", key="transcribe_wav"):
            with st.spinner("Processing WAV file..."):
                try:
                    # Read WAV file
                    audio_bytes = wav_upload.getvalue()
                    audio_tensor, original_sr = tf.audio.decode_wav(audio_bytes)
                    
                    # Convert to mono if stereo
                    if len(audio_tensor.shape) > 1 and audio_tensor.shape[1] == 2:
                        audio_mono = tf.reduce_mean(audio_tensor, axis=1)
                        st.info("🔄 Converted stereo to mono")
                    else:
                        audio_mono = tf.squeeze(audio_tensor, axis=-1)
                    
                    audio_mono = tf.cast(audio_mono, tf.float32)
                    
                    # Create spectrogram
                    spectrogram = tf.signal.stft(
                        audio_mono,
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
                    
                    # Predict
                    spectrogram = tf.expand_dims(spectrogram, axis=0)
                    prediction = st.session_state.model(spectrogram, training=False)
                    text = decode_prediction(prediction)
                    
                    if text and text.strip():
                        st.success("✅ Transcription:")
                        st.code(text)
                    else:
                        st.error("❌ No text detected")
                        
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# Installation instructions
with st.expander("🔧 Installation Requirements"):
    st.markdown("""
    ### Required Packages:
    ```bash
    pip install streamlit tensorflow numpy librosa soundfile scipy
    ```
    
    ### For M4A/MP3 Support:
    The app uses `librosa` which should handle most audio formats.
    If you encounter issues, convert files to WAV first.
    """)

# Footer
st.markdown("---")
st.markdown("🎤 Speech Recognition System - Using Librosa for Audio Processing")