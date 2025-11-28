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

# Audio conversion and processing functions
def convert_to_mono_16k(audio_path, output_path):
    """Convert any audio to mono 16kHz WAV format"""
    try:
        from pydub import AudioSegment
        import librosa
        import soundfile as sf
        
        # Try pydub first
        try:
            audio = AudioSegment.from_file(audio_path)
            audio = audio.set_channels(1)  # Convert to mono
            audio = audio.set_frame_rate(TARGET_SAMPLE_RATE)  # Convert to 16kHz
            audio = audio.set_sample_width(2)  # 16-bit
            audio.export(output_path, format="wav")
            return True
        except:
            # Fallback to librosa
            audio, sr = librosa.load(audio_path, sr=TARGET_SAMPLE_RATE, mono=True)
            sf.write(output_path, audio, TARGET_SAMPLE_RATE, subtype='PCM_16')
            return True
            
    except Exception as e:
        st.error(f"❌ Conversion error: {e}")
        return False

def resample_audio(audio, original_sr, target_sr):
    """Resample audio to target sample rate"""
    if original_sr == target_sr:
        return audio
    
    # Simple resampling for demonstration
    # In production, use proper resampling like tf.signal.resample
    ratio = original_sr / target_sr
    indices = tf.range(0, tf.shape(audio)[0], ratio, dtype=tf.int32)
    return tf.gather(audio, indices)

def process_audio_file(audio_path):
    """Process audio file with proper stereo-to-mono and resampling"""
    try:
        # Read file
        audio = tf.io.read_file(audio_path)
        audio_tensor, original_sr = tf.audio.decode_wav(audio)
        original_sr = original_sr.numpy()
        
        # Convert stereo to mono if needed
        if len(audio_tensor.shape) > 1 and audio_tensor.shape[1] == 2:
            # Average the two channels to create mono
            audio_mono = tf.reduce_mean(audio_tensor, axis=1)
        else:
            audio_mono = tf.squeeze(audio_tensor, axis=-1)
        
        # Resample to 16kHz if needed
        if original_sr != TARGET_SAMPLE_RATE:
            audio_mono = resample_audio(audio_mono, original_sr, TARGET_SAMPLE_RATE)
        
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

def get_audio_info(audio_path):
    """Get information about audio file"""
    try:
        audio = tf.io.read_file(audio_path)
        audio_tensor, sample_rate = tf.audio.decode_wav(audio)
        
        info = {
            'sample_rate': sample_rate.numpy(),
            'channels': audio_tensor.shape[1] if len(audio_tensor.shape) > 1 else 1,
            'duration': audio_tensor.shape[0] / sample_rate.numpy(),
            'shape': audio_tensor.shape
        }
        return info
    except Exception as e:
        st.error(f"❌ Error getting audio info: {e}")
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
                
                # Always convert to ensure mono 16kHz
                st.info("🔄 Converting to mono 16kHz WAV...")
                converted_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                converted_path = converted_temp_file.name
                converted_temp_file.close()
                
                conversion_success = convert_to_mono_16k(input_temp_path, converted_path)
                
                if conversion_success:
                    # Check audio properties before and after
                    original_info = get_audio_info(input_temp_path) if file_extension == 'wav' else None
                    converted_info = get_audio_info(converted_path)
                    
                    if original_info:
                        st.warning(f"📊 Original: {original_info['channels']}ch, {original_info['sample_rate']}Hz")
                    
                    if converted_info:
                        st.success(f"📊 Converted: {converted_info['channels']}ch, {converted_info['sample_rate']}Hz, {converted_info['duration']:.1f}s")
                    
                    # Process the converted WAV file
                    prediction = predict_from_audio(converted_path)
                    
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
                else:
                    st.error("❌ Audio conversion failed")
                
                # Clean up temporary files
                try:
                    if os.path.exists(input_temp_path):
                        os.unlink(input_temp_path)
                    if os.path.exists(converted_path):
                        os.unlink(converted_path)
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

# Instructions
with st.expander("🎯 Instructions for Best Results"):
    st.markdown("""
    **For accurate transcription:**
    
    ✅ **Good Quality Audio:**
    - Clear speech, no background noise
    - Single speaker
    - 3-10 seconds duration
    
    ✅ **Technical Requirements:**
    - Mono (1 channel) audio
    - 16kHz sample rate  
    - WAV format recommended
    
    ⚠️ **The system automatically:**
    - Converts stereo to mono
    - Resamples to 16kHz
    - Normalizes audio levels
    
    **If results are poor:**
    1. Record in a quiet environment
    2. Speak clearly and at normal pace
    3. Use a good microphone
    4. Avoid audio compression
    """)

# Footer
st.markdown("---")
st.markdown("🎤 Speech Recognition System - Auto-converts to Mono 16kHz")