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
        audio = audio.set_channels(1)  # Force mono
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
        
        # Load with mono conversion
        audio, sr = librosa.load(input_path, sr=16000, mono=True)
        
        # Save as WAV
        sf.write(output_path, audio, 16000, subtype='PCM_16')
        st.success("✅ Conversion successful with librosa!")
        return True
        
    except Exception as e:
        st.warning(f"⚠️ Librosa failed: {e}")
    
    return False

# Fixed audio processing functions
def process_audio_file(audio_path):
    """Process audio file - handle stereo to mono conversion"""
    try:
        # Read file
        audio = tf.io.read_file(audio_path)
        audio, sample_rate = tf.audio.decode_wav(audio)
        
        # Handle stereo to mono conversion
        if len(audio.shape) > 1 and audio.shape[1] == 2:
            # Convert stereo to mono by averaging channels
            audio = tf.reduce_mean(audio, axis=1)
        else:
            # If already mono, just squeeze
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

# Function to check audio file properties
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
                
                # If file is already WAV, use directly
                if file_extension == 'wav':
                    wav_path = input_temp_path
                    conversion_success = True
                    st.success("✅ WAV file - no conversion needed")
                    
                    # Check audio properties
                    audio_info = get_audio_info(wav_path)
                    if audio_info:
                        st.info(f"📊 Audio Info: {audio_info['channels']} channel(s), {audio_info['sample_rate']}Hz, {audio_info['duration']:.2f}s")
                        
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
                        # Check final audio properties
                        final_audio_info = get_audio_info(wav_path)
                        if final_audio_info:
                            st.info(f"📊 Final Audio: {final_audio_info['channels']} channel(s), {final_audio_info['sample_rate']}Hz")
                        
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
                    **Conversion failed! Try:**
                    1. Convert to WAV manually first
                    2. Use mono audio (1 channel)
                    3. Use 16kHz sample rate
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

# Quick fix for stereo files
with st.expander("🔧 Quick Fix for Stereo Files"):
    st.markdown("""
    **If you have stereo (2-channel) audio files:**
    
    1. **Use online converter:** https://online-audio-converter.com/
    2. **Select:** Mono, 16kHz, WAV format
    3. **Upload the converted file**
    
    **Or use this Python code to convert:**
    ```python
    from pydub import AudioSegment
    
    # Load stereo file
    audio = AudioSegment.from_file("your_file.wav")
    
    # Convert to mono
    audio = audio.set_channels(1)
    
    # Set to 16kHz
    audio = audio.set_frame_rate(16000)
    
    # Save
    audio.export("converted.wav", format="wav")
    ```
    """)

# Footer
st.markdown("---")
st.markdown("🎤 Speech Recognition System - Fixed for Stereo/Mono Issues")