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
    .recording-box {
        background-color: #fff3cd;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">🎤 Speech Recognition System</h1>', unsafe_allow_html=True)

# Debug information
import os
st.write("Current directory:", os.getcwd())
st.write("Files in directory:", os.listdir('.'))

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
            
        st.write(f"📁 Loading model from: {model_path}")
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

# Convert audio data to WAV format
def convert_to_wav(audio_data, sample_rate=16000):
    """Convert audio data to WAV format"""
    try:
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_filename = temp_file.name
        temp_file.close()
        
        # Save as WAV file
        wavfile.write(temp_filename, sample_rate, audio_data)
        return temp_filename
    except Exception as e:
        st.error(f"❌ Error converting audio: {e}")
        return None

# Tabs interface
tab1, tab2 = st.tabs(["🎤 Record Audio", "📁 Upload Audio File"])

with tab1:
    st.header("Record Audio with Microphone")
    
    # Streamlit audio recorder
    st.markdown('<div class="recording-box">', unsafe_allow_html=True)
    st.subheader("🎙️ Record Your Voice")
    
    # Audio recorder using streamlit-audiorecorder (alternative approach)
    st.info("""
    **Instructions:**
    1. Click the record button below
    2. Allow microphone access in your browser
    3. Speak clearly in English
    4. Click stop when finished
    5. Click 'Analyze Recording' to get the transcription
    """)
    
    # Using streamlit's built-in audio recorder (if available)
    # Alternative: Use audio_recorder component
    try:
        from audio_recorder_streamlit import audio_recorder
        
        audio_bytes = audio_recorder(
            text="Click to record",
            recording_color="#e8b62c",
            neutral_color="#6aa36f",
            icon_name="microphone",
            icon_size="2x",
        )
        
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            
            if st.button("🔍 Analyze Recording", key="analyze_record", use_container_width=True):
                with st.spinner("Processing audio..."):
                    # Convert bytes to numpy array
                    try:
                        import io
                        from scipy.io import wavfile
                        
                        # Create temporary file from bytes
                        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                        temp_file.write(audio_bytes)
                        temp_filename = temp_file.name
                        temp_file.close()
                        
                        prediction = predict_from_audio(temp_filename)
                        
                        if prediction:
                            st.session_state.last_prediction = prediction
                            st.success("✅ Analysis complete!")
                        
                        # Clean up
                        try:
                            os.unlink(temp_filename)
                        except:
                            pass
                            
                    except Exception as e:
                        st.error(f"❌ Error processing recording: {e}")
                        
    except ImportError:
        st.warning("""
        **Audio recorder not available. Alternative method:**
        
        Please use the **Upload Audio File** tab to upload pre-recorded audio files.
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Show results
    if st.session_state.get('last_prediction'):
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.success("**Predicted Text:**")
        st.code(st.session_state.last_prediction)
        
        # Text statistics
        text_length = len(st.session_state.last_prediction)
        word_count = len(st.session_state.last_prediction.split())
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Characters", text_length)
        with col2:
            st.metric("Words", word_count)
            
        st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.header("Upload Audio File")
    
    uploaded_audio = st.file_uploader("Choose WAV audio file", type=['wav'])
    
    if uploaded_audio is not None:
        # Display file info
        file_size = len(uploaded_audio.getvalue()) / 1024
        st.write(f"**File Info:** Size: {file_size:.1f} KB | Format: WAV")
        
        # Play audio
        st.audio(uploaded_audio, format='audio/wav')
        
        if st.button("🔍 Analyze Audio File", use_container_width=True):
            with st.spinner("Processing audio file..."):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(uploaded_audio.getvalue())
                    audio_path = tmp_file.name
                
                prediction = predict_from_audio(audio_path)
                
                if prediction:
                    st.session_state.last_prediction = prediction
                    st.success("✅ Analysis complete!")
                    
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    st.success("**Predicted Text:**")
                    st.code(prediction)
                    
                    # Text statistics
                    text_length = len(prediction)
                    word_count = len(prediction.split())
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Characters", text_length)
                    with col2:
                        st.metric("Words", word_count)
                        
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Clean up
                try:
                    os.unlink(audio_path)
                except:
                    pass

# Model information
with st.expander("ℹ️ Model Information"):
    st.markdown("""
    **Model Specifications:**
    - **Architecture:** DeepSpeech 2
    - **Training Data:** LJSpeech dataset
    - **Language:** English only
    - **Vocabulary:** 31 English characters
    - **Processing Settings:**
      - Frame Length: 256
      - Frame Step: 160  
      - FFT Length: 384
    
    **Tips for Better Results:**
    - Speak clearly and at moderate pace
    - Use good quality microphone
    - Record in quiet environment
    - Use 16kHz sample rate for best results
    """)

# Footer
st.markdown("---")
st.markdown("🎤 Speech Recognition System - Powered by TensorFlow & Streamlit")