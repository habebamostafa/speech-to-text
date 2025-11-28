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

st.title("🎤 Speech Recognition System")

# Load model
@st.cache_resource
def load_model():
    try:
        model_files = ['my_model.h5', 'my_model (1).h5', 'model.h5']
        model_path = None
        
        for file in model_files:
            if os.path.exists(file):
                model_path = file
                break
        
        if model_path is None:
            st.error("❌ Model file not found")
            return None
            
        model = keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# Load model
if 'model' not in st.session_state:
    with st.spinner("Loading model..."):
        model = load_model()
        if model:
            st.session_state.model = model
            st.success("✅ Model loaded successfully!")
        else:
            st.stop()

# Constants
frame_length = 256
frame_step = 160
fft_length = 384

# English vocabulary
characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=characters, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

# Audio processing functions
def process_audio_file(audio_path):
    try:
        audio = tf.io.read_file(audio_path)
        audio, sample_rate = tf.audio.decode_wav(audio)
        audio = tf.squeeze(audio, axis=-1)
        audio = tf.cast(audio, tf.float32)

        spectrogram = tf.signal.stft(
            audio,
            frame_length=frame_length,
            frame_step=frame_step,
            fft_length=fft_length
        )
        spectrogram = tf.abs(spectrogram)
        spectrogram = tf.math.pow(spectrogram, 0.5)

        means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
        stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
        spectrogram = (spectrogram - means) / (stddevs + 1e-10)

        return spectrogram
    
    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
        return None

def decode_prediction(pred):
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

# Try streamlit-microphone
try:
    from streamlit_microphone import streamlit_microphone
    
    tab1, tab2 = st.tabs(["🎤 Record Audio", "📁 Upload File"])
    
    with tab1:
        st.header("Record Audio with Microphone")
        
        st.info("""
        **Instructions:**
        1. Click the microphone button below
        2. Allow microphone access in your browser
        3. Speak clearly in English
        4. Click stop when finished
        5. Click 'Analyze Recording' to get transcription
        """)
        
        # Microphone recorder
        audio_bytes = streamlit_microphone(
            key="microphone",
            start_prompt="🎤 Start recording",
            stop_prompt="⏹️ Stop recording",
            just_once=False
        )
        
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            
            if st.button("🔍 Analyze Recording", use_container_width=True):
                with st.spinner("Processing audio..."):
                    try:
                        # Save audio bytes to temporary file
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
        
        # Show results
        if st.session_state.get('last_prediction'):
            st.success("**Predicted Text:**")
            st.code(st.session_state.last_prediction)
    
    with tab2:
        st.header("Upload Audio File")
        
        uploaded_audio = st.file_uploader("Choose WAV file", type=['wav'])
        
        if uploaded_audio:
            st.audio(uploaded_audio, format='audio/wav')
            
            if st.button("🔍 Analyze Uploaded File", use_container_width=True):
                with st.spinner("Processing file..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        tmp_file.write(uploaded_audio.getvalue())
                        audio_path = tmp_file.name
                    
                    prediction = predict_from_audio(audio_path)
                    
                    if prediction:
                        st.success("**Predicted Text:**")
                        st.code(prediction)
                    
                    try:
                        os.unlink(audio_path)
                    except:
                        pass

except ImportError:
    st.warning("streamlit-microphone not available. Using upload-only version.")
    
    st.header("📁 Upload Audio File")
    
    uploaded_audio = st.file_uploader("Choose WAV audio file", type=['wav'])
    
    if uploaded_audio:
        st.audio(uploaded_audio, format='audio/wav')
        
        if st.button("🔍 Transcribe Audio", use_container_width=True):
            with st.spinner("Processing audio..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(uploaded_audio.getvalue())
                    audio_path = tmp_file.name
                
                prediction = predict_from_audio(audio_path)
                
                if prediction:
                    st.success("✅ Transcription complete!")
                    st.code(prediction)
                
                try:
                    os.unlink(audio_path)
                except:
                    pass

# Model info
with st.expander("ℹ️ Model Information"):
    st.markdown("""
    **Model Specifications:**
    - **Architecture:** DeepSpeech 2
    - **Training Data:** LJSpeech dataset
    - **Language:** English only
    - **Vocabulary:** 31 English characters
    
    **Tips for Better Results:**
    - Speak clearly and at moderate pace
    - Use good quality microphone
    - Record in quiet environment
    - Use 16kHz sample rate for best results
    """)