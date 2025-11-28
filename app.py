import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import tempfile
import os

st.set_page_config(page_title="Speech Recognition", page_icon="🎤")

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
        st.error(f"❌ Error: {e}")
        return None

if 'model' not in st.session_state:
    model = load_model()
    if model:
        st.session_state.model = model
        st.success("✅ Model loaded!")

# Upload section
st.header("📁 Upload Audio File")
uploaded_file = st.file_uploader("Upload WAV file", type=['wav'])

if uploaded_file and st.session_state.get('model'):
    st.audio(uploaded_file, format='audio/wav')
    
    if st.button("Transcribe Audio"):
        with st.spinner("Processing..."):
            # Save temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
                f.write(uploaded_file.getvalue())
                temp_path = f.name
            
            try:
                # Process audio (simplified)
                audio = tf.io.read_file(temp_path)
                audio, _ = tf.audio.decode_wav(audio)
                audio = tf.squeeze(audio, axis=-1)
                
                # Your existing processing code here...
                st.info("Audio processing would happen here")
                
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                try:
                    os.unlink(temp_path)
                except:
                    pass

st.info("💡 Upload a WAV file to transcribe speech to text")