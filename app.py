import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import tempfile
import os
import sounddevice as sd
from scipy.io import wavfile
import time

st.set_page_config(page_title="Simple Speech Recorder", page_icon="🎤")

st.title("🎤 Simple Speech Recorder")

# Load model (same as before)
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

# Simple recorder using sounddevice
def record_audio(duration=5, sample_rate=16000):
    """Record audio using sounddevice"""
    try:
        st.info(f"🎤 Recording for {duration} seconds...")
        recording = sd.rec(int(duration * sample_rate), 
                          samplerate=sample_rate, 
                          channels=1, 
                          dtype='float32')
        
        # Progress bar
        progress_bar = st.progress(0)
        for i in range(duration):
            time.sleep(1)
            progress_bar.progress((i + 1) / duration)
        
        sd.wait()
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        wavfile.write(temp_file.name, sample_rate, recording)
        
        st.success("✅ Recording complete!")
        return temp_file.name
        
    except Exception as e:
        st.error(f"❌ Recording error: {e}")
        return None

# Main interface
st.header("Record Audio")

duration = st.slider("Recording duration (seconds)", 3, 10, 5)

if st.button("🎤 Start Recording", type="primary"):
    audio_file = record_audio(duration=duration)
    if audio_file:
        st.session_state.recorded_audio = audio_file
        st.audio(audio_file, format='audio/wav')

if st.session_state.get('recorded_audio'):
    if st.button("🔍 Transcribe"):
        # Add your transcription code here
        st.info("Transcription would happen here")