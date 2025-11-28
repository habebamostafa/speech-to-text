import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import tempfile
import os
import base64

st.set_page_config(page_title="Speech Recognition", page_icon="🎤", layout="wide")

st.title("🎤 Speech Recognition System")

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

# Audio processing functions (same as before)
frame_length = 256
frame_step = 160
fft_length = 384

characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=characters, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

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
        st.error(f"❌ Error: {e}")
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
        st.error(f"❌ Error: {e}")
        return None

# HTML Audio Recorder
st.header("🎤 Record Audio")

recorder_html = """
<script>
function startRecording() {
    navigator.mediaDevices.getUserMedia({ audio: true })
    .then(function(stream) {
        alert("Microphone access granted! This is a demo - please use the upload tab for actual transcription.");
    })
    .catch(function(err) {
        alert("Error accessing microphone: " + err);
    });
}
</script>

<button onclick="startRecording()" style="
    padding: 15px 30px;
    background-color: #4CAF50;
    color: white;
    border: none;
    border-radius: 8px;
    font-size: 16px;
    cursor: pointer;
    margin: 10px 0;
">
🎤 Start Recording (Demo)
</button>

<p><em>Note: For full recording functionality, please use the upload tab below.</em></p>
"""

st.components.v1.html(recorder_html, height=150)

st.header("📁 Upload Audio File")
uploaded_file = st.file_uploader("Upload WAV file for transcription", type=['wav'])

if uploaded_file and st.session_state.get('model'):
    st.audio(uploaded_file, format='audio/wav')
    
    if st.button("🔍 Transcribe Audio", use_container_width=True):
        with st.spinner("Processing audio..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
                f.write(uploaded_file.getvalue())
                temp_path = f.name
            
            prediction = predict_from_audio(temp_path)
            
            if prediction:
                st.success("✅ Transcription complete!")
                st.subheader("Predicted Text:")
                st.code(prediction)
                
                # Text statistics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Characters", len(prediction))
                with col2:
                    st.metric("Words", len(prediction.split()))
            else:
                st.error("❌ Failed to transcribe audio")
            
            try:
                os.unlink(temp_path)
            except:
                pass

st.info("💡 Upload a WAV file to transcribe speech to text")