import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import tempfile
import os
import av
from typing import List, Tuple

# Page configuration
st.set_page_config(
    page_title="Live Speech to Text",
    page_icon="🎤",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5rem;
        margin-bottom: 2rem;
    }
    .recording-box {
        background-color: #fff3cd;
        padding: 25px;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 20px 0;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 25px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 20px 0;
    }
    .status-recording {
        color: #dc3545;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .status-ready {
        color: #28a745;
        font-weight: bold;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-title">🎤 Live Speech to Text</h1>', unsafe_allow_html=True)

# Load model function
@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        possible_files = ['my_model.h5', 'my_model (1).h5', 'model.h5']
        model_path = None
        
        for file in possible_files:
            if os.path.exists(file):
                model_path = file
                st.info(f"📁 Found model file: {file}")
                break
        
        if model_path is None:
            st.error("❌ Model file not found! Please make sure my_model.h5 exists.")
            return None
            
        model = keras.models.load_model(model_path, compile=False)
        return model
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# Load model
if 'model' not in st.session_state:
    with st.spinner("🔄 Loading speech recognition model..."):
        model = load_model()
        if model is not None:
            st.session_state.model = model
            st.success("✅ Model loaded successfully!")
        else:
            st.stop()

# Model configuration
frame_length = 256
frame_step = 160
fft_length = 384

# English character vocabulary
characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=characters, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

# Audio processing functions
def process_audio_array(audio_array: np.ndarray, sample_rate: int = 16000) -> tf.Tensor:
    """Process audio array for prediction"""
    try:
        # Convert to tensor
        audio = tf.convert_to_tensor(audio_array, dtype=tf.float32)
        
        # Ensure correct shape
        if len(audio.shape) > 1:
            audio = tf.squeeze(audio, axis=-1)
        
        # Resample if necessary (assuming model expects 16kHz)
        if sample_rate != 16000:
            # Simple resampling by slicing (for demo - in production use proper resampling)
            ratio = sample_rate / 16000
            audio = audio[::int(ratio)]
        
        # Create spectrogram
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
        st.error(f"❌ Error processing audio: {str(e)}")
        return None

def decode_prediction(pred):
    """Convert model prediction to text"""
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

def predict_from_audio_array(audio_array: np.ndarray, sample_rate: int = 16000):
    """Predict text from audio array"""
    try:
        spectrogram = process_audio_array(audio_array, sample_rate)
        if spectrogram is None:
            return None
        
        spectrogram = tf.expand_dims(spectrogram, axis=0)
        prediction = st.session_state.model(spectrogram, training=False)
        text = decode_prediction(prediction)
        
        return text
        
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        return None

# Try to import streamlit-webrtc for live recording
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
    
    class AudioRecorder(AudioProcessorBase):
        def __init__(self):
            self.audio_frames = []
            self.is_recording = False
            
        def recv(self, frame):
            if self.is_recording:
                # Convert audio frame to numpy array
                audio = frame.to_ndarray()
                self.audio_frames.append(audio)
            return frame
        
        def start_recording(self):
            self.audio_frames = []
            self.is_recording = True
            
        def stop_recording(self):
            self.is_recording = False
            if self.audio_frames:
                # Combine all frames
                audio_data = np.concatenate(self.audio_frames, axis=0)
                return audio_data
            return None
    
    # Initialize audio recorder
    if 'audio_recorder' not in st.session_state:
        st.session_state.audio_recorder = AudioRecorder()
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="recording-box">', unsafe_allow_html=True)
        st.header("🎤 Live Recording")
        
        st.info("""
        **Instructions:**
        1. Click **Start Recording** below
        2. Allow microphone access in your browser
        3. Speak clearly in English
        4. Click **Stop Recording** when finished
        5. Click **Transcribe** to get the text
        """)
        
        # Recording controls
        col_controls1, col_controls2 = st.columns(2)
        
        with col_controls1:
            if st.button("🎤 Start Recording", use_container_width=True, type="primary"):
                st.session_state.audio_recorder.start_recording()
                st.session_state.recording_status = "recording"
                st.rerun()
        
        with col_controls2:
            if st.button("⏹️ Stop Recording", use_container_width=True):
                st.session_state.audio_recorder.stop_recording()
                st.session_state.recording_status = "stopped"
                st.rerun()
        
        # Show recording status
        if st.session_state.get('recording_status') == "recording":
            st.markdown('<p class="status-recording">● Recording... Speak now!</p>', unsafe_allow_html=True)
        elif st.session_state.get('recording_status') == "stopped":
            st.markdown('<p class="status-ready">✅ Recording stopped. Ready to transcribe.</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-ready">🎤 Ready to record</p>', unsafe_allow_html=True)
        
        # Transcribe button
        if st.session_state.get('recording_status') == "stopped":
            if st.button("🔍 Transcribe Recording", use_container_width=True):
                with st.spinner("🔄 Processing audio..."):
                    audio_data = st.session_state.audio_recorder.audio_frames
                    if audio_data:
                        # Combine frames and process
                        combined_audio = np.concatenate(audio_data, axis=0)
                        # Take only one channel if stereo
                        if len(combined_audio.shape) > 1:
                            combined_audio = combined_audio[:, 0]
                        
                        prediction = predict_from_audio_array(combined_audio, sample_rate=48000)
                        
                        if prediction:
                            st.session_state.last_prediction = prediction
                            st.success("✅ Transcription complete!")
                        else:
                            st.error("❌ Failed to transcribe audio")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.header("📝 Transcription Result")
        
        if st.session_state.get('last_prediction'):
            st.success("**Transcribed Text:**")
            st.code(st.session_state.last_prediction, language='text')
            
            # Text statistics
            col_stat1, col_stat2 = st.columns(2)
            with col_stat1:
                st.metric("Characters", len(st.session_state.last_prediction))
            with col_stat2:
                st.metric("Words", len(st.session_state.last_prediction.split()))
            
            # Clear button
            if st.button("🗑️ Clear Result", use_container_width=True):
                st.session_state.last_prediction = None
                st.rerun()
        else:
            st.info("👆 Record some audio and click 'Transcribe' to see results here")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # WebRTC streamer for audio input
    webrtc_ctx = webrtc_streamer(
        key="audio-recorder",
        mode=WebRtcMode.SENDRECV,
        audio_receiver_size=1024,
        media_stream_constraints={"audio": True, "video": False},
        audio_processor_factory=AudioRecorder,
    )
    
    if webrtc_ctx.audio_receiver:
        # This keeps the connection alive
        pass

except ImportError:
    st.error("""
    ❌ **streamlit-webrtc not available**
    
    To enable live recording, please install:
    ```bash
    pip install streamlit-webrtc aiortc
    ```
    
    **Alternative:** Use the file upload method below.
    """)
    
    # Fallback to file upload
    st.header("📁 Upload Audio File (Fallback)")
    
    uploaded_file = st.file_uploader("Choose a WAV file", type=['wav'])
    
    if uploaded_file:
        st.audio(uploaded_file, format='audio/wav')
        
        if st.button("🔍 Transcribe Audio File", use_container_width=True):
            with st.spinner("Processing audio file..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    audio_path = tmp_file.name
                
                # Use the existing file-based prediction
                def predict_from_file(audio_path):
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
                        
                        spectrogram = tf.expand_dims(spectrogram, axis=0)
                        prediction = st.session_state.model(spectrogram, training=False)
                        text = decode_prediction(prediction)
                        
                        return text
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
                        return None
                
                prediction = predict_from_file(audio_path)
                
                if prediction:
                    st.success("**Transcribed Text:**")
                    st.code(prediction)
                
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
    - **Sample Rate:** 16kHz
    
    **Tips for Better Results:**
    - Speak clearly and at moderate pace
    - Use good quality microphone
    - Record in quiet environment
    - Keep recordings under 10 seconds for best accuracy
    """)

# Footer
st.markdown("---")
st.markdown("🎤 Live Speech to Text - Powered by TensorFlow & Streamlit")