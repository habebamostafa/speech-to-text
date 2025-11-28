# -*- coding: utf-8 -*-
import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import tempfile
import os
import time

# محاولة استيراد مكتبات الصوت
try:
    import pyaudio
    import wave
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

# إعدادات الصفحة
st.set_page_config(
    page_title="نظام التعرف على الكلام",
    page_icon="🎤",
    layout="wide"
)

# العنوان الرئيسي
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
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🎤 نظام التعرف على الكلام</h1>', unsafe_allow_html=True)
import os
st.write("Current directory:", os.getcwd())
st.write("Files in directory:", os.listdir())

# تحميل النموذج تلقائياً
@st.cache_resource
def load_model():
    """تحميل النموذج من my_model (1).h5"""
    try:
        model = keras.models.load_model('my_model.h5', compile=False)
        return model
    except Exception as e:
        st.error(f"❌ خطأ في تحميل النموذج: {e}")
        return None

# تحميل النموذج
if 'model' not in st.session_state:
    with st.spinner("🔄 جاري تحميل النموذج..."):
        model = load_model()
        if model is not None:
            st.session_state.model = model
            st.success("✅ تم تحميل النموذج بنجاح!")
        else:
            st.stop()

# إعدادات ثابتة (نفس إعدادات التدريب)
frame_length = 256
frame_step = 160
fft_length = 384

# مفردات إنجليزية (نفس التدريب)
characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=characters, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

# دوال المعالجة
def process_audio_file(audio_path):
    """معالجة ملف صوتي - نفس دالة التدريب"""
    try:
        # قراءة الملف
        audio = tf.io.read_file(audio_path)
        audio, sample_rate = tf.audio.decode_wav(audio)
        audio = tf.squeeze(audio, axis=-1)
        audio = tf.cast(audio, tf.float32)

        # استخراج السبيكتروجرام
        spectrogram = tf.signal.stft(
            audio,
            frame_length=frame_length,
            frame_step=frame_step,
            fft_length=fft_length
        )
        spectrogram = tf.abs(spectrogram)
        spectrogram = tf.math.pow(spectrogram, 0.5)

        # تطبيع
        means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
        stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
        spectrogram = (spectrogram - means) / (stddevs + 1e-10)

        return spectrogram
    
    except Exception as e:
        st.error(f"❌ خطأ في معالجة الملف: {e}")
        return None

def decode_prediction(pred):
    """فك تشفير التنبؤ - نفس دالة التدريب"""
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
    """التنبؤ بالنص من ملف صوتي"""
    try:
        spectrogram = process_audio_file(audio_path)
        if spectrogram is None:
            return None
        
        spectrogram = tf.expand_dims(spectrogram, axis=0)
        prediction = st.session_state.model(spectrogram, training=False)
        text = decode_prediction(prediction)
        
        return text
    
    except Exception as e:
        st.error(f"❌ خطأ في التنبؤ: {e}")
        return None

# دالة التسجيل
def record_audio(duration=5, sample_rate=16000):
    """تسجيل صوت من الميكروفون"""
    try:
        chunk = 1024
        format = pyaudio.paInt16
        
        p = pyaudio.PyAudio()
        stream = p.open(
            format=format,
            channels=1,
            rate=sample_rate,
            input=True,
            frames_per_buffer=chunk
        )
        
        st.info("🎙️ جاري التسجيل... تكلم الآن!")
        frames = []
        
        total_chunks = int((sample_rate / chunk) * duration)
        progress_bar = st.progress(0)
        
        for i in range(total_chunks):
            data = stream.read(chunk)
            frames.append(data)
            progress_bar.progress((i + 1) / total_chunks)
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        progress_bar.empty()
        
        # حفظ الملف
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_filename = temp_file.name
        temp_file.close()
        
        wf = wave.open(temp_filename, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(format))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        st.success("✅ تم التسجيل بنجاح!")
        return temp_filename
        
    except Exception as e:
        st.error(f"❌ خطأ في التسجيل: {e}")
        return None

# تبويبات الواجهة
tab1, tab2 = st.tabs(["🎤 تسجيل من المايك", "📁 تحميل ملف صوتي"])

with tab1:
    st.header("التسجيل المباشر من الميكروفون")
    
    if not PYAUDIO_AVAILABLE:
        st.error("""
        ❌ **PyAudio غير مثبت**
        
        للتسجيل من المايكروفون:
        ```bash
        pip install pyaudio
        ```
        """)
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            duration = st.slider("مدة التسجيل (ثواني)", 1, 10, 3)
            
            if st.button("⏺️ بدء التسجيل", use_container_width=True):
                audio_file = record_audio(duration=duration)
                if audio_file:
                    st.session_state.recorded_audio = audio_file
                    st.rerun()
            
            if st.session_state.get('recorded_audio'):
                st.audio(st.session_state.recorded_audio, format='audio/wav')
                
                if st.button("🔍 تحليل التسجيل", use_container_width=True):
                    with st.spinner("جاري التعرف على الكلام..."):
                        prediction = predict_from_audio(st.session_state.recorded_audio)
                        if prediction:
                            st.session_state.last_prediction = prediction
                            st.rerun()
        
        with col2:
            if st.session_state.get('last_prediction'):
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.success("**النص المتوقع:**")
                st.code(st.session_state.last_prediction)
                st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.header("تحميل ملف صوتي")
    
    uploaded_audio = st.file_uploader("اختر ملف صوتي WAV", type=['wav'])
    
    if uploaded_audio is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_audio.getvalue())
            audio_path = tmp_file.name
        
        st.audio(uploaded_audio, format='audio/wav')
        
        if st.button("🔍 تحليل الملف", use_container_width=True):
            with st.spinner("جاري التعرف على الكلام..."):
                prediction = predict_from_audio(audio_path)
                
                if prediction:
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    st.success("**النص المتوقع:**")
                    st.code(prediction)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                try:
                    os.unlink(audio_path)
                except:
                    pass

# معلومات عن النموذج
with st.expander("ℹ️ معلومات عن النموذج"):
    st.markdown("""
    **مواصفات النموذج:**
    - مدرب على اللغة الإنجليزية
    - مجموعة بيانات: LJSpeech
    - مفردات: 31 حرف إنجليزي
    - إعدادات المعالجة:
      - Frame Length: 256
      - Frame Step: 160  
      - FFT Length: 384
    """)

# تذييل الصفحة
st.markdown("---")
st.markdown("🎤 نظام التعرف على الكلام - يعمل بـ my_model (1).h5")