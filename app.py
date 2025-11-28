# -*- coding: utf-8 -*-
import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import pickle
import sounddevice as sd
from scipy.io import wavfile
import wave
import os
import tempfile
import time
from jiwer import wer, cer

# إعدادات الصفحة
st.set_page_config(
    page_title="نظام التعرف على الكلام",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تخصيص التصميم
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
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
    .metric-box {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ff6b6b;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# العنوان الرئيسي
st.markdown('<h1 class="main-header">🎤 نظام التعرف على الكلام - STT</h1>', unsafe_allow_html=True)

# الشريط الجانبي
with st.sidebar:
    st.header("⚙️ الإعدادات")
    
    # تحميل النموذج
    st.subheader("تحميل النموذج")
    model_loaded = False
    
    if st.button("🔄 تحميل النموذج", use_container_width=True):
        with st.spinner("جاري تحميل النموذج..."):
            try:
                # تحميل النموذج والإعدادات
                model = keras.models.load_model('my_model (1).h5', compile=False)
                
                with open('improved_model_config.json', 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                with open('improved_preprocessors.pkl', 'rb') as f:
                    preprocessors = pickle.load(f)
                    char_to_num = preprocessors['char_to_num']
                    num_to_char = preprocessors['num_to_char']
                
                st.session_state.model = model
                st.session_state.config = config
                st.session_state.char_to_num = char_to_num
                st.session_state.num_to_char = num_to_char
                st.session_state.model_loaded = True
                
                st.success("✅ تم تحميل النموذج بنجاح!")
                
            except Exception as e:
                st.error(f"❌ خطأ في تحميل النموذج: {e}")
    
    st.divider()
    
    # إعدادات التسجيل
    st.subheader("إعدادات التسجيل")
    duration = st.slider("⏱️ مدة التسجيل (ثواني)", 1, 10, 5)
    sample_rate = st.selectbox("📊 معدل العينات", [16000, 22050, 44100], index=0)

# المحتوى الرئيسي
if not st.session_state.get('model_loaded', False):
    st.warning("⚠️ يرجى تحميل النموذج أولاً من الشريط الجانبي")
    st.stop()

# استخراج المتغيرات من session state
model = st.session_state.model
config = st.session_state.config
char_to_num = st.session_state.char_to_num
num_to_char = st.session_state.num_to_char

# إعدادات المعالجة
frame_length = config['frame_length']
frame_step = config['frame_step']
fft_length = config['fft_length']

# دوال المعالجة
def process_audio_file(audio_path):
    """معالجة ملف صوتي وتحويله لسبيكتروجرام"""
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
        
        return spectrogram, sample_rate.numpy()
    
    except Exception as e:
        st.error(f"❌ خطأ في معالجة الملف: {e}")
        return None, None

def decode_prediction(pred):
    """فك تشفير تنبؤ النموذج"""
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    
    # Greedy decoding
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    
    # تحويل لنص
    output_text = []
    for result in results:
        chars = num_to_char(result)
        text = tf.strings.reduce_join(chars).numpy().decode("utf-8")
        text = ' '.join(text.split()).strip()
        output_text.append(text)
    
    return output_text[0] if output_text else ""

def predict_from_audio(audio_path):
    """التنبؤ بالنص من ملف صوتي"""
    try:
        # معالجة الصوت
        spectrogram, sample_rate = process_audio_file(audio_path)
        if spectrogram is None:
            return None
        
        # إضافة بُعد الدفعة
        spectrogram = tf.expand_dims(spectrogram, axis=0)
        
        # التنبؤ
        prediction = model(spectrogram, training=False)
        text = decode_prediction(prediction)
        
        return text
    
    except Exception as e:
        st.error(f"❌ خطأ في التنبؤ: {e}")
        return None

# دالة تسجيل الصوت
def record_audio(duration=5, sample_rate=16000):
    """تسجيل الصوت من الميكروفون"""
    try:
        # إنشاء ملف مؤقت
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_filename = temp_file.name
        temp_file.close()
        
        # التسجيل
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='int16'
        )
        
        # شريط التقدم
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(duration):
            time.sleep(1)
            progress = (i + 1) / duration
            progress_bar.progress(progress)
            status_text.text(f"🎙️ جاري التسجيل... {i + 1}/{duration} ثانية")
        
        sd.wait()  # انتظار انتهاء التسجيل
        
        # حفظ الملف
        wavfile.write(temp_filename, sample_rate, recording)
        
        progress_bar.empty()
        status_text.empty()
        
        return temp_filename
    
    except Exception as e:
        st.error(f"❌ خطأ في التسجيل: {e}")
        return None

# تبويبات الواجهة
tab1, tab2, tab3 = st.tabs(["🎤 تسجيل صوتي", "📁 تحميل ملف", "📊 تقييم النموذج"])

with tab1:
    st.header("تسجيل صوتي مباشر")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("التسجيل")
        if st.button("🎙️ بدء التسجيل", use_container_width=True):
            audio_file = record_audio(duration=duration, sample_rate=sample_rate)
            if audio_file:
                st.session_state.recorded_audio = audio_file
                st.success("✅ تم التسجيل بنجاح!")
                
                # تشغيل الصوت المسجل
                st.audio(audio_file, format='audio/wav')
    
    with col2:
        st.subheader("النتيجة")
        if st.session_state.get('recorded_audio'):
            if st.button("🔍 تحليل التسجيل", use_container_width=True):
                with st.spinner("جاري التعرف على الكلام..."):
                    prediction = predict_from_audio(st.session_state.recorded_audio)
                    
                    if prediction:
                        st.markdown('<div class="result-box">', unsafe_allow_html=True)
                        st.subheader("📝 النص المتوقع:")
                        st.success(prediction)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # حفظ النتيجة
                        st.session_state.last_prediction = prediction

with tab2:
    st.header("تحميل ملف صوتي")
    
    uploaded_file = st.file_uploader("اختر ملف صوتي (WAV)", type=['wav'])
    
    if uploaded_file is not None:
        # حفظ الملف المؤقت
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            audio_path = tmp_file.name
        
        # عرض الملف
        st.audio(uploaded_file, format='audio/wav')
        
        if st.button("🔍 تحليل الملف", use_container_width=True):
            with st.spinner("جاري التعرف على الكلام..."):
                prediction = predict_from_audio(audio_path)
                
                if prediction:
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    st.subheader("📝 النص المتوقع:")
                    st.success(prediction)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # تنظيف الملف المؤقت
                    os.unlink(audio_path)

with tab3:
    st.header("تقييم أداء النموذج")
    
    st.subheader("اختبار مع نص مرجعي")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        reference_text = st.text_area("✍️ أدخل النص المرجعي:", 
                                    placeholder="أدخل النص الصحيح هنا...")
    
    with col2:
        predicted_text = st.text_area("🤖 النص المتوقع:", 
                                    value=st.session_state.get('last_prediction', ''),
                                    placeholder="سيظهر النص المتوقع هنا...")
    
    if st.button("📊 حساب مقاييس التقييم", use_container_width=True) and reference_text and predicted_text:
        try:
            # حساب المقاييس
            wer_score = wer(reference_text, predicted_text)
            cer_score = cer(reference_text, predicted_text)
            
            # عرض النتائج
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric("Word Error Rate (WER)", f"{wer_score:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col2:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric("Character Error Rate (CER)", f"{cer_score:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # تفسير النتائج
            st.subheader("📈 تفسير النتائج:")
            if wer_score < 0.1:
                st.success("🔹 **ممتاز**: النموذج يعمل بدقة عالية جداً")
            elif wer_score < 0.3:
                st.info("🔸 **جيد**: النموذج يعمل بدقة مقبولة")
            else:
                st.warning("🔺 **يحتاج تحسين**: دقة النموذج منخفضة")
                
        except Exception as e:
            st.error(f"❌ خطأ في حساب المقاييس: {e}")

# قسم المعلومات
with st.expander("ℹ️ معلومات عن النموذج"):
    st.markdown("""
    ### 📋 مواصفات النموذج:
    - **النموذج**: DeepSpeech 2 Architecture
    - **الإدخال**: ملفات صوتية WAV
    - **المخرج**: نصوص مكتوبة
    - **الدقة**: تختلف حسب جودة الصوت
    
    ### 💡 نصائح للاستخدام:
    - استخدم ميكروفون جيد
    - تكلم بوضوح وبطء معتدل
    - تجنب الضوضاء الخلفية
    - اختبر في بيئة هادئة
    """)

# التشغيل
if __name__ == "__main__":
    # تهيئة session state
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'recorded_audio' not in st.session_state:
        st.session_state.recorded_audio = None
    if 'last_prediction' not in st.session_state:
        st.session_state.last_prediction = ""