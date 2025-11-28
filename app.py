# -*- coding: utf-8 -*-
import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import pickle
import tempfile
import os
import time
from jiwer import wer, cer
import io
import base64

# محاولة استيراد مكتبات الصوت
try:
    import sounddevice as sd
    from scipy.io import wavfile
    AUDIO_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ مكتبات الصوت غير مثبتة: {e}")
    AUDIO_AVAILABLE = False

# إعدادات الصفحة
st.set_page_config(
    page_title="نظام التعرف على الكلام - النموذج الحقيقي",
    page_icon="🎤",
    layout="wide"
)

# تخصيص التصميم
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
    .success-box {
        background-color: #d1ecf1;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #0c5460;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# العنوان الرئيسي
st.markdown('<h1 class="main-header">🎤 نظام التعرف على الكلام - النموذج الحقيقي</h1>', unsafe_allow_html=True)

# تهيئة حالة الجلسة
if 'model' not in st.session_state:
    st.session_state.model = None
if 'config' not in st.session_state:
    st.session_state.config = None
if 'char_to_num' not in st.session_state:
    st.session_state.char_to_num = None
if 'num_to_char' not in st.session_state:
    st.session_state.num_to_char = None
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = ""
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False

# الشريط الجانبي
with st.sidebar:
    st.header("⚙️ تحميل النموذج")
    
    # تحميل النموذج
    uploaded_model = st.file_uploader("رفع النموذج (.h5 أو .keras)", type=['h5', 'keras'])
    uploaded_config = st.file_uploader("رفع ملف الإعدادات (.json)", type=['json'])
    uploaded_preprocessors = st.file_uploader("رفع ملف المعالجات (.pkl)", type=['pkl'])
    
    if st.button("🔄 تحميل النموذج", use_container_width=True):
        if uploaded_model and uploaded_config and uploaded_preprocessors:
            with st.spinner("جاري تحميل النموذج..."):
                try:
                    # حفظ الملفات المؤقتة
                    model_path = tempfile.NamedTemporaryFile(delete=False, suffix='.h5').name
                    with open(model_path, 'wb') as f:
                        f.write(uploaded_model.getvalue())
                    
                    # تحميل النموذج
                    model = keras.models.load_model(model_path, compile=False)
                    st.session_state.model = model
                    
                    # تحميل الإعدادات
                    config = json.load(uploaded_config)
                    st.session_state.config = config
                    
                    # تحميل المعالجات
                    preprocessors = pickle.load(uploaded_preprocessors)
                    st.session_state.char_to_num = preprocessors.get('char_to_num')
                    st.session_state.num_to_char = preprocessors.get('num_to_char')
                    
                    st.success("✅ تم تحميل النموذج بنجاح!")
                    
                    # تنظيف الملف المؤقت
                    os.unlink(model_path)
                    
                except Exception as e:
                    st.error(f"❌ خطأ في تحميل النموذج: {e}")
        else:
            st.error("⚠️ يرجى رفع جميع الملفات المطلوبة")
    
    st.divider()
    
    # إعدادات التسجيل
    if AUDIO_AVAILABLE:
        st.header("🎙️ إعدادات التسجيل")
        st.session_state.duration = st.slider("مدة التسجيل (ثواني)", 1, 15, 5)
        st.session_state.sample_rate = st.selectbox("معدل العينات", [16000, 22050, 44100], index=0)
    else:
        st.error("❌ مكتبات الصوت غير مثبتة")

# دوال المعالجة الصوتية
def process_audio_file(audio_path):
    """معالجة ملف صوتي وتحويله لسبيكتروجرام"""
    try:
        # قراءة الملف
        audio = tf.io.read_file(audio_path)
        audio, sample_rate = tf.audio.decode_wav(audio)
        audio = tf.squeeze(audio, axis=-1)
        audio = tf.cast(audio, tf.float32)
        
        # استخدام الإعدادات من config
        config = st.session_state.config
        frame_length = config.get('frame_length', 256)
        frame_step = config.get('frame_step', 160) 
        fft_length = config.get('fft_length', 384)
        
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
    try:
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        
        # Greedy decoding
        results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
        
        # تحويل لنص
        output_text = []
        for result in results:
            if st.session_state.num_to_char:
                chars = st.session_state.num_to_char(result)
                text = tf.strings.reduce_join(chars).numpy().decode("utf-8")
                output_text.append(text)
        
        return output_text[0] if output_text else ""
    
    except Exception as e:
        st.error(f"❌ خطأ في فك التشفير: {e}")
        return ""

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
        prediction = st.session_state.model(spectrogram, training=False)
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
        
        # عرض التقدم
        progress_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        for i in range(duration):
            time.sleep(1)
            progress = (i + 1) / duration
            progress_bar.progress(progress)
            progress_placeholder.text(f"🎙️ جاري التسجيل... {i + 1}/{duration} ثانية")
        
        sd.wait()  # انتظار انتهاء التسجيل
        
        # حفظ الملف
        wavfile.write(temp_filename, sample_rate, recording)
        
        progress_placeholder.empty()
        progress_bar.empty()
        
        return temp_filename
    
    except Exception as e:
        st.error(f"❌ خطأ في التسجيل: {e}")
        return None

# المحتوى الرئيسي
if st.session_state.model is None:
    st.markdown('<div class="success-box">', unsafe_allow_html=True)
    st.info("""
    ## 🎯 مرحباً بك في نظام التعرف على الكلام الحقيقي!
    
    **لبدء الاستخدام، يرجى تحميل النموذج من الشريط الجانبي:**
    
    1. **رفع النموذج**: ملف `.h5` أو `.keras`
    2. **رفع الإعدادات**: ملف `config.json` 
    3. **رفع المعالجات**: ملف `preprocessors.pkl`
    
    ### 📁 مثال على هيكل الملفات:
    ```
    النموذج المدرب/
    ├── my_model.h5           # النموذج المحفوظ
    ├── config.json           # إعدادات المعالجة
    └── preprocessors.pkl     # معالجات النص
    ```
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # قسم المساعدة
    with st.expander("🆘 مساعدة في تحضير الملفات"):
        st.markdown("""
        ### كيفية إنشاء الملفات المطلوبة:
        
        **1. ملف النموذج (`my_model.h5`):**
        ```python
        model.save('my_model.h5')
        ```
        
        **2. ملف الإعدادات (`config.json`):**
        ```python
        import json
        config = {
            'frame_length': 256,
            'frame_step': 160, 
            'fft_length': 384
        }
        with open('config.json', 'w') as f:
            json.dump(config, f)
        ```
        
        **3. ملف المعالجات (`preprocessors.pkl`):**
        ```python
        import pickle
        preprocessors = {
            'char_to_num': char_to_num,
            'num_to_char': num_to_char
        }
        with open('preprocessors.pkl', 'wb') as f:
            pickle.dump(preprocessors, f)
        ```
        """)
    
    st.stop()

# عرض معلومات النموذج
st.success(f"✅ النموذج محمل وجاهز للاستخدام!")
config = st.session_state.config
st.info(f"**إعدادات النموذج:** Frame Length: {config.get('frame_length', 'N/A')} | Frame Step: {config.get('frame_step', 'N/A')} | FFT Length: {config.get('fft_length', 'N/A')}")

# تبويبات الواجهة
tab1, tab2, tab3 = st.tabs(["🎤 تسجيل من المايك", "📁 تحميل ملف صوتي", "📊 تقييم الأداء"])

with tab1:
    st.header("التسجيل المباشر من الميكروفون")
    
    if not AUDIO_AVAILABLE:
        st.error("""
        ❌ **خاصية التسجيل غير متاحة**
        
        يرجى تثبيت مكتبات الصوت:
        ```bash
        pip install sounddevice scipy
        ```
        """)
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="recording-box">', unsafe_allow_html=True)
            st.subheader("🎙️ التحكم في التسجيل")
            
            if st.button("⏺️ بدء التسجيل", use_container_width=True, type="primary"):
                with st.spinner("جاري التحضير للتسجيل..."):
                    audio_file = record_audio(
                        duration=st.session_state.duration,
                        sample_rate=st.session_state.sample_rate
                    )
                    
                    if audio_file:
                        st.session_state.recorded_audio = audio_file
                        st.success("✅ تم التسجيل بنجاح!")
            
            if st.session_state.get('recorded_audio'):
                st.audio(st.session_state.recorded_audio, format='audio/wav')
                
                if st.button("🔍 تحليل التسجيل", use_container_width=True):
                    with st.spinner("جاري التعرف على الكلام..."):
                        prediction = predict_from_audio(st.session_state.recorded_audio)
                        
                        if prediction:
                            st.session_state.last_prediction = prediction
                            st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.subheader("📝 النتائج")
            if st.session_state.last_prediction:
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.success("**النص المتوقع:**")
                st.write(st.session_state.last_prediction)
                st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.header("تحميل ملف صوتي")
    
    uploaded_audio = st.file_uploader("اختر ملف صوتي WAV", type=['wav'], key="audio_upload")
    
    if uploaded_audio is not None:
        # حفظ الملف المؤقت
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_audio.getvalue())
            audio_path = tmp_file.name
        
        # عرض الملف
        st.audio(uploaded_audio, format='audio/wav')
        
        if st.button("🔍 تحليل الملف الصوتي", use_container_width=True):
            with st.spinner("جاري التعرف على الكلام..."):
                prediction = predict_from_audio(audio_path)
                
                if prediction:
                    st.session_state.last_prediction = prediction
                    st.success("✅ تم التحليل بنجاح!")
                    
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    st.subheader("📝 النص المتوقع:")
                    st.write(prediction)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # تنظيف الملف المؤقت
                os.unlink(audio_path)

with tab3:
    st.header("تقييم أداء النموذج")
    
    st.subheader("مقارنة مع النص المرجعي")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        reference_text = st.text_area(
            "النص المرجعي (الصحيح):",
            placeholder="أدخل النص الصحيح هنا...",
            height=100
        )
    
    with col2:
        predicted_text = st.text_area(
            "النص المتوقع:",
            value=st.session_state.get('last_prediction', ''),
            placeholder="سيظهر النص المتوقع هنا...",
            height=100
        )
    
    if st.button("📊 حساب مقاييس الدقة", use_container_width=True) and reference_text and predicted_text:
        try:
            # حساب المقاييس
            wer_score = wer(reference_text, predicted_text)
            cer_score = cer(reference_text, predicted_text)
            accuracy = max(0, 1 - wer_score)
            
            # عرض النتائج
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("معدل الخطأ في الكلمات (WER)", f"{wer_score:.4f}")
            with col2:
                st.metric("معدل الخطأ في الحروف (CER)", f"{cer_score:.4f}")
            with col3:
                st.metric("الدقة التقريبية", f"{accuracy:.2%}")
            
            # تفسير النتائج
            if wer_score < 0.1:
                st.success("🎉 دقة ممتازة! النموذج يعمل بشكل رائع")
            elif wer_score < 0.3:
                st.info("✅ دقة جيدة! النموذج يعمل بشكل مقبول")
            else:
                st.warning("⚠️ الدقة تحتاج تحسين. جرب تسجيلات أوضح")
                
        except Exception as e:
            st.error(f"❌ خطأ في الحساب: {e}")

# قسم المعلومات
with st.expander("ℹ️ معلومات عن النظام"):
    st.markdown("""
    ### 🎯 ميزات النظام:
    - ✅ تحميل النموذج الحقيقي المدرب
    - ✅ التسجيل المباشر من الميكروفون
    - ✅ تحليل ملفات صوتية مرفوعة
    - ✅ تقييم دقة النموذج
    - ✅ واجهة عربية كاملة
    
    ### 💡 نصائح للحصول على أفضل النتائج:
    1. استخدم ميكروفون جيد النوعية
    2. تسجل في بيئة هادئة
    3. تحدث بوضوح وبطء معتدل
    4. استخدم معدل عينات 16kHz للأفضل
    5. تجنب الضوضاء الخلفية
    """)

# تذييل الصفحة
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🎤 نظام التعرف على الكلام - النموذج الحقيقي | تم التطوير باستخدام TensorFlow & Streamlit"
    "</div>",
    unsafe_allow_html=True
)