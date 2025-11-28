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

# محاولة استيراد مكتبات الصوت
try:
    import pyaudio
    import wave
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

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

# تحميل النموذج تلقائياً عند التشغيل
@st.cache_resource
def load_model_and_config():
    """تحميل النموذج والإعدادات تلقائياً"""
    try:
        # تحميل النموذج
        model = keras.models.load_model('my_model (1).h5', compile=False)
        
        # تحميل الإعدادات
        with open('config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # تحميل المعالجات
        with open('preprocessors.pkl', 'rb') as f:
            preprocessors = pickle.load(f)
            char_to_num = preprocessors.get('char_to_num')
            num_to_char = preprocessors.get('num_to_char')
        
        return model, config, char_to_num, num_to_char
        
    except Exception as e:
        st.error(f"❌ خطأ في تحميل النموذج: {e}")
        return None, None, None, None

# تحميل النموذج تلقائياً
if 'model' not in st.session_state:
    with st.spinner("🔄 جاري تحميل النموذج والإعدادات..."):
        model, config, char_to_num, num_to_char = load_model_and_config()
        
        if model is not None:
            st.session_state.model = model
            st.session_state.config = config
            st.session_state.char_to_num = char_to_num
            st.session_state.num_to_char = num_to_char
            st.session_state.model_loaded = True
        else:
            st.session_state.model_loaded = False

# تهيئة حالة الجلسة
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = ""
if 'recorded_audio' not in st.session_state:
    st.session_state.recorded_audio = None

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

# دالة تسجيل الصوت باستخدام PyAudio
def record_audio_pyaudio(duration=5, sample_rate=16000, channels=1):
    """تسجيل الصوت من الميكروفون باستخدام PyAudio"""
    try:
        # إعدادات التسجيل
        chunk = 1024
        format = pyaudio.paInt16
        
        # إنشاء كائن PyAudio
        p = pyaudio.PyAudio()
        
        # فتح stream للتسجيل
        stream = p.open(
            format=format,
            channels=channels,
            rate=sample_rate,
            input=True,
            frames_per_buffer=chunk
        )
        
        st.info("🎙️ جاري التسجيل... تكلم الآن!")
        
        frames = []
        
        # حساب عدد القطع المطلوبة
        total_chunks = int((sample_rate / chunk) * duration)
        
        # شريط التقدم
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # التسجيل
        for i in range(total_chunks):
            data = stream.read(chunk)
            frames.append(data)
            
            # تحديث شريط التقدم
            progress = (i + 1) / total_chunks
            progress_bar.progress(progress)
            status_text.text(f"⏳ {int(progress * 100)}% - {i + 1}/{total_chunks}")
        
        # إيقاف التسجيل
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        progress_bar.empty()
        status_text.empty()
        
        # حفظ الملف المؤقت
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_filename = temp_file.name
        temp_file.close()
        
        # حفظ كملف WAV
        wf = wave.open(temp_filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        st.success("✅ تم التسجيل بنجاح!")
        return temp_filename
        
    except Exception as e:
        st.error(f"❌ خطأ في التسجيل: {e}")
        return None

# المحتوى الرئيسي
if not st.session_state.get('model_loaded', False):
    st.error("""
    ❌ **لم يتم تحميل النموذج بنجاح**
    
    **تأكد من وجود هذه الملفات في نفس المجلد:**
    - `my_model (1).h5` - النموذج المدرب
    - `config.json` - إعدادات المعالجة  
    - `preprocessors.pkl` - معالجات النص
    
    **طريقة الحل:**
    1. تأكد أن الملفات موجودة في نفس مجلد التطبيق
    2. تأكد أن أسماء الملفات مطابقة تماماً
    3. جدد تحميل الصفحة
    """)
    st.stop()

# عرض معلومات النموذج
st.success("✅ النموذج محمل وجاهز للاستخدام!")
config = st.session_state.config
st.info(f"**إعدادات النموذج:** Frame Length: {config.get('frame_length')} | Frame Step: {config.get('frame_step')} | FFT Length: {config.get('fft_length')}")

# الشريط الجانبي للإعدادات
with st.sidebar:
    st.header("⚙️ إعدادات التسجيل")
    
    if PYAUDIO_AVAILABLE:
        duration = st.slider("مدة التسجيل (ثواني)", 1, 15, 5)
        sample_rate = st.selectbox("معدل العينات", [16000, 22050, 44100], index=0)
        channels = st.selectbox("عدد القنوات", [1, 2], index=0)
    else:
        st.error("""
        ❌ **PyAudio غير مثبت**
        
        للتسجيل من المايكروفون:
        ```bash
        pip install pyaudio
        ```
        """)

# تبويبات الواجهة
tab1, tab2, tab3 = st.tabs(["🎤 تسجيل من المايك", "📁 تحميل ملف صوتي", "📊 تقييم الأداء"])

with tab1:
    st.header("التسجيل المباشر من الميكروفون")
    
    if not PYAUDIO_AVAILABLE:
        st.error("""
        ## ❌ خاصية التسجيل غير متاحة
        
        **لتمكين التسجيل من المايكروفون:**
        
        **على Windows:**
        ```bash
        pip install pipwin
        pipwin install pyaudio
        ```
        
        **على Mac/Linux:**
        ```bash
        pip install pyaudio
        ```
        
        **بديل فوري:** استخدم تبويب "تحميل ملف صوتي" لرفع تسجيلات جاهزة
        """)
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="recording-box">', unsafe_allow_html=True)
            st.subheader("🎙️ التحكم في التسجيل")
            
            if st.button("⏺️ بدء التسجيل", use_container_width=True, type="primary"):
                with st.spinner("جاري إعداد التسجيل..."):
                    audio_file = record_audio_pyaudio(
                        duration=duration,
                        sample_rate=sample_rate,
                        channels=channels
                    )
                    
                    if audio_file:
                        st.session_state.recorded_audio = audio_file
                        st.rerun()
            
            # عرض التسجيل إذا موجود
            if st.session_state.recorded_audio:
                st.audio(st.session_state.recorded_audio, format='audio/wav')
                
                col_btn1, col_btn2 = st.columns(2)
                
                with col_btn1:
                    if st.button("🔍 تحليل التسجيل", use_container_width=True):
                        with st.spinner("جاري التعرف على الكلام..."):
                            prediction = predict_from_audio(st.session_state.recorded_audio)
                            
                            if prediction:
                                st.session_state.last_prediction = prediction
                                st.success("✅ تم التحليل بنجاح!")
                                st.rerun()
                
                with col_btn2:
                    if st.button("🗑️ مسح التسجيل", use_container_width=True):
                        try:
                            if st.session_state.recorded_audio and os.path.exists(st.session_state.recorded_audio):
                                os.unlink(st.session_state.recorded_audio)
                            st.session_state.recorded_audio = None
                            st.session_state.last_prediction = ""
                            st.rerun()
                        except:
                            pass
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.subheader("📝 النتائج")
            if st.session_state.last_prediction:
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.success("**النص المتوقع:**")
                st.write(st.session_state.last_prediction)
                
                # إحصاءات النص
                text_length = len(st.session_state.last_prediction)
                word_count = len(st.session_state.last_prediction.split())
                
                col_stat1, col_stat2 = st.columns(2)
                with col_stat1:
                    st.metric("عدد الحروف", text_length)
                with col_stat2:
                    st.metric("عدد الكلمات", word_count)
                
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("👆 سجل صوتاً أولاً ثم اضغط على تحليل التسجيل")

with tab2:
    st.header("تحميل ملف صوتي")
    
    uploaded_audio = st.file_uploader("اختر ملف صوتي WAV", type=['wav'], key="audio_upload")
    
    if uploaded_audio is not None:
        # حفظ الملف المؤقت
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_audio.getvalue())
            audio_path = tmp_file.name
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # عرض الملف
            st.audio(uploaded_audio, format='audio/wav')
        
        with col2:
            # معلومات الملف
            file_size = len(uploaded_audio.getvalue()) / 1024
            st.metric("حجم الملف", f"{file_size:.1f} KB")
            st.metric("النوع", "WAV")
        
        if st.button("🔍 تحليل الملف الصوتي", use_container_width=True):
            with st.spinner("جاري التعرف على الكلام..."):
                prediction = predict_from_audio(audio_path)
                
                if prediction:
                    st.session_state.last_prediction = prediction
                    st.success("✅ تم التحليل بنجاح!")
                    
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    st.subheader("📝 النص المتوقع:")
                    st.write(prediction)
                    
                    # إحصاءات النص
                    text_length = len(prediction)
                    word_count = len(prediction.split())
                    
                    col_stat1, col_stat2 = st.columns(2)
                    with col_stat1:
                        st.metric("عدد الحروف", text_length)
                    with col_stat2:
                        st.metric("عدد الكلمات", word_count)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # تنظيف الملف المؤقت
                try:
                    os.unlink(audio_path)
                except:
                    pass

with tab3:
    st.header("تقييم أداء النموذج")
    
    st.subheader("مقارنة مع النص المرجعي")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        reference_text = st.text_area(
            "النص المرجعي (الصحيح):",
            placeholder="أدخل النص الصحيح هنا...",
            height=120,
            key="ref_text"
        )
    
    with col2:
        predicted_text = st.text_area(
            "النص المتوقع:",
            value=st.session_state.get('last_prediction', ''),
            placeholder="سيظهر النص المتوقع هنا...",
            height=120,
            key="pred_text"
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
            st.subheader("📈 تفسير النتائج:")
            if wer_score == 0:
                st.success("🎉 **ممتاز**: النموذج تعرف على النص بشكل كامل!")
            elif wer_score < 0.1:
                st.success("🔹 **ممتاز**: النموذج يعمل بدقة عالية جداً")
            elif wer_score < 0.3:
                st.info("🔸 **جيد**: النموذج يعمل بدقة مقبولة")
            elif wer_score < 0.5:
                st.warning("⚠️ **متوسط**: النموذج يحتاج لتحسين")
            else:
                st.error("❌ **منخفض**: دقة النموذج منخفضة وتحتاج تحسين كبير")
                
        except Exception as e:
            st.error(f"❌ خطأ في حساب المقاييس: {e}")

# قسم المعلومات
with st.expander("ℹ️ معلومات عن النظام"):
    st.markdown("""
    ### 🎯 ميزات النظام:
    - ✅ تحميل النموذج تلقائياً من الملفات
    - ✅ التسجيل المباشر من الميكروفون (PyAudio)
    - ✅ تحليل ملفات صوتية مرفوعة  
    - ✅ تقييم دقة النموذج
    - ✅ واجهة عربية كاملة
    - ✅ إحصاءات النص تلقائياً
    
    ### 💡 نصائح للحصول على أفضل النتائج:
    1. **استخدم ميكروفون جيد** النوعية
    2. **سجل في بيئة هادئة** بعيداً عن الضوضاء
    3. **تحدث بوضوح** وبطء معتدل
    4. **استخدم معدل عينات 16kHz** للأفضل
    5. **تجنب الصدى** والضوضاء الخلفية
    """)

# تذييل الصفحة
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🎤 نظام التعرف على الكلام - النموذج محمل تلقائياً 🚀"
    "</div>",
    unsafe_allow_html=True
)