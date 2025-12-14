import streamlit as st
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import sounddevice as sd
import scipy.io.wavfile as wavfile
import numpy as np
import tempfile
import os
from audio_recorder_streamlit import audio_recorder
import time
import librosa
import soundfile as sf

# Konfigurasi halaman
st.set_page_config(
    page_title="Belajar Membaca 📚",
    page_icon="🎤",
    layout="wide"
)

# CSS Custom untuk tema anak-anak
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stButton>button {
        background-color: #FF6B6B;
        color: white;
        font-size: 20px;
        padding: 15px 30px;
        border-radius: 25px;
        border: none;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #FF5252;
        transform: scale(1.05);
    }
    .title-box {
        background: white;
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        margin-bottom: 30px;
    }
    .sentence-box {
        background: linear-gradient(135deg, #FEC163 0%, #DE4313 100%);
        padding: 25px;
        border-radius: 20px;
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        color: white;
        margin: 20px 0;
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    .result-box {
        background: white;
        padding: 25px;
        border-radius: 20px;
        margin: 20px 0;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
    }
    .success-box {
        background: #4CAF50;
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        animation: bounce 0.5s;
    }
    .try-again-box {
        background: #FFA726;
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
    }
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    .instruction-box {
        background: rgba(255,255,255,0.9);
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# Judul aplikasi
st.markdown("""
<div class="title-box">
    <h1>🎤 Belajar Membaca dengan Suara 📚</h1>
    <p style="font-size: 18px; color: #666;">Yuk, belajar membaca dengan menyebutkan kalimat!</p>
</div>
""", unsafe_allow_html=True)

# Cache model untuk performa
@st.cache_resource
def load_model():
    model_id = "khaerulilman/Speech-to-text-model"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    with st.spinner("🔄 Memuat model AI... Tunggu sebentar ya!"):
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        model.to(device)
        
        processor = AutoProcessor.from_pretrained(model_id)
        
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
            generate_kwargs={"language": "indonesian", "task": "transcribe"}
        )
    
    return pipe

# Daftar kalimat untuk latihan
sentences = [
    "indonesia adalah negara yang megah",
    "aku suka bermain di taman",
    "ibu memasak di dapur",
    "ayah pergi ke kantor",
    "kucing tidur di atas kasur",
    "bunga mawar berwarna merah",
    "hari ini cuaca cerah sekali",
    "aku belajar membaca buku"
]

# Inisialisasi session state
if 'current_sentence' not in st.session_state:
    st.session_state.current_sentence = sentences[0]
if 'result_text' not in st.session_state:
    st.session_state.result_text = None
if 'show_result' not in st.session_state:
    st.session_state.show_result = False
if 'audio_cache' not in st.session_state:
    st.session_state.audio_cache = None
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'recorder_key' not in st.session_state:
    st.session_state.recorder_key = 0

# Sidebar untuk pilihan kalimat
with st.sidebar:
    st.markdown("### 📝 Pilih Kalimat")
    selected_sentence = st.selectbox(
        "Kalimat untuk dibaca:",
        sentences,
        index=sentences.index(st.session_state.current_sentence)
    )
    st.session_state.current_sentence = selected_sentence
    
    st.markdown("---")
    st.markdown("### ℹ️ Cara Bermain")
    st.markdown("""
    1. 👀 Lihat kalimat yang muncul
    2. 🎤 Rekam atau upload suaramu
    3. ✅ Lihat hasilnya!
    4. 🎉 Coba lagi jika belum tepat
    """)

# Tampilkan kalimat yang harus dibaca
st.markdown(f"""
<div class="sentence-box">
    📖 Bacalah kalimat ini: <br>
    "{st.session_state.current_sentence}"
</div>
""", unsafe_allow_html=True)

# Layout untuk input audio
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="instruction-box">
        <h3 style="color: #667eea;">🎙️ Rekam Suaramu</h3>
        <p>Klik tombol merah untuk mulai/berhenti rekam!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Tombol hapus cache di atas recorder
    if st.session_state.audio_cache:
        if st.button("🗑️ Hapus Rekaman & Rekam Ulang", key="clear_mic_top", use_container_width=True, type="secondary"):
            st.session_state.audio_cache = None
            st.session_state.is_recording = False
            st.session_state.recorder_key += 1  # Increment key untuk force refresh
            st.success("✅ Rekaman dihapus! Silakan rekam ulang.")
            time.sleep(0.5)
            st.rerun()
    
    # Audio recorder dengan dynamic key
    audio_bytes = audio_recorder(
        text="",
        recording_color="#FF6B6B",
        neutral_color="#667eea",
        icon_size="3x",
        key=f"audio_recorder_{st.session_state.recorder_key}",
        pause_threshold=2.0,
    )
    
    # Deteksi perubahan status rekaman - hanya update jika benar-benar ada perubahan
    if audio_bytes is not None:
        # Cek apakah ini audio baru (berbeda dari cache)
        if st.session_state.audio_cache is None or audio_bytes != st.session_state.audio_cache:
            # Hanya update jika audio_bytes lebih panjang (berarti recording selesai)
            if st.session_state.audio_cache is None or len(audio_bytes) > len(st.session_state.audio_cache):
                st.session_state.audio_cache = audio_bytes
                st.session_state.is_recording = False
                st.rerun()
    
    # Tampilkan preview audio HANYA jika ada cache dan tidak sedang recording
    if st.session_state.audio_cache and not st.session_state.is_recording:
        st.success("✅ Rekaman selesai! Audio tersimpan.")
        st.info("💡 Klik tombol 'Proses Suara Saya!' di bawah untuk transkripsi.")
        st.audio(st.session_state.audio_cache, format='audio/wav')

with col2:
    st.markdown("""
    <div class="instruction-box">
        <h3 style="color: #667eea;">📁 Upload File Audio</h3>
        <p>Atau upload file audio WAV kamu!</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Pilih file audio (WAV)",
        type=['wav'],
        label_visibility="collapsed",
        key="file_uploader"
    )
    
    # Tampilkan status upload
    if uploaded_file:
        st.success("✅ File terupload!")
        st.audio(uploaded_file, format='audio/wav')
        
        # Tombol hapus file upload
        if st.button("🗑️ Hapus File & Upload Ulang", key="clear_file", use_container_width=True, type="secondary"):
            # Clear dengan rerun
            st.session_state.pop('file_uploader', None)
            st.success("✅ File dihapus!")
            time.sleep(0.5)
            st.rerun()

# Fungsi untuk memproses audio
def process_audio(audio_data, pipe):
    try:
        # Progress bar yang menarik
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.markdown("🎵 Memproses audio...")
        progress_bar.progress(30)
        time.sleep(0.3)
        
        status_text.markdown("🤖 AI sedang mendengarkan...")
        progress_bar.progress(60)
        
        # Konversi audio bytes ke numpy array menggunakan librosa
        # Ini akan handle berbagai format audio (wav, webm, dll)
        import io
        
        # Simpan ke temporary file terlebih dahulu
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_data)
            tmp_path = tmp_file.name
        
        try:
            # Load audio dengan librosa - ini akan handle berbagai format
            audio_array, sampling_rate = librosa.load(tmp_path, sr=16000, mono=True)
            
            # Normalisasi audio
            audio_array = audio_array.astype(np.float32)
            
            # Proses dengan model - pipeline whisper akan handle array numpy
            result = pipe(audio_array)
            
            status_text.markdown("✨ Menganalisis hasil...")
            progress_bar.progress(90)
            time.sleep(0.2)
            
            progress_bar.progress(100)
            status_text.markdown("✅ Selesai!")
            time.sleep(0.5)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            return result["text"].lower().strip()
        
        finally:
            # Hapus file temporary
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    except Exception as e:
        st.error(f"❌ Error memproses audio: {str(e)}")
        st.info("💡 Pastikan audio terekam dengan jelas dan coba lagi!")
        return None

# Tombol proses
process_col1, process_col2 = st.columns([3, 1])

with process_col1:
    if st.button("🚀 Proses Suara Saya!", use_container_width=True):
        audio_data = None
        
        # Cek sumber audio - prioritaskan audio dari microphone cache
        if st.session_state.audio_cache:
            audio_data = st.session_state.audio_cache
            st.info("📝 Memproses audio dari rekaman microphone...")
        elif uploaded_file:
            audio_data = uploaded_file.read()
            st.info("📝 Memproses audio dari file upload...")
        
        if audio_data:
            # Load model
            pipe = load_model()
            
            # Proses audio
            result_text = process_audio(audio_data, pipe)
            
            if result_text:
                st.session_state.result_text = result_text
                st.session_state.show_result = True
                st.rerun()
        else:
            st.warning("⚠️ Rekam atau upload audio terlebih dahulu!")
            st.info("💡 Tips: Klik tombol microphone untuk merekam, atau upload file WAV")

with process_col2:
    if st.button("🔄 Reset Semua", use_container_width=True, type="secondary"):
        # Hapus SEMUA cache dan state
        st.session_state.audio_cache = None
        st.session_state.result_text = None
        st.session_state.show_result = False
        st.session_state.is_recording = False
        st.session_state.recorder_key += 1  # Force refresh recorder
        
        # Hapus file uploader
        if 'file_uploader' in st.session_state:
            st.session_state.pop('file_uploader')
        
        st.success("✅ Semua data telah direset!")
        time.sleep(0.8)
        st.rerun()

# Tampilkan hasil
if st.session_state.show_result and st.session_state.result_text:
    st.markdown("---")
    
    expected = st.session_state.current_sentence.lower().strip()
    actual = st.session_state.result_text
    
    # Tampilkan kalimat yang seharusnya dibaca
    st.markdown(f"""
    <div class="result-box">
        <h3 style="color: #667eea;">📖 Kalimat yang Seharusnya:</h3>
        <h2 style='color: #333; text-align: center; margin: 15px 0;'>"{expected}"</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Tampilkan hasil prediksi model
    st.markdown(f"""
    <div class="result-box" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); margin-top: 20px;">
        <h3 style="color: white;">🎤 Hasil Prediksi Model:</h3>
        <h2 style='color: white; text-align: center; margin: 15px 0;'>"{actual}"</h2>
    </div>
    """, unsafe_allow_html=True)


# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: white; padding: 20px;">
    <p>💡 Tip: Bacalah dengan jelas dan lantang untuk hasil terbaik!</p>
    <p>🎈 Buat oleh AI untuk belajar yang menyenangkan</p>
</div>
""", unsafe_allow_html=True)