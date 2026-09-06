import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import plotly.express as px
import emoji # Untuk demojize emoji di preprocessing
from googleapiclient.discovery import build # Untuk YouTube API
from transformers import pipeline # Untuk model sentiment fine-tuned Anda
import torch # Biasanya diperlukan oleh transformers

# Impor untuk BERTopic dan visualisasi (dari kode dashboard sebelumnya)
from bertopic import BERTopic
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import collections
# import hdbscan # BERTopic akan mengimpornya jika menggunakan clustering default

# Impor untuk Google Sheets (opsional, untuk menyimpan hasil)
import gspread
from gspread_dataframe import set_with_dataframe
from google.oauth2.service_account import Credentials
import os

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(page_title="Dashboard Analisis YouTube Lanjutan", layout="wide")
st.title("📊 Dashboard Analisis Komentar YouTube dengan Model Fine-tuned & BERTopic")

# --- Fungsi untuk Memuat Kredensial dengan Aman ---
# Untuk YouTube API Key
def get_youtube_api_key():
    try:
        return st.secrets["youtube"]["api_key"]
    except:
        st.error("YouTube API Key tidak ditemukan di st.secrets. Harap konfigurasikan.")
        return None

# Untuk Google Sheets JSON Key
def get_gsheets_credentials():
    try:
        creds_json = {
            "type": st.secrets["gsheets"]["type"],
            "project_id": st.secrets["gsheets"]["project_id"],
            "private_key_id": st.secrets["gsheets"]["private_key_id"],
            "private_key": st.secrets["gsheets"]["private_key"].replace('\\n', '\n'), # Ganti escape char
            "client_email": st.secrets["gsheets"]["client_email"],
            "client_id": st.secrets["gsheets"]["client_id"],
            "auth_uri": st.secrets["gsheets"]["auth_uri"],
            "token_uri": st.secrets["gsheets"]["token_uri"],
            "auth_provider_x509_cert_url": st.secrets["gsheets"]["auth_provider_x509_cert_url"],
            "client_x509_cert_url": st.secrets["gsheets"]["client_x509_cert_url"]
        }
        return Credentials.from_service_account_info(creds_json, scopes=["https://www.googleapis.com/auth/spreadsheets"])
    except Exception as e:
        st.error(f"Gagal memuat kredensial Google Sheets dari st.secrets: {e}")
        return None

# --- Cache untuk Model (agar tidak di-load ulang setiap kali) ---
@st.cache_resource # Penting untuk performa
def load_sentiment_model_locally(): # Nama fungsi diubah untuk kejelasan
    # Path relatif ke folder model Anda di dalam repositori GitHub
    local_model_path = "model/fine-tuned-indobert" 
    
    st.info(f"Memuat model sentimen dari path lokal: '{local_model_path}'...")
    
    # Opsional: Cek apakah direktori ada saat pengembangan lokal
    if not os.path.isdir(local_model_path):
        st.error(f"Direktori model lokal '{local_model_path}' tidak ditemukan. Pastikan path sudah benar dan folder ada di repositori.")
        # Anda mungkin perlu menambahkan path absolut untuk debugging lokal:
        st.info(f"Path absolut yang dicari: {os.path.abspath(local_model_path)}")
        return None

    try:
        sentiment_pipeline_instance = pipeline(
            "text-classification",
            model=local_model_path,             # Menggunakan path lokal
            tokenizer=local_model_path,         # Menggunakan path lokal (biasanya sama)
            truncation=True,
            max_length=512,
            device=0 if torch.cuda.is_available() else -1 # Gunakan GPU jika tersedia
        )
        st.success(f"Model sentimen berhasil dimuat dari '{local_model_path}'!")
        return sentiment_pipeline_instance
    except Exception as e:
        st.error(f"Gagal memuat model sentimen dari '{local_model_path}': {e}")
        st.error("Pastikan semua file model dan tokenizer yang diperlukan (termasuk model.safetensors, config.json, dll.) ada di direktori tersebut dan Git LFS telah menarik file dengan benar saat deployment.")
        st.exception(e) # Tampilkan traceback untuk debugging lebih lanjut
        return None
        
sentiment_model_pipeline = load_sentiment_model_locally()

# --- Fungsi dari kode Colab Anda (dengan sedikit penyesuaian) ---
def fetch_youtube_comments(video_id, api_key_youtube):
    if not api_key_youtube:
        return None
    replies_data = []
    try:
        youtube = build('youtube', 'v3', developerKey=api_key_youtube)
        video_response = youtube.commentThreads().list(part='snippet,replies', videoId=video_id, maxResults=100).execute() # maxResults bisa disesuaikan

        while video_response:
            for item in video_response['items']:
                published = item['snippet']['topLevelComment']['snippet']['publishedAt']
                user = item['snippet']['topLevelComment']['snippet']['authorDisplayName']
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                likeCount = item['snippet']['topLevelComment']['snippet']['likeCount']
                replies_data.append([published, user, comment, likeCount])

                if item['snippet']['totalReplyCount'] > 0 and 'replies' in item: # Cek 'replies' key
                    for reply in item['replies']['comments']:
                        published_reply = reply['snippet']['publishedAt']
                        user_reply = reply['snippet']['authorDisplayName']
                        repl_text = reply['snippet']['textDisplay']
                        likeCount_reply = reply['snippet']['likeCount']
                        replies_data.append([published_reply, user_reply, repl_text, likeCount_reply])
            
            if 'nextPageToken' in video_response:
                video_response = youtube.commentThreads().list(
                    part='snippet,replies',
                    pageToken=video_response['nextPageToken'],
                    videoId=video_id,
                    maxResults=100
                ).execute()
            else:
                break
        return pd.DataFrame(replies_data, columns=['Time', 'DisplayName', 'Text', 'likeCount'])
    except Exception as e:
        st.error(f"Gagal mengambil komentar YouTube: {e}")
        return None

# FUNGSI BARU UNTUK MENGAMBIL STATISTIK VIDEO
def get_video_statistics(video_id, api_key_youtube):
    if not api_key_youtube:
        st.error("API Key YouTube tidak tersedia untuk mengambil statistik video.")
        return None
    try:
        youtube = build('youtube', 'v3', developerKey=api_key_youtube)
        video_response = youtube.videos().list(
            part='snippet,statistics',  # Ambil snippet untuk judul, statistics untuk like, view, dll.
            id=video_id
        ).execute()

        if video_response.get('items'):
            video_item = video_response['items'][0]
            stats = video_item.get('statistics', {})
            snippet = video_item.get('snippet', {})
            
            video_title = snippet.get('title', "Judul Tidak Diketahui")
            like_count = stats.get('likeCount')
            view_count = stats.get('viewCount')
            # comment_count_from_api = stats.get('commentCount') # Ini jumlah total komentar (thread) menurut API

            return {
                "title": video_title,
                "like_count": int(like_count) if like_count is not None else 0,
                "view_count": int(view_count) if view_count is not None else 0,
                # "api_comment_count": int(comment_count_from_api) if comment_count_from_api is not None else 0
            }
        else:
            st.warning(f"Tidak ditemukan statistik untuk video ID: {video_id}")
            return None
    except Exception as e:
        st.error(f"Gagal mengambil statistik video: {e}")
        return None

# Lexicon dan fungsi preprocessing dari Colab
slang_lexicon = {
    "gk": "tidak", "ga": "tidak", "nggak": "tidak", "ngga": "tidak", "bgt": "banget",
    "btw": "ngomong-ngomong", "pdhl": "padahal", "tp": "tapi", "jd": "jadi", "dr": "dari",
    "yg": "yang", "udh": "sudah", "dpt": "dapat", "trs": "terus", "sm": "sama", "sy": "saya",
    "gw": "saya", "gue": "saya", "lo": "kamu", "lu": "kamu", "aja": "saja",
    # Tambahkan lebih banyak jika perlu
}

def translate_emoji_auto(text):
    demojized = emoji.demojize(text, language='en')
    demojized = demojized.replace(":", " ")
    demojized = demojized.replace("_", " ")
    return demojized

def apply_lexicon(text, lexicon):
    return ' '.join([lexicon.get(word, word) for word in text.split()])

def light_preprocess_text(text):
    text = str(text) # Pastikan input adalah string
    text = translate_emoji_auto(text)
    text = text.strip().lower()
    text = re.sub(r'<.*?>', ' ', text) # Hapus HTML tags
    text = re.sub(r"http\S+|www\S+|https\S+", "<url>", text) # Ganti URL
    text = re.sub(r"@\w+", "<user>", text) # Ganti mention
    text = re.sub(r"#\w+", "", text) # Hapus hashtag
    text = re.sub(r'[^\w\s<>]', '', text) # Hapus tanda baca kecuali untuk <tag>
    text = re.sub(r'(.)\1{2,}', r'\1\1', text) # Normalisasi karakter berulang (misal, bagusss -> baguss)
    text = re.sub(r'\s+', ' ', text).strip() # Hapus spasi berlebih
    text = apply_lexicon(text, slang_lexicon) # Terapkan lexicon slang
    return text

def classify_sentiment_text(text):
    if not sentiment_model_pipeline or not text or pd.isna(text):
        return "neutral" # Atau penanganan lain untuk input tidak valid
    try:
        result = sentiment_model_pipeline(text)[0]
        label = result["label"].lower()
        score = result["score"]
        
        # Logika threshold dari kode Colab Anda.
        # Catatan: Ini mungkin perlu disesuaikan. Biasanya, label dari model langsung digunakan.
        # Jika model Anda sudah mengeluarkan 'positive', 'negative', 'neutral', bagian if score < 0.5 ini mungkin tidak ideal.
        if score < 0.5 and label in ["positive", "negative"]: # Hanya override jika skor rendah untuk label non-netral
             # st.caption(f"Warning: Skor rendah ({score:.2f}) untuk label '{label}' pada teks '{text[:50]}...'. Diklasifikasikan sebagai netral.")
             return "neutral"
        return label
    except Exception as e:
        # st.warning(f"Error saat klasifikasi sentimen untuk teks: '{text[:50]}...' Error: {e}")
        return "neutral" # Fallback jika ada error

# Inisialisasi Sastrawi StopWordRemover (jika ingin digunakan, kode Anda mengimpor tapi tidak memakai)
# factory = StopWordRemoverFactory()
# sastrawi_stopwords = factory.get_stop_words()


# Impor yang mungkin belum ada di bagian atas file Anda (pastikan sudah ada):
import pandas as pd # Untuk pd.api.types.is_string_dtype dan pd.DataFrame()

# BAGIAN 1: Input Pengguna (URL YouTube) dan Inisialisasi Session State
# (Ini sudah ada di kode Anda, saya sertakan untuk konteks)
st.sidebar.header("Input Data")
youtube_url_input = st.sidebar.text_input("Masukkan URL Video YouTube:")
analyze_button = st.sidebar.button("🚀 Ambil & Analisis Komentar")

# Inisialisasi session state untuk menyimpan DataFrame dan statistik video
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'raw_df' not in st.session_state:
    st.session_state.raw_df = None
if 'current_video_title' not in st.session_state:
    st.session_state.current_video_title = "Belum ada video yang dianalisis"
if 'current_video_like_count' not in st.session_state:
    st.session_state.current_video_like_count = 0
if 'current_video_view_count' not in st.session_state:
    st.session_state.current_video_view_count = 0
if 'total_fetched_comments' not in st.session_state:
    st.session_state.total_fetched_comments = 0
if 'analyze_button_pressed_once' not in st.session_state: # Flag untuk menandai tombol pernah ditekan
    st.session_state.analyze_button_pressed_once = False

# Variabel untuk kolom yang akan digunakan di dashboard
TEXT_COLUMN_FOR_ANALYSIS = "cleanedText"
RAW_TEXT_COLUMN = "Text"
SENTIMENT_COLUMN = "sentimen"
LIKE_COUNT_COLUMN = "likeCount" # Ini adalah like per komentar
TIME_COLUMN = "Time"
SENTIMENT_COUNT_COLUMN = "sentimenCount"
# -----------------------------------------------------------------------------

# BAGIAN 2: Pengambilan Data, Pra-Pemrosesan, dan Analisis Sentimen
if analyze_button and youtube_url_input:
    st.session_state.analyze_button_pressed_once = True # Tandai tombol sudah ditekan
    
    # Reset data sebelumnya setiap kali tombol ditekan untuk analisis baru
    st.session_state.processed_df = None
    st.session_state.raw_df = None
    st.session_state.current_video_title = "Memproses..."
    st.session_state.current_video_like_count = 0
    st.session_state.current_video_view_count = 0
    st.session_state.total_fetched_comments = 0

    if not sentiment_model_pipeline: # Pastikan sentiment_model_pipeline sudah di-load di BAGIAN 0
        st.error("Model sentimen tidak berhasil dimuat. Tidak dapat melanjutkan analisis.")
        st.stop() # Hentikan eksekusi jika model sentimen gagal dimuat
    
    video_id = None
    try:
        if "v=" in youtube_url_input:
            video_id = youtube_url_input.split("v=")[-1].split("&")[0]
        elif "youtu.be/" in youtube_url_input:
             video_id = youtube_url_input.split("youtu.be/")[-1].split("?")[0]
        elif "shorts/" in youtube_url_input:
            video_id = youtube_url_input.split("shorts/")[-1].split("?")[0]
        # Tambahkan penanganan untuk googleusercontent.com jika masih relevan atau sering ditemui
        # elif "googleusercontent.com/youtube_content/" in youtube_url_input: # Contoh, sesuaikan polanya
        #     # Logika parsing spesifik untuk URL ini
        #     pass
        else:
            st.error("Format URL YouTube tidak valid atau tidak didukung (misalnya, harus mengandung 'v=', 'youtu.be/', atau 'shorts/').")
            video_id = None # Eksplisit set None
    except Exception as e_url:
        st.error(f"Error memproses URL YouTube: {e_url}")
        video_id = None

    if video_id:
        youtube_api_key = get_youtube_api_key() # Pastikan fungsi ini ada di BAGIAN 0 dan mengembalikan key atau None

        if not youtube_api_key:
            st.error("Kunci API YouTube tidak tersedia. Tidak dapat melanjutkan.")
            st.stop()

        with st.spinner("Mengambil statistik video... 📈"):
            video_stats = get_video_statistics(video_id, youtube_api_key) # Pastikan fungsi ini ada di BAGIAN 0
        
        if video_stats:
            st.session_state.current_video_title = video_stats["title"]
            st.session_state.current_video_like_count = video_stats["like_count"]
            st.session_state.current_video_view_count = video_stats["view_count"]
        else:
            st.session_state.current_video_title = "Judul Video Tidak Dapat Diambil"
            # Biarkan like dan view count 0 jika gagal
        
        with st.spinner("Mengambil komentar dari YouTube... Ini mungkin memerlukan waktu ⏳"):
            df_youtube = fetch_youtube_comments(video_id, youtube_api_key) # Pastikan fungsi ini ada di BAGIAN 0
        st.session_state.raw_df = df_youtube

        if df_youtube is not None and not df_youtube.empty:
            st.success(f"Berhasil mengambil {len(df_youtube)} komentar & balasan!")
            st.session_state.total_fetched_comments = len(df_youtube)
            
            current_df = df_youtube.copy()

            with st.spinner("Melakukan pra-pemrosesan teks... ⚙️"):
                current_df[TEXT_COLUMN_FOR_ANALYSIS] = current_df[RAW_TEXT_COLUMN].apply(light_preprocess_text) # Pastikan fungsi ini ada
            
            with st.spinner("Melakukan analisis sentimen dengan model fine-tuned Anda... 🧠"):
                current_df[SENTIMENT_COLUMN] = current_df[TEXT_COLUMN_FOR_ANALYSIS].apply(classify_sentiment_text) # Pastikan fungsi ini ada
            
            # --- Standardisasi Label Sentimen ke Bahasa Indonesia (PENTING) ---
            # Asumsi: classify_sentiment_text() bisa mengembalikan label B.Inggris atau generik.
            # Kita akan memetakan ke B.Indonesia yang konsisten.
            sentiment_mapping = {
                "negative": "negatif",    # Dari output model B.Inggris
                "positive": "positif",    # Dari output model B.Inggris
                "neutral": "netral",     # Dari output model B.Inggris
                "label_0": "negatif",  # Contoh jika model Anda mengeluarkan label generik (sesuaikan!)
                "label_1": "netral",   # Contoh jika model Anda mengeluarkan label generik (sesuaikan!)
                "label_2": "positif",  # Contoh jika model Anda mengeluarkan label generik (sesuaikan!)
                # Tambahkan mapping lain jika perlu, atau pastikan classify_sentiment_text konsisten
            }
            # Cek dulu apakah kolom SENTIMENT_COLUMN ada dan tipenya string
            if SENTIMENT_COLUMN in current_df.columns and pd.api.types.is_string_dtype(current_df[SENTIMENT_COLUMN]):
                # Terapkan mapping, nilai yang tidak ada di keys akan tetap (atau bisa diisi default jika pakai .get)
                current_df[SENTIMENT_COLUMN] = current_df[SENTIMENT_COLUMN].str.lower().map(sentiment_mapping).fillna(current_df[SENTIMENT_COLUMN].str.lower())
                # Pastikan semuanya lowercase dan tidak ada spasi ekstra setelah mapping
                current_df[SENTIMENT_COLUMN] = current_df[SENTIMENT_COLUMN].str.lower().str.strip()
            # --- Akhir Standardisasi ---

            current_df[TIME_COLUMN] = pd.to_datetime(current_df[TIME_COLUMN], errors="coerce")
            current_df["date"] = current_df[TIME_COLUMN].dt.date
            current_df[LIKE_COUNT_COLUMN] = pd.to_numeric(current_df[LIKE_COUNT_COLUMN], errors="coerce").fillna(0).astype(int)
            current_df[SENTIMENT_COUNT_COLUMN] = current_df[LIKE_COUNT_COLUMN] + 1
            
            st.session_state.processed_df = current_df
            st.success("Pra-pemrosesan dan analisis sentimen selesai!")
            # Checkbox untuk pratinjau data dipindahkan setelah bagian ringkasan umum agar tidak mengganggu alur
        elif df_youtube is None: # Gagal mengambil komentar
            st.error("Gagal mengambil komentar. Periksa API Key atau URL Video.")
            # st.session_state.processed_df sudah di-reset di awal
            # st.session_state.total_fetched_comments sudah di-reset di awal
        else: # Tidak ada komentar ditemukan (df_youtube kosong)
            st.warning("Tidak ada komentar yang ditemukan untuk video ini.")
            st.session_state.processed_df = pd.DataFrame() # Set DataFrame kosong agar tidak error di hilir
            # st.session_state.total_fetched_comments sudah di-reset di awal
    else: # video_id tidak valid atau tidak bisa diekstrak
        if youtube_url_input: # Hanya tampilkan error jika user memang memasukkan sesuatu
             st.error("Video ID tidak dapat diekstrak dari URL yang diberikan. Pastikan URL valid.")
        st.session_state.analyze_button_pressed_once = False # Reset flag jika URL tidak valid dari awal

elif analyze_button and not youtube_url_input:
    st.warning("Harap masukkan URL YouTube terlebih dahulu.")
    st.session_state.analyze_button_pressed_once = False # Reset flag

# --- BAGIAN TAMPILAN UTAMA SETELAH TOMBOL DIKLIK ATAU DATA ADA ---

# Tampilkan Ringkasan Umum Video jika tombol pernah ditekan
# dan judul video bukan placeholder awal (menandakan proses pengambilan statistik setidaknya dimulai)
if st.session_state.analyze_button_pressed_once and st.session_state.current_video_title != "Belum ada video yang dianalisis":
    st.markdown("---")
    st.header(f"📊 Analisis untuk Video: {st.session_state.current_video_title}")

    video_views = st.session_state.current_video_view_count
    video_likes = st.session_state.current_video_like_count
    fetched_comments_count = st.session_state.total_fetched_comments

    col_metric1, col_metric2, col_metric3 = st.columns(3)
    with col_metric1:
        st.metric(label="👁️ Total Tayangan Video", value=f"{video_views:,}")
    with col_metric2:
        st.metric(label="👍 Total Likes Video", value=f"{video_likes:,}")
    with col_metric3:
        st.metric(label="💬 Komentar & Balasan (Diproses)", value=f"{fetched_comments_count:,}")
    st.markdown("---")

    # Tampilkan checkbox pratinjau data di sini, setelah ringkasan
    if st.session_state.processed_df is not None and not st.session_state.processed_df.empty:
        if st.checkbox("Tampilkan pratinjau data hasil analisis komentar", value=True, key="preview_data_checkbox"): # value=True agar default terbuka
            # Pilih kolom yang relevan untuk ditampilkan, pastikan semua ada
            cols_to_show = [RAW_TEXT_COLUMN, TEXT_COLUMN_FOR_ANALYSIS, SENTIMENT_COLUMN, LIKE_COUNT_COLUMN, SENTIMENT_COUNT_COLUMN]
            # Filter hanya kolom yang ada di DataFrame untuk menghindari error
            existing_cols_to_show = [col for col in cols_to_show if col in st.session_state.processed_df.columns]
            if existing_cols_to_show:
                 st.dataframe(st.session_state.processed_df[existing_cols_to_show].head())
            else:
                st.caption("Tidak ada kolom pratinjau yang bisa ditampilkan.")

if st.session_state.processed_df is not None and not st.session_state.processed_df.empty:
    df_dashboard = st.session_state.processed_df.copy()

    # --- PEMBUATAN STOPWORDS (termasuk dinamis dari emoji di cleanedText) ---
    # (Kode ini sama seperti di versi dashboard sebelumnya, tapi menggunakan TEXT_COLUMN_FOR_ANALYSIS)
    from nltk.corpus import stopwords
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError: # <-- GANTI DENGAN LookupError
        st.caption("NLTK stopwords resource not found. Attempting to download...") # Baris ini bisa Anda uncomment untuk debugging
        nltk.download('stopwords', quiet=True)
    
    dynamic_emoji_stopwords = set()
    if not df_dashboard[TEXT_COLUMN_FOR_ANALYSIS].empty:
        all_text_for_emoji_scan = " ".join(df_dashboard[TEXT_COLUMN_FOR_ANALYSIS].astype(str).dropna().unique())
        unique_emoji_chars = set()
        if all_text_for_emoji_scan.strip():
            # Meskipun emoji sudah di-translate_emoji_auto, fungsi ini mencari karakter emoji asli.
            # Seharusnya, setelah translate_emoji_auto, tidak ada lagi karakter emoji asli.
            # Jadi, dynamic_emoji_stopwords mungkin akan kosong jika translate_emoji_auto bekerja sempurna.
            # Jika translate_emoji_auto tidak sempurna atau ada emoji baru, ini masih bisa berguna.
            # Atau, kita bisa membuat stopwords dari *hasil* demojize jika itu yang diinginkan.
            # Untuk saat ini, kita ikuti logika sebelumnya. Jika teks input sudah demojized, bagian ini mungkin tidak banyak menambahkan.
            emoji_data_list_dashboard = emoji.emoji_list(all_text_for_emoji_scan) # cari emoji asli
            for item in emoji_data_list_dashboard:
                unique_emoji_chars.add(item['emoji'])

        if unique_emoji_chars:
            for char_emoji in unique_emoji_chars:
                demojized_text = emoji.demojize(char_emoji, language='en')
                cleaned_demojized_text = demojized_text.strip(':').replace('_', ' ').lower()
                words_from_emoji = re.findall(r'\b\w+\b', cleaned_demojized_text)
                for word in words_from_emoji:
                    if len(word) > 1: 
                        dynamic_emoji_stopwords.add(word)
    
    default_stopwords_id = set(stopwords.words('indonesian'))
    # Ambil custom_stopwords_list dari kode dashboard sebelumnya
    custom_stopwords_list = {
        'yg', 'dg', 'rt', 'dr', 'kpd', 'ny', 'dgn', 'gue', 'lo', 'elu', 'gw', # sangat umum
        'video', 'channel', 'komen', 'komentar', 'konten', 'youtube', 'youtuber', 'subscribe', 'like', 'share', # terkait platform
        'admin', 'kak', 'bang', 'mas', 'mbak', 'gan', 'bro', 'sis', # sapaan umum
        'nya', 'sih', 'dong', 'kok', 'deh', 'mah', 'tuh', 'nih', 'amp', # partikel
        'url', 'user', # dari placeholder preprocessing
        # Tambahkan kata-kata dari slang_lexicon yang mungkin ingin tetap dihilangkan meskipun sudah dinormalisasi
        # Tambahkan kata-kata hasil demojize emoji yang umum jika dynamic_emoji_stopwords tidak menangkapnya (karena teks sudah diproses)
        'face', 'tears', 'joy', 'red', 'heart', 'blue', 'black', 'white', 'green', 'yellow', 'tongue', 'like','tone', 'skin', 'from', 'nose', 'steam',# komponen umum emoji
        'hand', 'hands', 'eyes', 'smiling', 'loudly', 'crying', 'rolling', 'floor', 'laughing', 'on', 'the',
        'thinking', 'pondering', 'pleading', 'point', 'right', 'left', 'up', 'down', 'backhand', 'index',
        'ok', 'flexed', 'biceps', 'folded', 'clapping', 'thumbs', 'of', 'grinning', 'beaming', 'sweat', 'mouth', 'open', 'grimacing',
        'satu', 'dua', 'tiga', 'empat', 'lima', 'enam', 'tujuh', 'delapan', 'sembilan', 'sepuluh', # angka
        'januari', 'februari', 'maret', 'april', 'mei', 'juni', 'juli', 'agustus', 'september', 'oktober', 'november', 'desember', # bulan
        'senin', 'selasa', 'rabu', 'kamis', 'jumat', 'sabtu', 'minggu', # hari
         # Kata dari kode dashboard sebelumnya
        'yang', 'itu', 'dan', 'di', 'ke', 'dari', 'pada', 'untuk', 'oleh', 'dengan', 'tri',
        'saat', 'kemarin', 'nanti', 'ada', 'adalah', 'baik', 'buruk', 'dll',
        'saya', 'kamu', 'dia', 'mereka', 'kita', 'kami', 'anda', 'juga',
        'ini', 'itu', 'loh', 'ga', 'gak', 'enggak', 'nggak',
        'tapi', 'namun', 'atau', 'ataupun', 'sebab', 'karena', 'jika', 'kalau',
        'supaya', 'biar', 'agar', 'ketika', 'setelah', 'sebelum', 'sampai', 'hingga',
        'semua', 'beberapa', 'banyak', 'sedikit', 'orang', 'rumah', 'kota',
        'hal', 'masalah', 'sesuatu', 'merupakan', 'menjadi', 'terjadi',
        'berada', 'sedang', 'telah', 'ya', 'ngga', 'bekasi', 'walikota',
        'wali', 'pak', 'daerah', 'bikin', 'tolong', 'lg', 'udah', 'org', 'semoga', 'klo',
        'jgn', 'udh', 'dah', 'karna', 'br', 'gk', 'sy', 'aq',
        'banget', 'kali', 'aja', 'saja', 'pula',
        'belum', 'sudah', 'akan', 'selalu', 'sering', 'kadang', 'mungkin',
        'kenapa', 'gimana', 'bagaimana', 'apa', 'siapa', 'kapan', 'mana',
        'pas', 'ayo', 'mari',
        'bgt', 'tdk', 'gaes', 'guys', 'mntp', 'mantap', 'keren',
        'partai', 'politik', 'pemerintah', 'dpr', 'presiden', 'pilpres', 'pemilu', 'with'
    }
    stop_words_final = default_stopwords_id.union(custom_stopwords_list).union(dynamic_emoji_stopwords)
    st.sidebar.caption(f"Total stopwords (NLTK, kustom, emoji dinamis): {len(stop_words_final)}")

# --- EDA ---
    with st.expander("📊 Eksplorasi Hasil Analisis Sentimen", expanded=True):
        # (Kode EDA dari dashboard sebelumnya, disesuaikan untuk menggunakan df_dashboard dan nama kolom yang benar)
        # Contoh:
        col1_eda, col2_eda = st.columns(2)
        plot_height = 400
        margin_bawah_plot = 80
        
        with col1_eda:
            st.subheader(f"📊 Sebaran Jumlah Sentimen")
            fig_sent = px.histogram(df_dashboard, x=SENTIMENT_COLUMN, color=SENTIMENT_COLUMN,
                                    color_discrete_map={"positif":"#77DD77", "negatif":"#FF6961", "netral":"#AEC6CF"},
                                    text_auto='.2s')
            fig_sent.update_layout(
                height=plot_height,
                margin=dict(l=20, r=20, t=50, b=margin_bawah_plot) # l, r, t adalah contoh, fokus pada 'b'
            )
            fig_sent.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
            st.plotly_chart(fig_sent, use_container_width=True)
        with col2_eda:
            st.subheader(f"👍 Total Jumlah Like Komentar") # Atau st.markdown untuk judul yang lebih kecil
            like_avg = df_dashboard.groupby(SENTIMENT_COLUMN)[LIKE_COUNT_COLUMN].sum().reset_index().sort_values(by=LIKE_COUNT_COLUMN, ascending=False)
            fig_bar_like_sent = px.bar(like_avg, x=SENTIMENT_COLUMN, y=LIKE_COUNT_COLUMN, text_auto='.2s',
                                       color=SENTIMENT_COLUMN, color_discrete_map={"positif":"#77DD77", "negatif":"#FF6961", "netral":"#AEC6CF"}) # kode px.bar Anda untuk like_avg
            fig_bar_like_sent.update_layout(
                height=plot_height, # Menggunakan variabel tinggi yang sama (misal: 400)
                margin=dict(l=20, r=20, t=50, b=margin_bawah_plot) # Menggunakan variabel margin yang sama (misal: b=80)
            )
            fig_bar_like_sent.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)

            st.plotly_chart(fig_bar_like_sent, use_container_width=True)

        
        st.subheader("📅 Tren Sentimen Seiring Waktu")
        df_non_na_date_dashboard = df_dashboard.dropna(subset=['date'])
        if not df_non_na_date_dashboard.empty:
            time_series = df_non_na_date_dashboard.groupby(["date", SENTIMENT_COLUMN]).size().reset_index(name="jumlah_komentar")
            fig_line_trend = px.line(time_series, x="date", y="jumlah_komentar", color=SENTIMENT_COLUMN, markers=True,
                                     color_discrete_map={"positif":"#77DD77", "negatif":"#FF6961", "netral":"#AEC6CF"})
            st.plotly_chart(fig_line_trend, use_container_width=True)
        else:
            st.info("Tidak ada data waktu yang valid untuk tren sentimen.")

        # WordCloud dan Top 20 Kata Komentar Negatif
        st.subheader("💬 Analisis Teks Komentar Negatif (dari Teks Bersih)")
        df_negatif_eda = df_dashboard[df_dashboard[SENTIMENT_COLUMN] == "negatif"]

        if not df_negatif_eda.empty:
            # Teks untuk WordCloud dan Counter sudah dibersihkan oleh light_preprocess_text
            # Kita hanya perlu menerapkan stopwords tambahan dari stop_words_final
            
            all_negative_words_for_counter = []
            full_negative_text_for_wc = ""

            temp_cleaned_negative_texts = []
            for comment_text in df_negatif_eda[TEXT_COLUMN_FOR_ANALYSIS].astype(str).dropna():
                # Teks sudah melalui light_preprocess_text. Sekarang filter dengan stop_words_final.
                words = [word for word in comment_text.split() if word not in stop_words_final and len(word) > 1]
                all_negative_words_for_counter.extend(words)
                temp_cleaned_negative_texts.append(" ".join(words))
            full_negative_text_for_wc = " ".join(temp_cleaned_negative_texts)

            st.markdown("☁️ **WordCloud Komentar Negatif**")
            if full_negative_text_for_wc.strip():
                try:
                    wc = WordCloud(width=800, height=400, background_color='white', collocations=False).generate(full_negative_text_for_wc)
                    fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
                    ax_wc.imshow(wc, interpolation='bilinear')
                    ax_wc.axis("off")
                    st.pyplot(fig_wc)
                except Exception as e_wc:
                    st.error(f"Gagal membuat WordCloud: {e_wc}")
            else:
                st.info("Tidak ada kata tersisa untuk WordCloud komentar negatif setelah filtering stopwords.")

            st.markdown("🔝 **20 Kata Teratas Komentar Negatif**")
            if all_negative_words_for_counter:
                word_counts_negative = collections.Counter(all_negative_words_for_counter)
                most_common_words_negative = word_counts_negative.most_common(10)
                if most_common_words_negative:
                    df_most_common_negative = pd.DataFrame(most_common_words_negative, columns=['Kata', 'Frekuensi'])
                    fig_top_words_neg = px.bar(df_most_common_negative, x='Frekuensi', y='Kata', orientation='h', text_auto=True)
                    fig_top_words_neg.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_top_words_neg, use_container_width=True)
                else:
                    st.info("Tidak cukup kata untuk 20 kata teratas komentar negatif.")
            else:
                st.info("Tidak ada teks negatif tersisa untuk analisis kata teratas.")
        else:
            st.info("Tidak ada komentar negatif untuk dianalisis dalam EDA.")
    # --- BERTopic Analysis ---
    # (Kode BERTopic dari dashboard sebelumnya, disesuaikan)
    # Termasuk opsi load/train model BERTopic, coherence score, visualisasi topik
    # Pastikan menggunakan TEXT_COLUMN_FOR_ANALYSIS untuk input ke BERTopic
    # dan df_dashboard[df_dashboard[SENTIMENT_COLUMN] == "negatif"]

    st.header("🔬 Analisis Topik Komentar Negatif dengan BERTopic")
    df_negatif_bertopic = df_dashboard[df_dashboard[SENTIMENT_COLUMN] == "negatif"].copy()

    MODEL_BERTOPIC_PATH = "my_trained_bertopic_model_youtube" # Path untuk menyimpan/load model BERTopic

    @st.cache_resource
    def load_bertopic_model_from_path(path):
        st.info(f"Mencoba memuat model BERTopic dari: {path}")
        # Cek apakah path direktori ada (BERTopic.save() biasanya membuat direktori)
        if os.path.isdir(path):
            try:
                loaded_model = BERTopic.load(path)
                st.success(f"✅ Model BERTopic berhasil dimuat dari '{path}'!")
                return loaded_model
            except Exception as e:
                st.error(f"❌ Gagal memuat model BERTopic dari '{path}': {e}")
                return None
        else:
            st.warning(f"⚠️ Direktori model BERTopic di '{path}' tidak ditemukan.")
            return None

    model_action_bertopic = st.radio(
        "Pilih tindakan untuk model BERTopic:",
        ('Latih Model Baru', 'Muat Model BERTopic yang Sudah Ada'),
        horizontal=True, key="bertopic_action"
    )

    active_topic_model = None # Inisialisasi

    if model_action_bertopic == 'Muat Model BERTopic yang Sudah Ada':
        active_topic_model = load_bertopic_model_from_path(MODEL_BERTOPIC_PATH)
        if active_topic_model is None:
            st.info("Tidak dapat memuat model BERTopic. Silakan latih model baru atau periksa path.")

    if model_action_bertopic == 'Latih Model Baru' or active_topic_model is None:
        if model_action_bertopic == 'Latih Model Baru':
            st.info("Opsi 'Latih Model Baru BERTopic' dipilih.")
        
        if len(df_negatif_bertopic) < 5: # BERTopic butuh cukup dokumen
            st.warning("⚠️ Jumlah komentar negatif terlalu sedikit (kurang dari 5) untuk analisis topik BERTopic yang bermakna.")
        else:
            train_button_label_bertopic = "🚀 Latih Model BERTopic Baru Sekarang"
            if active_topic_model is None and model_action_bertopic == 'Muat Model BERTopic yang Sudah Ada':
                train_button_label_bertopic = "🔄 Gagal Memuat, Latih Model BERTopic Baru?"

            if st.button(train_button_label_bertopic, key="train_bertopic_btn"):
                with st.spinner("Melatih model BERTopic pada komentar negatif (teks bersih)... ⏳"):
                    # Input untuk BERTopic adalah teks yang sudah melalui light_preprocess_text
                    docs_negatif_bertopic = df_negatif_bertopic[TEXT_COLUMN_FOR_ANALYSIS].astype(str).dropna().tolist()
                    
                    # Hapus string kosong yang mungkin muncul setelah preprocessing (meskipun light_preprocess_text harusnya .strip())
                    processed_docs_negatif_bertopic = [doc for doc in docs_negatif_bertopic if doc]

                    if not processed_docs_negatif_bertopic or len(processed_docs_negatif_bertopic) < 5:
                        st.error("Tidak ada dokumen negatif yang valid tersisa untuk BERTopic atau jumlahnya terlalu sedikit.")
                    else:
                        try:
                            # Opsi: Gunakan vectorizer dengan stop_words_final
                            from sklearn.feature_extraction.text import CountVectorizer
                            vectorizer_model_custom = CountVectorizer(stop_words=list(stop_words_final))
                            
                            temp_topic_model_bt = BERTopic(
                                language="multilingual", # Lebih aman jika ada campuran atau jika vectorizer kustom
                                verbose=True, 
                                min_topic_size=st.sidebar.slider("Ukuran Topik Minimum (BERTopic)", 2, 20, 3), 
                                nr_topics=None, # Atau "auto" atau angka int
                                vectorizer_model=vectorizer_model_custom # Menggunakan stopwords kustom kita
                            )
                            topics_bt, probs_bt = temp_topic_model_bt.fit_transform(processed_docs_negatif_bertopic)
                            active_topic_model = temp_topic_model_bt
                            st.success("✅ Model BERTopic berhasil dilatih!")
                            
                            if st.button("💾 Simpan Model BERTopic yang Baru Dilatih Ini?", key="save_bertopic_btn"):
                                try:
                                    active_topic_model.save(MODEL_BERTOPIC_PATH, serialization="pickle") # pickle lebih portabel kadang
                                    st.success(f"Model BERTopic disimpan ke '{MODEL_BERTOPIC_PATH}'.")
                                except Exception as e_save_bt:
                                    st.error(f"Gagal menyimpan model BERTopic: {e_save_bt}")
                        except Exception as e_bt_train:
                            st.error(f"❌ Error saat melatih BERTopic: {e_bt_train}")
                            st.exception(e_bt_train)
    
    if active_topic_model:
        st.subheader("Analisis Menggunakan Model BERTopic Aktif")
        # Ambil teks yang SAMA dengan yang mungkin digunakan untuk melatih / atau untuk transform
        docs_for_bertopic_analysis = df_negatif_bertopic[TEXT_COLUMN_FOR_ANALYSIS].astype(str).dropna().tolist()
        processed_docs_for_bertopic_analysis = [doc for doc in docs_for_bertopic_analysis if doc]

        if not processed_docs_for_bertopic_analysis:
            st.warning("Tidak ada dokumen negatif bersih yang valid untuk dianalisis dengan model BERTopic.")
        else:
            # --- Perhitungan Coherence Score ---
            # (Kode Coherence dari dashboard sebelumnya, pastikan menggunakan active_topic_model dan processed_docs_for_bertopic_analysis)
            from gensim.corpora.dictionary import Dictionary
            from gensim.models.coherencemodel import CoherenceModel
            st.subheader("💯 Coherence Score Topik BERTopic")
            try:
                tokenized_docs_for_coh_bt = [doc.split() for doc in processed_docs_for_bertopic_analysis]
                dictionary_gensim_bt = Dictionary(tokenized_docs_for_coh_bt)
                
                keywords_per_topic_gensim_bt = []
                valid_topic_ids_bt = sorted([tid for tid in active_topic_model.get_topics().keys() if tid != -1])

                for topic_id_bt in valid_topic_ids_bt:
                    topic_words_scores_bt = active_topic_model.get_topic(topic_id_bt)
                    if topic_words_scores_bt:
                        keywords_per_topic_gensim_bt.append([word for word, score in topic_words_scores_bt[:10]])

                if keywords_per_topic_gensim_bt:
                    cm_cv_bt = CoherenceModel(topics=keywords_per_topic_gensim_bt, texts=tokenized_docs_for_coh_bt, dictionary=dictionary_gensim_bt, coherence='c_v')
                    st.metric(label="Coherence (c_v - manual Gensim)", value=f"{cm_cv_bt.get_coherence():.4f}")
                    cm_umass_bt = CoherenceModel(topics=keywords_per_topic_gensim_bt, texts=tokenized_docs_for_coh_bt, dictionary=dictionary_gensim_bt, coherence='u_mass')
                    st.metric(label="Coherence (u_mass - manual Gensim)", value=f"{cm_umass_bt.get_coherence():.4f}")
                else:
                    st.info("Tidak ada topik yang diekstrak untuk perhitungan koherensi BERTopic.")
            except Exception as e_coh_bt:
                st.error(f"Gagal menghitung Coherence Score BERTopic: {e_coh_bt}")

            # --- Tampilkan Informasi Topik BERTopic ---
            df_topic_info_bt = active_topic_model.get_topic_info()
            # Filter outlier topic (-1) jika tidak ingin ditampilkan dan ambil top 15 topik
            df_topic_info_filtered_bt = df_topic_info_bt[df_topic_info_bt["Topic"] != -1].head(10)

            if not df_topic_info_filtered_bt.empty:
                st.subheader("📈 Topik Komentar Negatif (BERTopic)")
                # Membuat kolom 'DisplayName' yang lebih ramah dibaca dari kolom 'Name' BERTopic
                # Menggunakan .loc untuk menghindari SettingWithCopyWarning
                df_topic_info_filtered_bt.loc[:, "DisplayName"] = df_topic_info_filtered_bt["Name"].apply(
                    lambda x: x[x.find("_") + 1:].replace("_", " ") if isinstance(x, str) else "N/A"
                )
                
                fig_topic_bar_bt = px.bar(df_topic_info_filtered_bt, 
                                          x="DisplayName", 
                                          y="Count", 
                                          text_auto=True,
                                          title="Distribusi Komentar per Topik (Top 15)",
                                          labels={"DisplayName": "Representasi Topik", "Count": "Jumlah Komentar"})
                fig_topic_bar_bt.update_layout(xaxis_title="Representasi Topik", yaxis_title="Jumlah Komentar")
                st.plotly_chart(fig_topic_bar_bt, use_container_width=True)

                st.subheader("🧠 Rangkuman, Contoh Komentar & WordCloud per Topik (BERTopic)")
                for _, row_bt in df_topic_info_filtered_bt.iterrows():
                    topic_id_bt_disp = row_bt["Topic"]
                    label_bt = row_bt["DisplayName"]
                    count_bt = row_bt["Count"]
                    
                    keywords_scores_bt = active_topic_model.get_topic(topic_id_bt_disp)
                    
                    if keywords_scores_bt is None: 
                        st.caption(f"Tidak ada kata kunci untuk Topik #{topic_id_bt_disp}: {label_bt}")
                        continue
                        
                    keywords_only_bt = [word for word, score in keywords_scores_bt[:10]]

                    with st.expander(f"Topik #{topic_id_bt_disp}: {label_bt} (Jumlah: {count_bt})"):
                        st.markdown(f"**🔑 Kata Kunci Utama:** {', '.join(keywords_only_bt)}")
                        
                        # --- MENAMPILKAN CONTOH KOMENTAR ASLI REPRESENTATIF ---
                        try:
                            # Dapatkan dokumen representatif (ini adalah teks yang sudah dibersihkan)
                            rep_cleaned_docs_bt = active_topic_model.get_representative_docs(topic_id_bt_disp)

                            if rep_cleaned_docs_bt:
                                st.markdown("**💬 Contoh Komentar ASLI Representatif:**") # Judul diubah
                                displayed_count = 0
                                
                                # df_negatif_bertopic adalah DataFrame yang berisi komentar negatif
                                # dengan kolom RAW_TEXT_COLUMN (teks asli) dan TEXT_COLUMN_FOR_ANALYSIS (teks bersih)
                                # Pastikan df_negatif_bertopic, RAW_TEXT_COLUMN, dan TEXT_COLUMN_FOR_ANALYSIS
                                # terdefinisi dengan benar dan tersedia dalam scope ini.

                                for cleaned_doc_sample in rep_cleaned_docs_bt:
                                    if displayed_count >= 3: # Batasi hingga 3 contoh
                                        break
                                    
                                    # Cari baris di df_negatif_bertopic yang kolom teks bersihnya cocok dengan sampel
                                    # Ini mengasumsikan bahwa teks bersih unik atau kita ambil yang pertama cocok.
                                    matching_rows = df_negatif_bertopic[df_negatif_bertopic[TEXT_COLUMN_FOR_ANALYSIS] == cleaned_doc_sample]
                                    
                                    if not matching_rows.empty:
                                        # Ambil teks asli dari baris pertama yang cocok
                                        original_comment_sample = matching_rows.iloc[0][RAW_TEXT_COLUMN]
                                        st.markdown(f"> _{original_comment_sample}_")
                                        displayed_count += 1
                                    else:
                                        # Fallback jika karena alasan tertentu teks bersih tidak ditemukan di df_negatif_bertopic
                                        # Ini seharusnya jarang terjadi jika data konsisten.
                                        st.markdown(f"> _(Mapping ke teks asli gagal. Teks bersih: {cleaned_doc_sample})_")
                                        displayed_count += 1 # Tetap hitung sebagai sudah ditampilkan
                            else:
                                st.markdown("_Tidak ada contoh komentar representatif untuk topik ini._")
                        except Exception as e_rep_doc:
                            st.caption(f"Info: Tidak bisa mengambil/memetakan contoh dokumen representatif untuk Topik #{topic_id_bt_disp} ({e_rep_doc})")
                        # --- AKHIR BAGIAN CONTOH KOMENTAR ASLI ---
                        

                        st.markdown("**☁️ WordCloud Kata Kunci Topik** (setelah filter stopwords):")
                        word_freq_wc_topic_bt = {
                            word: score for word, score in keywords_scores_bt 
                            if word.lower() not in stop_words_final and len(word) > 1 # Pastikan stop_words_final terdefinisi
                        }
                        
                        if word_freq_wc_topic_bt:
                            try:
                                wc_topic_bt = WordCloud(
                                    width=600, height=300, 
                                    background_color='white', 
                                    collocations=False
                                ).generate_from_frequencies(word_freq_wc_topic_bt)
                                
                                fig_wc_topic_bt, ax_wc_topic_bt = plt.subplots(figsize=(8,4)) # Ganti nama variabel agar unik
                                ax_wc_topic_bt.imshow(wc_topic_bt, interpolation='bilinear')
                                ax_wc_topic_bt.axis("off")
                                st.pyplot(fig_wc_topic_bt) # Tampilkan plot yang benar
                            except Exception as e_wc_topic:
                                st.error(f"Gagal membuat WordCloud untuk Topik #{topic_id_bt_disp}: {e_wc_topic}")
                        else:
                            st.info("Tidak ada kata kunci tersisa untuk WordCloud topik ini setelah filter stopwords.")
            else:
                # Pesan ini hanya ditampilkan jika analisis sudah coba dijalankan dan tidak ada topik valid
                if st.session_state.get('analyze_button_pressed_once', False): 
                    st.info("Tidak ada topik signifikan yang ditemukan oleh BERTopic (selain outlier) untuk ditampilkan detailnya.")
            
            # --- Tombol Download Data dengan Topik BERTopic ---
            # Pastikan processed_docs_for_bertopic_analysis dan df_negatif_bertopic terdefinisi dengan benar sebelum blok ini
            if 'processed_docs_for_bertopic_analysis' in locals() and 'df_negatif_bertopic' in locals():
                try:
                    # Dapatkan topik untuk dokumen yang dianalisis (jika belum ada)
                    # Jika active_topic_model adalah hasil fit_transform, topics_bt sudah ada.
                    # Jika dimuat, atau jika ingin memastikan, jalankan transform.
                    # Untuk konsistensi, kita asumsikan kita perlu .transform() pada data yang relevan.
                    current_topics_bt, _ = active_topic_model.transform(processed_docs_for_bertopic_analysis)
                    
                    # Pastikan panjangnya cocok sebelum menambahkan kolom baru
                    if len(df_negatif_bertopic) == len(current_topics_bt):
                        df_to_download_bt = df_negatif_bertopic.copy() # Buat salinan untuk diubah
                        df_to_download_bt['BERTopic_ID'] = current_topics_bt
                        
                        # Siapkan data CSV untuk diunduh
                        @st.cache_data # Cache data CSV agar tidak dibuat ulang terus-menerus
                        def convert_df_to_csv(df_input):
                            return df_input.to_csv(index=False).encode('utf-8')

                        csv_data_bt_dl = convert_df_to_csv(df_to_download_bt)
                        
                        st.download_button(
                            label="💾 Unduh Data Negatif dengan ID Topik BERTopic", 
                            data=csv_data_bt_dl,
                            file_name='komentar_negatif_youtube_bertopic.csv', 
                            mime='text/csv',
                            key='download_bertopic_csv' # Tambahkan key unik
                        )
                    else:
                        st.warning(f"Gagal mencocokkan topik dengan DataFrame asli untuk unduhan (perbedaan panjang: {len(df_negatif_bertopic)} vs {len(current_topics_bt)}).")
                except Exception as e_transform_dl_bt:
                    st.error(f"Gagal melakukan transformasi BERTopic untuk unduhan: {e_transform_dl_bt}")
            else:
                st.caption("Data yang diperlukan untuk tombol unduh BERTopic tidak tersedia.")

        # ... (else: jika active_topic_model is None) ...
        # Anda bisa menambahkan pesan di sini jika model BERTopic tidak aktif (misalnya, gagal dimuat dan tidak dilatih ulang)
        # else:
        #     if st.session_state.get('analyze_button_pressed_once', False): # Hanya tampilkan jika analisis sudah coba dijalankan
        #         st.warning("Model BERTopic tidak aktif. Tidak ada analisis topik yang dapat ditampilkan.")
elif st.session_state.processed_df is not None and st.session_state.processed_df.empty:
    st.info("Tidak ada data komentar yang diproses untuk ditampilkan di dashboard.")
elif st.session_state.analyze_button_pressed_once and (st.session_state.processed_df is None or st.session_state.processed_df.empty):
    # Pesan ini muncul jika tombol ditekan, statistik video mungkin tampil, tapi tidak ada komentar yang diproses
    st.warning("Statistik video mungkin telah ditampilkan, tetapi tidak ada data komentar yang berhasil diproses untuk analisis lebih lanjut.")
elif not st.session_state.analyze_button_pressed_once:
    st.info("👋 Selamat datang! Masukkan URL YouTube dan klik 'Ambil & Analisis Komentar' untuk memulai.")

# --- BAGIAN 5: Opsional - Simpan ke Google Sheets ---
if st.session_state.processed_df is not None and not st.session_state.processed_df.empty:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Simpan Hasil ke Google Sheets")
    gsheet_url_input = st.sidebar.text_input("URL Google Sheet Tujuan:", help="Contoh: https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID")
    gsheet_sheet_name_input = st.sidebar.text_input("Nama Worksheet (Sheet) Tujuan:", value="Sheet1_HasilAnalisis")

    if st.sidebar.button("💾 Simpan ke Google Sheets"):
        if gsheet_url_input and gsheet_sheet_name_input:
            creds_gsheets = get_gsheets_credentials()
            if creds_gsheets:
                try:
                    client_gsheets = gspread.authorize(creds_gsheets)
                    sheet_gs = client_gsheets.open_by_url(gsheet_url_input)
                    
                    # Coba dapatkan worksheet, buat jika tidak ada
                    try:
                        worksheet_gs = sheet_gs.worksheet(gsheet_sheet_name_input)
                    except gspread.exceptions.WorksheetNotFound:
                        st.info(f"Worksheet '{gsheet_sheet_name_input}' tidak ditemukan, membuat worksheet baru...")
                        worksheet_gs = sheet_gs.add_worksheet(title=gsheet_sheet_name_input, rows="1", cols="1") # Buat dg ukuran minimal

                    worksheet_gs.clear() # Hapus konten lama
                    # Konversi semua kolom ke string untuk menghindari error gspread dengan tipe data campuran
                    df_to_upload = st.session_state.processed_df.astype(str)
                    worksheet_gs.update([df_to_upload.columns.tolist()] + df_to_upload.values.tolist())
                    st.sidebar.success(f"Data berhasil disimpan ke Google Sheet '{gsheet_sheet_name_input}'!")
                except Exception as e_gsheets:
                    st.sidebar.error(f"Gagal menyimpan ke Google Sheets: {e_gsheets}")
            else:
                st.sidebar.error("Kredensial Google Sheets tidak valid.")
        else:
            st.sidebar.warning("Harap masukkan URL Google Sheet dan nama Worksheet.")

st.markdown("---")
st.caption("Dashboard Analisis Komentar YouTube | Dibuat dengan Model Fine-tuned & BERTopic")
# keep awake Sun Jun 15 18:32:40 UTC 2025
# keep awake Mon Jun 16 02:08:59 UTC 2025
# keep awake Mon Jun 16 06:38:44 UTC 2025
# keep awake Mon Jun 16 12:54:27 UTC 2025
# keep awake Mon Jun 16 18:35:44 UTC 2025
# keep awake Tue Jun 17 02:05:32 UTC 2025
# keep awake Tue Jun 17 06:36:52 UTC 2025
# keep awake Tue Jun 17 12:54:03 UTC 2025
# keep awake Tue Jun 17 18:35:59 UTC 2025
# keep awake Wed Jun 18 02:04:46 UTC 2025
# keep awake Wed Jun 18 06:36:55 UTC 2025
# keep awake Wed Jun 18 12:54:01 UTC 2025
# keep awake Wed Jun 18 18:35:52 UTC 2025
# keep awake Thu Jun 19 02:05:34 UTC 2025
# keep awake Thu Jun 19 06:36:48 UTC 2025
# keep awake Thu Jun 19 12:52:57 UTC 2025
# keep awake Thu Jun 19 18:34:25 UTC 2025
# keep awake Fri Jun 20 02:04:33 UTC 2025
# keep awake Fri Jun 20 06:36:40 UTC 2025
# keep awake Fri Jun 20 12:52:37 UTC 2025
# keep awake Fri Jun 20 18:35:07 UTC 2025
# keep awake Sat Jun 21 02:02:22 UTC 2025
# keep awake Sat Jun 21 06:34:04 UTC 2025
# keep awake Sat Jun 21 12:48:27 UTC 2025
# keep awake Sat Jun 21 18:31:56 UTC 2025
# keep awake Sun Jun 22 02:20:23 UTC 2025
# keep awake Sun Jun 22 06:34:25 UTC 2025
# keep awake Sun Jun 22 12:48:12 UTC 2025
# keep awake Sun Jun 22 18:32:44 UTC 2025
# keep awake Mon Jun 23 02:18:32 UTC 2025
# keep awake Mon Jun 23 06:38:54 UTC 2025
# keep awake Mon Jun 23 12:54:47 UTC 2025
# keep awake Mon Jun 23 18:36:07 UTC 2025
# keep awake Tue Jun 24 02:06:23 UTC 2025
# keep awake Tue Jun 24 06:37:57 UTC 2025
# keep awake Tue Jun 24 12:54:05 UTC 2025
# keep awake Tue Jun 24 18:35:56 UTC 2025
# keep awake Wed Jun 25 02:06:39 UTC 2025
# keep awake Wed Jun 25 06:37:56 UTC 2025
# keep awake Wed Jun 25 12:54:49 UTC 2025
# keep awake Wed Jun 25 18:36:45 UTC 2025
# keep awake Thu Jun 26 02:05:36 UTC 2025
# keep awake Thu Jun 26 06:38:00 UTC 2025
# keep awake Thu Jun 26 12:54:18 UTC 2025
# keep awake Thu Jun 26 18:35:52 UTC 2025
# keep awake Fri Jun 27 02:06:28 UTC 2025
# keep awake Fri Jun 27 06:37:27 UTC 2025
# keep awake Fri Jun 27 12:52:37 UTC 2025
# keep awake Fri Jun 27 18:35:09 UTC 2025
# keep awake Sat Jun 28 02:02:01 UTC 2025
# keep awake Sat Jun 28 06:34:07 UTC 2025
# keep awake Sat Jun 28 12:48:40 UTC 2025
# keep awake Sat Jun 28 18:32:43 UTC 2025
# keep awake Sun Jun 29 02:22:53 UTC 2025
# keep awake Sun Jun 29 06:35:24 UTC 2025
# keep awake Sun Jun 29 12:49:27 UTC 2025
# keep awake Sun Jun 29 18:32:49 UTC 2025
# keep awake Mon Jun 30 02:11:35 UTC 2025
# keep awake Mon Jun 30 06:38:43 UTC 2025
# keep awake Mon Jun 30 12:53:59 UTC 2025
# keep awake Mon Jun 30 18:35:54 UTC 2025
# keep awake Tue Jul  1 02:23:54 UTC 2025
# keep awake Tue Jul  1 06:38:24 UTC 2025
# keep awake Tue Jul  1 12:54:09 UTC 2025
# keep awake Tue Jul  1 18:35:21 UTC 2025
# keep awake Wed Jul  2 02:05:58 UTC 2025
# keep awake Wed Jul  2 06:37:57 UTC 2025
# keep awake Wed Jul  2 12:53:37 UTC 2025
# keep awake Wed Jul  2 18:35:59 UTC 2025
# keep awake Thu Jul  3 02:06:30 UTC 2025
# keep awake Thu Jul  3 06:38:08 UTC 2025
# keep awake Thu Jul  3 12:52:04 UTC 2025
# keep awake Thu Jul  3 18:35:37 UTC 2025
# keep awake Fri Jul  4 02:05:41 UTC 2025
# keep awake Fri Jul  4 06:38:13 UTC 2025
# keep awake Fri Jul  4 12:52:33 UTC 2025
# keep awake Fri Jul  4 18:34:24 UTC 2025
# keep awake Sat Jul  5 02:01:42 UTC 2025
# keep awake Sat Jul  5 06:34:26 UTC 2025
# keep awake Sat Jul  5 12:48:54 UTC 2025
# keep awake Sat Jul  5 18:32:17 UTC 2025
# keep awake Sun Jul  6 02:21:16 UTC 2025
# keep awake Sun Jul  6 06:35:09 UTC 2025
# keep awake Sun Jul  6 12:49:28 UTC 2025
# keep awake Sun Jul  6 18:33:22 UTC 2025
# keep awake Mon Jul  7 02:14:07 UTC 2025
# keep awake Mon Jul  7 06:39:18 UTC 2025
# keep awake Mon Jul  7 12:54:23 UTC 2025
# keep awake Mon Jul  7 18:36:24 UTC 2025
# keep awake Tue Jul  8 02:07:06 UTC 2025
# keep awake Tue Jul  8 06:38:35 UTC 2025
# keep awake Tue Jul  8 12:54:18 UTC 2025
# keep awake Tue Jul  8 18:36:46 UTC 2025
# keep awake Wed Jul  9 02:08:27 UTC 2025
# keep awake Wed Jul  9 06:38:23 UTC 2025
# keep awake Wed Jul  9 12:54:21 UTC 2025
# keep awake Wed Jul  9 18:37:22 UTC 2025
# keep awake Thu Jul 10 02:08:45 UTC 2025
# keep awake Thu Jul 10 06:38:52 UTC 2025
# keep awake Thu Jul 10 12:54:59 UTC 2025
# keep awake Thu Jul 10 18:36:33 UTC 2025
# keep awake Fri Jul 11 02:11:20 UTC 2025
# keep awake Fri Jul 11 06:38:38 UTC 2025
# keep awake Fri Jul 11 12:53:46 UTC 2025
# keep awake Fri Jul 11 18:36:13 UTC 2025
# keep awake Sat Jul 12 02:11:58 UTC 2025
# keep awake Sat Jul 12 06:35:52 UTC 2025
# keep awake Sat Jul 12 12:50:01 UTC 2025
# keep awake Sat Jul 12 18:34:02 UTC 2025
# keep awake Sun Jul 13 02:26:06 UTC 2025
# keep awake Sun Jul 13 06:36:25 UTC 2025
# keep awake Sun Jul 13 12:50:39 UTC 2025
# keep awake Sun Jul 13 18:34:03 UTC 2025
# keep awake Mon Jul 14 02:21:40 UTC 2025
# keep awake Mon Jul 14 06:40:30 UTC 2025
# keep awake Mon Jul 14 12:55:43 UTC 2025
# keep awake Mon Jul 14 18:39:39 UTC 2025
# keep awake Tue Jul 15 02:21:15 UTC 2025
# keep awake Tue Jul 15 06:40:02 UTC 2025
# keep awake Tue Jul 15 12:56:09 UTC 2025
# keep awake Tue Jul 15 18:39:02 UTC 2025
# keep awake Wed Jul 16 02:18:00 UTC 2025
# keep awake Wed Jul 16 06:39:53 UTC 2025
# keep awake Wed Jul 16 12:56:44 UTC 2025
# keep awake Wed Jul 16 18:38:27 UTC 2025
# keep awake Thu Jul 17 02:18:36 UTC 2025
# keep awake Thu Jul 17 06:39:34 UTC 2025
# keep awake Thu Jul 17 12:56:05 UTC 2025
# keep awake Thu Jul 17 18:39:15 UTC 2025
# keep awake Fri Jul 18 02:20:07 UTC 2025
# keep awake Fri Jul 18 06:39:55 UTC 2025
# keep awake Fri Jul 18 12:56:59 UTC 2025
# keep awake Fri Jul 18 18:37:21 UTC 2025
# keep awake Sat Jul 19 02:08:14 UTC 2025
# keep awake Sat Jul 19 06:36:50 UTC 2025
# keep awake Sat Jul 19 12:51:25 UTC 2025
# keep awake Sat Jul 19 18:34:53 UTC 2025
# keep awake Sun Jul 20 02:28:25 UTC 2025
# keep awake Sun Jul 20 06:36:45 UTC 2025
# keep awake Sun Jul 20 12:51:42 UTC 2025
# keep awake Sun Jul 20 18:35:03 UTC 2025
# keep awake Mon Jul 21 02:25:14 UTC 2025
# keep awake Mon Jul 21 06:40:16 UTC 2025
# keep awake Mon Jul 21 12:57:28 UTC 2025
# keep awake Mon Jul 21 18:39:18 UTC 2025
# keep awake Tue Jul 22 02:19:23 UTC 2025
# keep awake Tue Jul 22 06:40:06 UTC 2025
# keep awake Tue Jul 22 12:57:32 UTC 2025
# keep awake Tue Jul 22 18:39:22 UTC 2025
# keep awake Wed Jul 23 02:20:35 UTC 2025
# keep awake Wed Jul 23 06:40:25 UTC 2025
# keep awake Wed Jul 23 12:57:51 UTC 2025
# keep awake Wed Jul 23 18:38:58 UTC 2025
# keep awake Thu Jul 24 02:19:37 UTC 2025
# keep awake Thu Jul 24 06:40:29 UTC 2025
# keep awake Thu Jul 24 12:57:20 UTC 2025
# keep awake Thu Jul 24 18:38:57 UTC 2025
# keep awake Fri Jul 25 02:19:11 UTC 2025
# keep awake Fri Jul 25 06:40:39 UTC 2025
# keep awake Fri Jul 25 12:56:12 UTC 2025
# keep awake Fri Jul 25 18:39:09 UTC 2025
# keep awake Sat Jul 26 02:09:45 UTC 2025
# keep awake Sat Jul 26 06:37:33 UTC 2025
# keep awake Sat Jul 26 12:52:00 UTC 2025
# keep awake Sat Jul 26 18:35:44 UTC 2025
# keep awake Sun Jul 27 02:29:27 UTC 2025
# keep awake Sun Jul 27 06:38:21 UTC 2025
# keep awake Sun Jul 27 12:53:23 UTC 2025
# keep awake Sun Jul 27 18:36:18 UTC 2025
# keep awake Mon Jul 28 02:26:18 UTC 2025
# keep awake Mon Jul 28 06:43:13 UTC 2025
# keep awake Mon Jul 28 12:58:37 UTC 2025
# keep awake Mon Jul 28 18:40:43 UTC 2025
# keep awake Tue Jul 29 02:31:17 UTC 2025
# keep awake Tue Jul 29 06:40:49 UTC 2025
# keep awake Tue Jul 29 12:59:16 UTC 2025
# keep awake Tue Jul 29 18:41:01 UTC 2025
# keep awake Wed Jul 30 02:22:44 UTC 2025
# keep awake Wed Jul 30 06:42:47 UTC 2025
# keep awake Wed Jul 30 12:59:05 UTC 2025
# keep awake Wed Jul 30 18:39:58 UTC 2025
# keep awake Thu Jul 31 02:22:00 UTC 2025
# keep awake Thu Jul 31 06:40:39 UTC 2025
# keep awake Thu Jul 31 12:57:22 UTC 2025
# keep awake Thu Jul 31 18:40:11 UTC 2025
# keep awake Fri Aug  1 02:33:14 UTC 2025
# keep awake Fri Aug  1 06:43:23 UTC 2025
# keep awake Fri Aug  1 12:58:01 UTC 2025
# keep awake Fri Aug  1 18:39:28 UTC 2025
# keep awake Sat Aug  2 02:10:25 UTC 2025
# keep awake Sat Aug  2 06:37:01 UTC 2025
# keep awake Sat Aug  2 12:52:41 UTC 2025
# keep awake Sat Aug  2 18:36:36 UTC 2025
# keep awake Sun Aug  3 02:31:49 UTC 2025
# keep awake Sun Aug  3 06:37:43 UTC 2025
# keep awake Sun Aug  3 12:55:41 UTC 2025
# keep awake Sun Aug  3 18:37:21 UTC 2025
# keep awake Mon Aug  4 02:31:22 UTC 2025
# keep awake Mon Aug  4 06:49:03 UTC 2025
# keep awake Mon Aug  4 13:00:06 UTC 2025
# keep awake Mon Aug  4 18:41:11 UTC 2025
# keep awake Tue Aug  5 02:25:01 UTC 2025
# keep awake Tue Aug  5 06:43:03 UTC 2025
# keep awake Tue Aug  5 13:00:13 UTC 2025
# keep awake Tue Aug  5 18:42:38 UTC 2025
# keep awake Wed Aug  6 02:23:27 UTC 2025
# keep awake Wed Aug  6 06:43:24 UTC 2025
# keep awake Wed Aug  6 12:59:56 UTC 2025
# keep awake Wed Aug  6 18:39:20 UTC 2025
# keep awake Thu Aug  7 02:23:36 UTC 2025
# keep awake Thu Aug  7 06:43:11 UTC 2025
# keep awake Thu Aug  7 12:59:27 UTC 2025
# keep awake Thu Aug  7 18:41:31 UTC 2025
# keep awake Fri Aug  8 02:23:27 UTC 2025
# keep awake Fri Aug  8 06:42:41 UTC 2025
# keep awake Fri Aug  8 12:58:18 UTC 2025
# keep awake Fri Aug  8 18:36:59 UTC 2025
# keep awake Sat Aug  9 02:05:40 UTC 2025
# keep awake Sat Aug  9 06:37:10 UTC 2025
# keep awake Sat Aug  9 12:51:51 UTC 2025
# keep awake Sat Aug  9 18:35:42 UTC 2025
# keep awake Sun Aug 10 02:27:15 UTC 2025
# keep awake Sun Aug 10 06:36:56 UTC 2025
# keep awake Sun Aug 10 12:52:25 UTC 2025
# keep awake Sun Aug 10 18:34:54 UTC 2025
# keep awake Mon Aug 11 02:23:12 UTC 2025
# keep awake Mon Aug 11 06:42:22 UTC 2025
# keep awake Mon Aug 11 12:57:38 UTC 2025
# keep awake Mon Aug 11 18:39:38 UTC 2025
# keep awake Tue Aug 12 02:05:03 UTC 2025
# keep awake Tue Aug 12 06:38:52 UTC 2025
# keep awake Tue Aug 12 12:54:49 UTC 2025
# keep awake Tue Aug 12 18:38:43 UTC 2025
# keep awake Wed Aug 13 02:06:50 UTC 2025
# keep awake Wed Aug 13 06:39:10 UTC 2025
# keep awake Wed Aug 13 12:55:45 UTC 2025
# keep awake Wed Aug 13 18:36:46 UTC 2025
# keep awake Thu Aug 14 02:07:25 UTC 2025
# keep awake Thu Aug 14 06:39:40 UTC 2025
# keep awake Thu Aug 14 12:56:25 UTC 2025
# keep awake Thu Aug 14 18:38:06 UTC 2025
# keep awake Fri Aug 15 02:08:03 UTC 2025
# keep awake Fri Aug 15 06:38:45 UTC 2025
# keep awake Fri Aug 15 12:52:23 UTC 2025
# keep awake Fri Aug 15 18:37:30 UTC 2025
# keep awake Sat Aug 16 02:02:03 UTC 2025
# keep awake Sat Aug 16 06:35:31 UTC 2025
# keep awake Sat Aug 16 12:50:06 UTC 2025
# keep awake Sat Aug 16 18:33:25 UTC 2025
# keep awake Sun Aug 17 02:19:49 UTC 2025
# keep awake Sun Aug 17 06:36:21 UTC 2025
# keep awake Sun Aug 17 12:50:46 UTC 2025
# keep awake Sun Aug 17 18:34:42 UTC 2025
# keep awake Mon Aug 18 02:20:11 UTC 2025
# keep awake Mon Aug 18 06:42:03 UTC 2025
# keep awake Mon Aug 18 12:56:34 UTC 2025
# keep awake Mon Aug 18 18:38:29 UTC 2025
# keep awake Tue Aug 19 02:00:56 UTC 2025
# keep awake Tue Aug 19 06:37:35 UTC 2025
# keep awake Tue Aug 19 12:51:02 UTC 2025
# keep awake Tue Aug 19 18:33:40 UTC 2025
# keep awake Wed Aug 20 01:58:51 UTC 2025
# keep awake Wed Aug 20 06:37:39 UTC 2025
# keep awake Wed Aug 20 12:52:46 UTC 2025
# keep awake Wed Aug 20 18:35:54 UTC 2025
# keep awake Thu Aug 21 01:57:39 UTC 2025
# keep awake Thu Aug 21 06:47:22 UTC 2025
# keep awake Thu Aug 21 12:52:12 UTC 2025
# keep awake Thu Aug 21 18:33:53 UTC 2025
# keep awake Fri Aug 22 01:58:35 UTC 2025
# keep awake Fri Aug 22 06:37:00 UTC 2025
# keep awake Fri Aug 22 12:50:43 UTC 2025
# keep awake Fri Aug 22 18:34:25 UTC 2025
# keep awake Sat Aug 23 01:54:19 UTC 2025
# keep awake Sat Aug 23 06:33:37 UTC 2025
# keep awake Sat Aug 23 12:47:30 UTC 2025
# keep awake Sat Aug 23 18:31:36 UTC 2025
# keep awake Sun Aug 24 02:09:23 UTC 2025
# keep awake Sun Aug 24 06:34:32 UTC 2025
# keep awake Sun Aug 24 12:47:52 UTC 2025
# keep awake Sun Aug 24 18:32:53 UTC 2025
# keep awake Mon Aug 25 02:03:32 UTC 2025
# keep awake Mon Aug 25 06:39:11 UTC 2025
# keep awake Mon Aug 25 12:53:23 UTC 2025
# keep awake Mon Aug 25 18:36:12 UTC 2025
# keep awake Tue Aug 26 01:59:34 UTC 2025
# keep awake Tue Aug 26 06:38:01 UTC 2025
# keep awake Tue Aug 26 12:54:32 UTC 2025
# keep awake Tue Aug 26 18:34:14 UTC 2025
# keep awake Wed Aug 27 01:55:38 UTC 2025
# keep awake Wed Aug 27 06:35:10 UTC 2025
# keep awake Wed Aug 27 12:50:48 UTC 2025
# keep awake Wed Aug 27 18:33:10 UTC 2025
# keep awake Thu Aug 28 01:54:59 UTC 2025
# keep awake Thu Aug 28 06:36:17 UTC 2025
# keep awake Thu Aug 28 12:51:15 UTC 2025
# keep awake Thu Aug 28 18:34:29 UTC 2025
# keep awake Fri Aug 29 01:54:57 UTC 2025
# keep awake Fri Aug 29 06:35:35 UTC 2025
# keep awake Fri Aug 29 12:50:02 UTC 2025
# keep awake Fri Aug 29 18:31:09 UTC 2025
# keep awake Sat Aug 30 01:49:18 UTC 2025
# keep awake Sat Aug 30 06:32:13 UTC 2025
# keep awake Sat Aug 30 12:44:47 UTC 2025
# keep awake Sat Aug 30 18:30:52 UTC 2025
# keep awake Sun Aug 31 02:00:39 UTC 2025
# keep awake Sun Aug 31 06:33:01 UTC 2025
# keep awake Sun Aug 31 12:45:38 UTC 2025
# keep awake Sun Aug 31 18:31:45 UTC 2025
# keep awake Mon Sep  1 02:11:04 UTC 2025
# keep awake Mon Sep  1 06:39:27 UTC 2025
# keep awake Mon Sep  1 12:52:47 UTC 2025
# keep awake Mon Sep  1 18:33:02 UTC 2025
# keep awake Tue Sep  2 01:57:02 UTC 2025
# keep awake Tue Sep  2 06:37:16 UTC 2025
# keep awake Tue Sep  2 12:52:00 UTC 2025
# keep awake Tue Sep  2 18:32:16 UTC 2025
# keep awake Wed Sep  3 01:48:50 UTC 2025
# keep awake Wed Sep  3 06:33:55 UTC 2025
# keep awake Wed Sep  3 12:49:41 UTC 2025
# keep awake Wed Sep  3 18:33:12 UTC 2025
# keep awake Thu Sep  4 01:49:27 UTC 2025
# keep awake Thu Sep  4 06:34:34 UTC 2025
# keep awake Thu Sep  4 12:48:26 UTC 2025
# keep awake Thu Sep  4 18:32:58 UTC 2025
# keep awake Fri Sep  5 01:52:12 UTC 2025
# keep awake Fri Sep  5 06:35:07 UTC 2025
# keep awake Fri Sep  5 12:49:05 UTC 2025
# keep awake Fri Sep  5 18:31:54 UTC 2025
# keep awake Sat Sep  6 01:49:08 UTC 2025
# keep awake Sat Sep  6 06:31:22 UTC 2025
# keep awake Sat Sep  6 12:42:23 UTC 2025
# keep awake Sat Sep  6 18:29:36 UTC 2025
# keep awake Sun Sep  7 01:58:08 UTC 2025
# keep awake Sun Sep  7 06:32:04 UTC 2025
# keep awake Sun Sep  7 12:43:22 UTC 2025
# keep awake Sun Sep  7 18:29:52 UTC 2025
# keep awake Mon Sep  8 01:57:57 UTC 2025
# keep awake Mon Sep  8 06:37:17 UTC 2025
# keep awake Mon Sep  8 12:52:35 UTC 2025
# keep awake Mon Sep  8 18:34:14 UTC 2025
# keep awake Tue Sep  9 01:53:52 UTC 2025
# keep awake Tue Sep  9 06:36:19 UTC 2025
# keep awake Tue Sep  9 12:53:06 UTC 2025
# keep awake Tue Sep  9 18:31:31 UTC 2025
# keep awake Wed Sep 10 01:49:58 UTC 2025
# keep awake Wed Sep 10 06:35:10 UTC 2025
# keep awake Wed Sep 10 12:49:36 UTC 2025
# keep awake Wed Sep 10 18:33:56 UTC 2025
# keep awake Thu Sep 11 01:53:10 UTC 2025
# keep awake Thu Sep 11 06:35:42 UTC 2025
# keep awake Thu Sep 11 12:48:45 UTC 2025
# keep awake Thu Sep 11 18:30:13 UTC 2025
# keep awake Fri Sep 12 01:48:42 UTC 2025
# keep awake Fri Sep 12 06:34:59 UTC 2025
# keep awake Fri Sep 12 12:47:44 UTC 2025
# keep awake Fri Sep 12 18:29:44 UTC 2025
# keep awake Sat Sep 13 01:44:02 UTC 2025
# keep awake Sat Sep 13 06:31:27 UTC 2025
# keep awake Sat Sep 13 12:42:27 UTC 2025
# keep awake Sat Sep 13 18:28:46 UTC 2025
# keep awake Sun Sep 14 01:57:30 UTC 2025
# keep awake Sun Sep 14 06:32:07 UTC 2025
# keep awake Sun Sep 14 12:42:29 UTC 2025
# keep awake Sun Sep 14 18:29:05 UTC 2025
# keep awake Mon Sep 15 01:59:10 UTC 2025
# keep awake Mon Sep 15 06:37:24 UTC 2025
# keep awake Mon Sep 15 12:51:36 UTC 2025
# keep awake Mon Sep 15 18:34:04 UTC 2025
# keep awake Tue Sep 16 01:49:53 UTC 2025
# keep awake Tue Sep 16 06:35:41 UTC 2025
# keep awake Tue Sep 16 12:51:10 UTC 2025
# keep awake Tue Sep 16 18:34:22 UTC 2025
# keep awake Wed Sep 17 01:49:46 UTC 2025
# keep awake Wed Sep 17 06:35:49 UTC 2025
# keep awake Wed Sep 17 12:51:20 UTC 2025
# keep awake Wed Sep 17 18:34:03 UTC 2025
# keep awake Thu Sep 18 01:49:52 UTC 2025
# keep awake Thu Sep 18 06:35:31 UTC 2025
# keep awake Thu Sep 18 12:50:04 UTC 2025
# keep awake Thu Sep 18 18:34:54 UTC 2025
# keep awake Fri Sep 19 01:53:33 UTC 2025
# keep awake Fri Sep 19 06:35:14 UTC 2025
# keep awake Fri Sep 19 12:50:18 UTC 2025
# keep awake Fri Sep 19 18:33:01 UTC 2025
# keep awake Sat Sep 20 01:48:08 UTC 2025
# keep awake Sat Sep 20 06:32:31 UTC 2025
# keep awake Sat Sep 20 12:45:41 UTC 2025
# keep awake Sat Sep 20 18:29:31 UTC 2025
# keep awake Sun Sep 21 02:00:52 UTC 2025
# keep awake Sun Sep 21 06:33:39 UTC 2025
# keep awake Sun Sep 21 12:44:48 UTC 2025
# keep awake Sun Sep 21 18:31:20 UTC 2025
# keep awake Mon Sep 22 02:00:53 UTC 2025
# keep awake Mon Sep 22 06:37:42 UTC 2025
# keep awake Mon Sep 22 12:51:59 UTC 2025
# keep awake Mon Sep 22 18:33:16 UTC 2025
# keep awake Tue Sep 23 01:50:40 UTC 2025
# keep awake Tue Sep 23 06:36:03 UTC 2025
# keep awake Tue Sep 23 12:50:27 UTC 2025
# keep awake Tue Sep 23 18:35:06 UTC 2025
# keep awake Wed Sep 24 01:52:16 UTC 2025
# keep awake Wed Sep 24 06:35:55 UTC 2025
# keep awake Wed Sep 24 12:51:17 UTC 2025
# keep awake Wed Sep 24 18:32:12 UTC 2025
# keep awake Thu Sep 25 01:53:08 UTC 2025
# keep awake Thu Sep 25 06:36:41 UTC 2025
# keep awake Thu Sep 25 12:52:24 UTC 2025
# keep awake Thu Sep 25 18:35:13 UTC 2025
# keep awake Fri Sep 26 01:52:17 UTC 2025
# keep awake Fri Sep 26 06:35:14 UTC 2025
# keep awake Fri Sep 26 12:50:58 UTC 2025
# keep awake Fri Sep 26 18:31:52 UTC 2025
# keep awake Sat Sep 27 01:47:57 UTC 2025
# keep awake Sat Sep 27 06:31:14 UTC 2025
# keep awake Sat Sep 27 12:44:10 UTC 2025
# keep awake Sat Sep 27 18:30:31 UTC 2025
# keep awake Sun Sep 28 02:01:40 UTC 2025
# keep awake Sun Sep 28 06:32:56 UTC 2025
# keep awake Sun Sep 28 12:44:45 UTC 2025
# keep awake Sun Sep 28 18:29:59 UTC 2025
# keep awake Mon Sep 29 01:55:30 UTC 2025
# keep awake Mon Sep 29 06:37:49 UTC 2025
# keep awake Mon Sep 29 12:53:14 UTC 2025
# keep awake Mon Sep 29 18:35:09 UTC 2025
# keep awake Tue Sep 30 01:50:18 UTC 2025
# keep awake Tue Sep 30 06:36:40 UTC 2025
# keep awake Tue Sep 30 12:53:25 UTC 2025
# keep awake Tue Sep 30 18:33:15 UTC 2025
# keep awake Wed Oct  1 02:03:00 UTC 2025
# keep awake Wed Oct  1 06:36:25 UTC 2025
# keep awake Wed Oct  1 12:53:14 UTC 2025
# keep awake Wed Oct  1 18:34:32 UTC 2025
# keep awake Thu Oct  2 01:50:49 UTC 2025
# keep awake Thu Oct  2 06:34:59 UTC 2025
# keep awake Thu Oct  2 12:49:07 UTC 2025
# keep awake Thu Oct  2 18:34:03 UTC 2025
# keep awake Fri Oct  3 01:50:10 UTC 2025
# keep awake Fri Oct  3 06:34:40 UTC 2025
# keep awake Fri Oct  3 12:49:11 UTC 2025
# keep awake Fri Oct  3 18:33:33 UTC 2025
# keep awake Sat Oct  4 01:46:26 UTC 2025
# keep awake Sat Oct  4 06:32:25 UTC 2025
# keep awake Sat Oct  4 12:43:58 UTC 2025
# keep awake Sat Oct  4 18:30:53 UTC 2025
# keep awake Sun Oct  5 02:00:59 UTC 2025
# keep awake Sun Oct  5 06:32:18 UTC 2025
# keep awake Sun Oct  5 12:44:35 UTC 2025
# keep awake Sun Oct  5 18:30:56 UTC 2025
# keep awake Mon Oct  6 01:53:41 UTC 2025
# keep awake Mon Oct  6 06:36:42 UTC 2025
# keep awake Mon Oct  6 12:52:26 UTC 2025
# keep awake Mon Oct  6 18:34:55 UTC 2025
# keep awake Tue Oct  7 01:51:12 UTC 2025
# keep awake Tue Oct  7 06:35:53 UTC 2025
# keep awake Tue Oct  7 12:52:07 UTC 2025
# keep awake Tue Oct  7 18:35:38 UTC 2025
# keep awake Wed Oct  8 01:51:14 UTC 2025
# keep awake Wed Oct  8 06:36:25 UTC 2025
# keep awake Wed Oct  8 12:52:21 UTC 2025
# keep awake Wed Oct  8 18:36:04 UTC 2025
# keep awake Thu Oct  9 01:52:24 UTC 2025
# keep awake Thu Oct  9 06:36:46 UTC 2025
# keep awake Thu Oct  9 12:52:16 UTC 2025
# keep awake Thu Oct  9 18:34:30 UTC 2025
# keep awake Fri Oct 10 01:52:48 UTC 2025
# keep awake Fri Oct 10 06:36:16 UTC 2025
# keep awake Fri Oct 10 12:50:56 UTC 2025
# keep awake Fri Oct 10 18:33:58 UTC 2025
# keep awake Sat Oct 11 01:48:00 UTC 2025
# keep awake Sat Oct 11 06:32:22 UTC 2025
# keep awake Sat Oct 11 12:44:02 UTC 2025
# keep awake Sat Oct 11 18:28:54 UTC 2025
# keep awake Sun Oct 12 01:56:40 UTC 2025
# keep awake Sun Oct 12 06:32:40 UTC 2025
# keep awake Sun Oct 12 12:45:01 UTC 2025
# keep awake Sun Oct 12 18:31:07 UTC 2025
# keep awake Mon Oct 13 01:59:59 UTC 2025
# keep awake Mon Oct 13 06:38:11 UTC 2025
# keep awake Mon Oct 13 12:52:20 UTC 2025
# keep awake Mon Oct 13 18:33:41 UTC 2025
# keep awake Tue Oct 14 01:53:13 UTC 2025
# keep awake Tue Oct 14 06:36:03 UTC 2025
# keep awake Tue Oct 14 12:54:19 UTC 2025
# keep awake Tue Oct 14 18:35:19 UTC 2025
# keep awake Wed Oct 15 01:56:06 UTC 2025
# keep awake Wed Oct 15 06:36:17 UTC 2025
# keep awake Wed Oct 15 12:53:54 UTC 2025
# keep awake Wed Oct 15 18:36:26 UTC 2025
# keep awake Thu Oct 16 01:54:50 UTC 2025
# keep awake Thu Oct 16 06:36:15 UTC 2025
# keep awake Thu Oct 16 12:53:44 UTC 2025
# keep awake Thu Oct 16 18:35:18 UTC 2025
# keep awake Fri Oct 17 01:53:26 UTC 2025
# keep awake Fri Oct 17 06:35:25 UTC 2025
# keep awake Fri Oct 17 12:52:04 UTC 2025
# keep awake Fri Oct 17 18:32:20 UTC 2025
# keep awake Sat Oct 18 01:48:44 UTC 2025
# keep awake Sat Oct 18 06:33:08 UTC 2025
# keep awake Sat Oct 18 12:46:05 UTC 2025
# keep awake Sat Oct 18 18:31:03 UTC 2025
# keep awake Sun Oct 19 02:06:47 UTC 2025
# keep awake Sun Oct 19 06:34:16 UTC 2025
# keep awake Sun Oct 19 12:46:19 UTC 2025
# keep awake Sun Oct 19 18:31:53 UTC 2025
# keep awake Mon Oct 20 02:03:26 UTC 2025
# keep awake Mon Oct 20 06:37:43 UTC 2025
# keep awake Mon Oct 20 12:53:04 UTC 2025
# keep awake Mon Oct 20 18:36:55 UTC 2025
# keep awake Tue Oct 21 01:56:48 UTC 2025
# keep awake Tue Oct 21 06:36:44 UTC 2025
# keep awake Tue Oct 21 12:54:47 UTC 2025
# keep awake Tue Oct 21 18:35:25 UTC 2025
# keep awake Wed Oct 22 02:00:11 UTC 2025
# keep awake Wed Oct 22 06:37:49 UTC 2025
# keep awake Wed Oct 22 12:55:42 UTC 2025
# keep awake Wed Oct 22 18:37:15 UTC 2025
# keep awake Thu Oct 23 01:55:39 UTC 2025
# keep awake Thu Oct 23 06:37:08 UTC 2025
# keep awake Thu Oct 23 12:54:36 UTC 2025
# keep awake Thu Oct 23 18:35:53 UTC 2025
# keep awake Fri Oct 24 01:51:35 UTC 2025
# keep awake Fri Oct 24 06:35:41 UTC 2025
# keep awake Fri Oct 24 12:53:57 UTC 2025
# keep awake Fri Oct 24 18:33:26 UTC 2025
# keep awake Sat Oct 25 01:53:27 UTC 2025
# keep awake Sat Oct 25 06:33:05 UTC 2025
# keep awake Sat Oct 25 12:45:58 UTC 2025
# keep awake Sat Oct 25 18:31:56 UTC 2025
# keep awake Sun Oct 26 02:02:28 UTC 2025
# keep awake Sun Oct 26 06:34:19 UTC 2025
# keep awake Sun Oct 26 12:47:38 UTC 2025
# keep awake Sun Oct 26 18:33:03 UTC 2025
# keep awake Mon Oct 27 02:06:41 UTC 2025
# keep awake Mon Oct 27 06:40:19 UTC 2025
# keep awake Mon Oct 27 12:54:29 UTC 2025
# keep awake Mon Oct 27 18:36:18 UTC 2025
# keep awake Tue Oct 28 01:56:07 UTC 2025
# keep awake Tue Oct 28 06:38:17 UTC 2025
# keep awake Tue Oct 28 12:53:01 UTC 2025
# keep awake Tue Oct 28 18:37:40 UTC 2025
# keep awake Wed Oct 29 02:03:12 UTC 2025
# keep awake Wed Oct 29 06:38:45 UTC 2025
# keep awake Wed Oct 29 12:54:58 UTC 2025
# keep awake Wed Oct 29 18:36:48 UTC 2025
# keep awake Thu Oct 30 02:02:20 UTC 2025
# keep awake Thu Oct 30 06:36:38 UTC 2025
# keep awake Thu Oct 30 12:53:13 UTC 2025
# keep awake Thu Oct 30 18:36:48 UTC 2025
# keep awake Fri Oct 31 01:59:08 UTC 2025
# keep awake Fri Oct 31 06:37:21 UTC 2025
# keep awake Fri Oct 31 12:52:48 UTC 2025
# keep awake Fri Oct 31 18:36:14 UTC 2025
# keep awake Sat Nov  1 02:02:44 UTC 2025
# keep awake Sat Nov  1 06:33:41 UTC 2025
# keep awake Sat Nov  1 12:46:48 UTC 2025
# keep awake Sat Nov  1 18:31:37 UTC 2025
# keep awake Sun Nov  2 02:05:30 UTC 2025
# keep awake Sun Nov  2 06:35:37 UTC 2025
# keep awake Sun Nov  2 12:46:06 UTC 2025
# keep awake Sun Nov  2 18:31:02 UTC 2025
# keep awake Mon Nov  3 02:04:33 UTC 2025
# keep awake Mon Nov  3 06:39:09 UTC 2025
# keep awake Mon Nov  3 12:54:00 UTC 2025
# keep awake Mon Nov  3 18:34:13 UTC 2025
# keep awake Tue Nov  4 01:59:42 UTC 2025
# keep awake Tue Nov  4 06:38:23 UTC 2025
# keep awake Tue Nov  4 12:56:42 UTC 2025
# keep awake Tue Nov  4 18:36:27 UTC 2025
# keep awake Wed Nov  5 02:01:03 UTC 2025
# keep awake Wed Nov  5 06:37:42 UTC 2025
# keep awake Wed Nov  5 12:53:55 UTC 2025
# keep awake Wed Nov  5 18:36:25 UTC 2025
# keep awake Thu Nov  6 02:01:11 UTC 2025
# keep awake Thu Nov  6 06:37:59 UTC 2025
# keep awake Thu Nov  6 12:53:48 UTC 2025
# keep awake Thu Nov  6 18:37:44 UTC 2025
# keep awake Fri Nov  7 01:59:42 UTC 2025
# keep awake Fri Nov  7 06:37:35 UTC 2025
# keep awake Fri Nov  7 12:52:30 UTC 2025
# keep awake Fri Nov  7 18:35:08 UTC 2025
# keep awake Sat Nov  8 01:53:35 UTC 2025
# keep awake Sat Nov  8 06:34:23 UTC 2025
# keep awake Sat Nov  8 12:47:10 UTC 2025
# keep awake Sat Nov  8 18:32:39 UTC 2025
# keep awake Sun Nov  9 02:04:22 UTC 2025
# keep awake Sun Nov  9 06:34:41 UTC 2025
# keep awake Sun Nov  9 12:47:15 UTC 2025
# keep awake Sun Nov  9 18:31:42 UTC 2025
# keep awake Mon Nov 10 02:05:46 UTC 2025
# keep awake Mon Nov 10 06:39:33 UTC 2025
# keep awake Mon Nov 10 12:55:00 UTC 2025
# keep awake Mon Nov 10 18:36:11 UTC 2025
# keep awake Tue Nov 11 02:01:49 UTC 2025
# keep awake Tue Nov 11 06:38:21 UTC 2025
# keep awake Tue Nov 11 12:53:57 UTC 2025
# keep awake Tue Nov 11 18:35:39 UTC 2025
# keep awake Wed Nov 12 02:00:49 UTC 2025
# keep awake Wed Nov 12 06:38:13 UTC 2025
# keep awake Wed Nov 12 12:55:30 UTC 2025
# keep awake Wed Nov 12 18:34:25 UTC 2025
# keep awake Thu Nov 13 02:03:00 UTC 2025
# keep awake Thu Nov 13 06:38:12 UTC 2025
# keep awake Thu Nov 13 12:55:31 UTC 2025
# keep awake Thu Nov 13 18:36:23 UTC 2025
# keep awake Fri Nov 14 02:01:28 UTC 2025
# keep awake Fri Nov 14 06:37:56 UTC 2025
# keep awake Fri Nov 14 12:53:07 UTC 2025
# keep awake Fri Nov 14 18:35:43 UTC 2025
# keep awake Sat Nov 15 01:57:13 UTC 2025
# keep awake Sat Nov 15 06:34:56 UTC 2025
# keep awake Sat Nov 15 12:48:00 UTC 2025
# keep awake Sat Nov 15 18:32:22 UTC 2025
# keep awake Sun Nov 16 02:07:21 UTC 2025
# keep awake Sun Nov 16 06:35:46 UTC 2025
# keep awake Sun Nov 16 12:47:37 UTC 2025
# keep awake Sun Nov 16 18:33:24 UTC 2025
# keep awake Mon Nov 17 02:02:48 UTC 2025
# keep awake Mon Nov 17 06:38:50 UTC 2025
# keep awake Mon Nov 17 12:54:26 UTC 2025
# keep awake Mon Nov 17 18:36:09 UTC 2025
# keep awake Tue Nov 18 02:00:44 UTC 2025
# keep awake Tue Nov 18 06:37:26 UTC 2025
# keep awake Tue Nov 18 12:54:32 UTC 2025
# keep awake Tue Nov 18 18:37:49 UTC 2025
# keep awake Wed Nov 19 02:00:37 UTC 2025
# keep awake Wed Nov 19 06:37:40 UTC 2025
# keep awake Wed Nov 19 12:54:31 UTC 2025
# keep awake Wed Nov 19 18:37:55 UTC 2025
# keep awake Thu Nov 20 01:59:06 UTC 2025
# keep awake Thu Nov 20 06:37:17 UTC 2025
# keep awake Thu Nov 20 12:53:29 UTC 2025
# keep awake Thu Nov 20 18:37:18 UTC 2025
# keep awake Fri Nov 21 01:59:49 UTC 2025
# keep awake Fri Nov 21 06:39:06 UTC 2025
# keep awake Fri Nov 21 12:52:44 UTC 2025
# keep awake Fri Nov 21 18:33:22 UTC 2025
# keep awake Sat Nov 22 01:56:02 UTC 2025
# keep awake Sat Nov 22 06:34:30 UTC 2025
# keep awake Sat Nov 22 12:46:35 UTC 2025
# keep awake Sat Nov 22 18:33:57 UTC 2025
# keep awake Sun Nov 23 02:20:55 UTC 2025
# keep awake Sun Nov 23 06:36:07 UTC 2025
# keep awake Sun Nov 23 12:46:16 UTC 2025
# keep awake Sun Nov 23 18:34:13 UTC 2025
# keep awake Mon Nov 24 02:09:09 UTC 2025
# keep awake Mon Nov 24 06:39:14 UTC 2025
# keep awake Mon Nov 24 12:56:02 UTC 2025
# keep awake Mon Nov 24 18:38:19 UTC 2025
# keep awake Tue Nov 25 02:02:10 UTC 2025
# keep awake Tue Nov 25 06:40:00 UTC 2025
# keep awake Tue Nov 25 12:55:17 UTC 2025
# keep awake Tue Nov 25 18:38:50 UTC 2025
# keep awake Wed Nov 26 02:02:22 UTC 2025
# keep awake Wed Nov 26 06:39:38 UTC 2025
# keep awake Wed Nov 26 12:56:37 UTC 2025
# keep awake Wed Nov 26 18:33:38 UTC 2025
# keep awake Thu Nov 27 01:59:51 UTC 2025
# keep awake Thu Nov 27 06:40:08 UTC 2025
# keep awake Thu Nov 27 12:55:18 UTC 2025
# keep awake Thu Nov 27 18:35:28 UTC 2025
# keep awake Fri Nov 28 01:59:28 UTC 2025
# keep awake Fri Nov 28 06:39:34 UTC 2025
# keep awake Fri Nov 28 12:53:42 UTC 2025
# keep awake Fri Nov 28 18:35:38 UTC 2025
# keep awake Sat Nov 29 01:59:14 UTC 2025
# keep awake Sat Nov 29 06:36:23 UTC 2025
# keep awake Sat Nov 29 12:50:18 UTC 2025
# keep awake Sat Nov 29 18:34:32 UTC 2025
# keep awake Sun Nov 30 02:19:23 UTC 2025
# keep awake Sun Nov 30 06:37:52 UTC 2025
# keep awake Sun Nov 30 12:49:44 UTC 2025
# keep awake Sun Nov 30 18:34:43 UTC 2025
# keep awake Mon Dec  1 02:28:12 UTC 2025
# keep awake Mon Dec  1 06:42:29 UTC 2025
# keep awake Mon Dec  1 12:56:45 UTC 2025
# keep awake Mon Dec  1 18:42:27 UTC 2025
# keep awake Tue Dec  2 02:04:26 UTC 2025
# keep awake Tue Dec  2 06:40:46 UTC 2025
# keep awake Tue Dec  2 12:57:44 UTC 2025
# keep awake Tue Dec  2 18:41:27 UTC 2025
# keep awake Wed Dec  3 02:03:42 UTC 2025
# keep awake Wed Dec  3 06:40:26 UTC 2025
# keep awake Wed Dec  3 12:58:21 UTC 2025
# keep awake Wed Dec  3 18:40:28 UTC 2025
# keep awake Thu Dec  4 02:05:04 UTC 2025
# keep awake Thu Dec  4 06:40:19 UTC 2025
# keep awake Thu Dec  4 12:58:12 UTC 2025
# keep awake Thu Dec  4 18:41:19 UTC 2025
# keep awake Fri Dec  5 02:05:17 UTC 2025
# keep awake Fri Dec  5 06:40:37 UTC 2025
# keep awake Fri Dec  5 12:54:57 UTC 2025
# keep awake Fri Dec  5 18:35:30 UTC 2025
# keep awake Sat Dec  6 01:58:29 UTC 2025
# keep awake Sat Dec  6 06:36:05 UTC 2025
# keep awake Sat Dec  6 12:50:37 UTC 2025
# keep awake Sat Dec  6 18:34:28 UTC 2025
# keep awake Sun Dec  7 02:20:10 UTC 2025
# keep awake Sun Dec  7 06:35:56 UTC 2025
# keep awake Sun Dec  7 12:49:26 UTC 2025
# keep awake Sun Dec  7 18:34:24 UTC 2025
# keep awake Mon Dec  8 02:06:16 UTC 2025
# keep awake Mon Dec  8 06:44:19 UTC 2025
# keep awake Mon Dec  8 12:56:21 UTC 2025
# keep awake Mon Dec  8 18:40:09 UTC 2025
# keep awake Tue Dec  9 02:04:55 UTC 2025
# keep awake Tue Dec  9 06:40:59 UTC 2025
# keep awake Tue Dec  9 12:58:17 UTC 2025
# keep awake Tue Dec  9 18:36:22 UTC 2025
# keep awake Wed Dec 10 02:07:09 UTC 2025
# keep awake Wed Dec 10 06:41:34 UTC 2025
# keep awake Wed Dec 10 12:58:41 UTC 2025
# keep awake Wed Dec 10 18:39:04 UTC 2025
# keep awake Thu Dec 11 02:08:44 UTC 2025
# keep awake Thu Dec 11 06:42:57 UTC 2025
# keep awake Thu Dec 11 12:59:50 UTC 2025
# keep awake Thu Dec 11 18:39:40 UTC 2025
# keep awake Fri Dec 12 02:08:32 UTC 2025
# keep awake Fri Dec 12 06:41:51 UTC 2025
# keep awake Fri Dec 12 12:57:01 UTC 2025
# keep awake Fri Dec 12 18:40:04 UTC 2025
# keep awake Sat Dec 13 02:01:47 UTC 2025
# keep awake Sat Dec 13 06:38:30 UTC 2025
# keep awake Sat Dec 13 12:51:27 UTC 2025
# keep awake Sat Dec 13 18:34:10 UTC 2025
# keep awake Sun Dec 14 02:20:57 UTC 2025
# keep awake Sun Dec 14 06:38:05 UTC 2025
# keep awake Sun Dec 14 12:51:27 UTC 2025
# keep awake Sun Dec 14 18:35:37 UTC 2025
# keep awake Mon Dec 15 02:18:17 UTC 2025
# keep awake Mon Dec 15 06:45:34 UTC 2025
# keep awake Mon Dec 15 13:00:34 UTC 2025
# keep awake Mon Dec 15 18:40:59 UTC 2025
# keep awake Tue Dec 16 02:09:16 UTC 2025
# keep awake Tue Dec 16 06:42:32 UTC 2025
# keep awake Tue Dec 16 12:59:05 UTC 2025
# keep awake Tue Dec 16 18:40:45 UTC 2025
# keep awake Wed Dec 17 02:04:14 UTC 2025
# keep awake Wed Dec 17 06:42:07 UTC 2025
# keep awake Wed Dec 17 12:58:37 UTC 2025
# keep awake Wed Dec 17 18:41:29 UTC 2025
# keep awake Thu Dec 18 02:04:59 UTC 2025
# keep awake Thu Dec 18 06:41:54 UTC 2025
# keep awake Thu Dec 18 12:56:58 UTC 2025
# keep awake Thu Dec 18 18:40:18 UTC 2025
# keep awake Fri Dec 19 02:08:08 UTC 2025
# keep awake Fri Dec 19 06:40:34 UTC 2025
# keep awake Fri Dec 19 12:55:29 UTC 2025
# keep awake Fri Dec 19 18:38:39 UTC 2025
# keep awake Sat Dec 20 02:01:23 UTC 2025
# keep awake Sat Dec 20 06:38:28 UTC 2025
# keep awake Sat Dec 20 12:51:19 UTC 2025
# keep awake Sat Dec 20 18:34:00 UTC 2025
# keep awake Sun Dec 21 02:20:59 UTC 2025
# keep awake Sun Dec 21 06:38:53 UTC 2025
# keep awake Sun Dec 21 12:52:01 UTC 2025
# keep awake Sun Dec 21 18:35:51 UTC 2025
# keep awake Mon Dec 22 02:19:54 UTC 2025
# keep awake Mon Dec 22 06:44:17 UTC 2025
# keep awake Mon Dec 22 12:56:21 UTC 2025
# keep awake Mon Dec 22 18:39:30 UTC 2025
# keep awake Tue Dec 23 02:09:15 UTC 2025
# keep awake Tue Dec 23 06:43:29 UTC 2025
# keep awake Tue Dec 23 12:57:56 UTC 2025
# keep awake Tue Dec 23 18:39:54 UTC 2025
# keep awake Wed Dec 24 02:07:18 UTC 2025
# keep awake Wed Dec 24 06:43:25 UTC 2025
# keep awake Wed Dec 24 12:55:35 UTC 2025
# keep awake Wed Dec 24 18:38:09 UTC 2025
# keep awake Thu Dec 25 02:09:34 UTC 2025
# keep awake Thu Dec 25 06:41:53 UTC 2025
# keep awake Thu Dec 25 12:55:13 UTC 2025
# keep awake Thu Dec 25 18:37:46 UTC 2025
# keep awake Fri Dec 26 02:08:26 UTC 2025
# keep awake Fri Dec 26 06:41:11 UTC 2025
# keep awake Fri Dec 26 12:55:17 UTC 2025
# keep awake Fri Dec 26 18:37:35 UTC 2025
# keep awake Sat Dec 27 02:06:11 UTC 2025
# keep awake Sat Dec 27 06:39:38 UTC 2025
# keep awake Sat Dec 27 12:52:48 UTC 2025
# keep awake Sat Dec 27 18:36:20 UTC 2025
# keep awake Sun Dec 28 02:26:53 UTC 2025
# keep awake Sun Dec 28 06:39:49 UTC 2025
# keep awake Sun Dec 28 12:53:29 UTC 2025
# keep awake Sun Dec 28 18:37:16 UTC 2025
# keep awake Mon Dec 29 02:24:49 UTC 2025
# keep awake Mon Dec 29 06:46:10 UTC 2025
# keep awake Mon Dec 29 12:58:51 UTC 2025
# keep awake Mon Dec 29 18:38:45 UTC 2025
# keep awake Tue Dec 30 02:10:09 UTC 2025
# keep awake Tue Dec 30 06:42:04 UTC 2025
# keep awake Tue Dec 30 12:58:07 UTC 2025
# keep awake Tue Dec 30 18:41:03 UTC 2025
# keep awake Wed Dec 31 02:10:24 UTC 2025
# keep awake Wed Dec 31 06:42:28 UTC 2025
# keep awake Wed Dec 31 12:56:10 UTC 2025
# keep awake Wed Dec 31 18:37:49 UTC 2025
# keep awake Thu Jan  1 02:27:31 UTC 2026
# keep awake Thu Jan  1 06:42:32 UTC 2026
# keep awake Thu Jan  1 12:56:23 UTC 2026
# keep awake Thu Jan  1 18:38:22 UTC 2026
# keep awake Fri Jan  2 02:11:55 UTC 2026
# keep awake Fri Jan  2 06:43:04 UTC 2026
# keep awake Fri Jan  2 12:55:22 UTC 2026
# keep awake Fri Jan  2 18:37:58 UTC 2026
# keep awake Sat Jan  3 02:05:31 UTC 2026
# keep awake Sat Jan  3 06:39:39 UTC 2026
# keep awake Sat Jan  3 12:53:13 UTC 2026
# keep awake Sat Jan  3 18:36:39 UTC 2026
# keep awake Sun Jan  4 02:28:55 UTC 2026
# keep awake Sun Jan  4 06:40:39 UTC 2026
# keep awake Sun Jan  4 12:53:59 UTC 2026
# keep awake Sun Jan  4 18:37:05 UTC 2026
# keep awake Mon Jan  5 02:28:25 UTC 2026
# keep awake Mon Jan  5 06:49:26 UTC 2026
# keep awake Mon Jan  5 13:01:07 UTC 2026
# keep awake Mon Jan  5 18:41:42 UTC 2026
# keep awake Tue Jan  6 02:11:53 UTC 2026
# keep awake Tue Jan  6 06:45:23 UTC 2026
# keep awake Tue Jan  6 12:58:44 UTC 2026
# keep awake Tue Jan  6 18:40:49 UTC 2026
# keep awake Wed Jan  7 02:11:40 UTC 2026
# keep awake Wed Jan  7 06:43:59 UTC 2026
# keep awake Wed Jan  7 12:59:42 UTC 2026
# keep awake Wed Jan  7 18:42:12 UTC 2026
# keep awake Thu Jan  8 02:11:38 UTC 2026
# keep awake Thu Jan  8 06:44:18 UTC 2026
# keep awake Thu Jan  8 13:00:37 UTC 2026
# keep awake Thu Jan  8 18:38:23 UTC 2026
# keep awake Fri Jan  9 02:18:50 UTC 2026
# keep awake Fri Jan  9 06:44:32 UTC 2026
# keep awake Fri Jan  9 12:59:29 UTC 2026
# keep awake Fri Jan  9 18:41:15 UTC 2026
# keep awake Sat Jan 10 02:08:44 UTC 2026
# keep awake Sat Jan 10 06:39:13 UTC 2026
# keep awake Sat Jan 10 12:53:20 UTC 2026
# keep awake Sat Jan 10 18:37:29 UTC 2026
# keep awake Sun Jan 11 02:28:40 UTC 2026
# keep awake Sun Jan 11 06:41:02 UTC 2026
# keep awake Sun Jan 11 12:54:34 UTC 2026
# keep awake Sun Jan 11 18:37:10 UTC 2026
# keep awake Mon Jan 12 02:24:32 UTC 2026
# keep awake Mon Jan 12 06:48:41 UTC 2026
# keep awake Mon Jan 12 13:01:46 UTC 2026
# keep awake Mon Jan 12 18:41:57 UTC 2026
# keep awake Tue Jan 13 02:09:46 UTC 2026
# keep awake Tue Jan 13 06:44:33 UTC 2026
# keep awake Tue Jan 13 13:01:40 UTC 2026
# keep awake Tue Jan 13 18:39:38 UTC 2026
# keep awake Wed Jan 14 02:23:15 UTC 2026
# keep awake Wed Jan 14 06:44:28 UTC 2026
# keep awake Wed Jan 14 13:01:43 UTC 2026
# keep awake Wed Jan 14 18:42:40 UTC 2026
# keep awake Thu Jan 15 02:11:21 UTC 2026
# keep awake Thu Jan 15 06:43:54 UTC 2026
# keep awake Thu Jan 15 13:00:09 UTC 2026
# keep awake Thu Jan 15 18:47:16 UTC 2026
# keep awake Fri Jan 16 02:19:30 UTC 2026
# keep awake Fri Jan 16 06:44:17 UTC 2026
# keep awake Fri Jan 16 12:59:24 UTC 2026
# keep awake Fri Jan 16 18:41:55 UTC 2026
# keep awake Sat Jan 17 02:07:03 UTC 2026
# keep awake Sat Jan 17 06:39:20 UTC 2026
# keep awake Sat Jan 17 12:53:18 UTC 2026
# keep awake Sat Jan 17 18:36:43 UTC 2026
# keep awake Sun Jan 18 02:27:23 UTC 2026
# keep awake Sun Jan 18 06:40:19 UTC 2026
# keep awake Sun Jan 18 12:53:24 UTC 2026
# keep awake Sun Jan 18 18:35:55 UTC 2026
# keep awake Mon Jan 19 02:24:45 UTC 2026
# keep awake Mon Jan 19 06:50:13 UTC 2026
# keep awake Mon Jan 19 13:04:52 UTC 2026
# keep awake Mon Jan 19 18:41:06 UTC 2026
# keep awake Tue Jan 20 02:19:43 UTC 2026
# keep awake Tue Jan 20 06:48:01 UTC 2026
# keep awake Tue Jan 20 13:05:27 UTC 2026
# keep awake Tue Jan 20 18:53:44 UTC 2026
# keep awake Wed Jan 21 02:22:36 UTC 2026
# keep awake Wed Jan 21 06:48:05 UTC 2026
# keep awake Wed Jan 21 13:04:22 UTC 2026
# keep awake Wed Jan 21 18:52:54 UTC 2026
# keep awake Thu Jan 22 02:23:55 UTC 2026
# keep awake Thu Jan 22 06:46:05 UTC 2026
# keep awake Thu Jan 22 13:05:16 UTC 2026
# keep awake Thu Jan 22 18:43:49 UTC 2026
# keep awake Fri Jan 23 02:19:51 UTC 2026
# keep awake Fri Jan 23 06:45:44 UTC 2026
# keep awake Fri Jan 23 13:02:22 UTC 2026
# keep awake Fri Jan 23 18:44:00 UTC 2026
# keep awake Sat Jan 24 02:09:25 UTC 2026
# keep awake Sat Jan 24 06:40:41 UTC 2026
# keep awake Sat Jan 24 12:54:37 UTC 2026
# keep awake Sat Jan 24 18:38:35 UTC 2026
# keep awake Sun Jan 25 02:31:04 UTC 2026
# keep awake Sun Jan 25 06:41:07 UTC 2026
# keep awake Sun Jan 25 12:56:26 UTC 2026
# keep awake Sun Jan 25 18:38:35 UTC 2026
# keep awake Mon Jan 26 02:30:09 UTC 2026
# keep awake Mon Jan 26 06:49:14 UTC 2026
# keep awake Mon Jan 26 13:03:19 UTC 2026
# keep awake Mon Jan 26 18:47:30 UTC 2026
# keep awake Tue Jan 27 02:25:53 UTC 2026
# keep awake Tue Jan 27 06:47:48 UTC 2026
# keep awake Tue Jan 27 13:05:49 UTC 2026
# keep awake Tue Jan 27 18:50:19 UTC 2026
# keep awake Wed Jan 28 02:22:05 UTC 2026
# keep awake Wed Jan 28 06:47:41 UTC 2026
# keep awake Wed Jan 28 13:06:02 UTC 2026
# keep awake Wed Jan 28 18:46:54 UTC 2026
# keep awake Thu Jan 29 02:38:57 UTC 2026
# keep awake Thu Jan 29 06:58:21 UTC 2026
# keep awake Thu Jan 29 13:13:09 UTC 2026
# keep awake Thu Jan 29 18:57:29 UTC 2026
# keep awake Fri Jan 30 02:39:27 UTC 2026
# keep awake Fri Jan 30 06:59:41 UTC 2026
# keep awake Fri Jan 30 13:11:28 UTC 2026
# keep awake Fri Jan 30 18:54:48 UTC 2026
# keep awake Sat Jan 31 02:34:28 UTC 2026
# keep awake Sat Jan 31 06:50:59 UTC 2026
# keep awake Sat Jan 31 13:02:43 UTC 2026
# keep awake Sat Jan 31 18:41:57 UTC 2026
# keep awake Sun Feb  1 03:16:30 UTC 2026
# keep awake Sun Feb  1 06:59:11 UTC 2026
# keep awake Sun Feb  1 13:05:14 UTC 2026
# keep awake Sun Feb  1 18:46:17 UTC 2026
# keep awake Mon Feb  2 02:49:23 UTC 2026
# keep awake Mon Feb  2 07:11:30 UTC 2026
# keep awake Mon Feb  2 13:16:08 UTC 2026
# keep awake Tue Feb  3 02:45:45 UTC 2026
# keep awake Tue Feb  3 07:01:12 UTC 2026
# keep awake Tue Feb  3 13:16:50 UTC 2026
# keep awake Tue Feb  3 19:04:22 UTC 2026
# keep awake Wed Feb  4 02:40:16 UTC 2026
# keep awake Wed Feb  4 07:02:14 UTC 2026
# keep awake Wed Feb  4 13:16:10 UTC 2026
# keep awake Wed Feb  4 18:59:56 UTC 2026
# keep awake Thu Feb  5 02:42:30 UTC 2026
# keep awake Thu Feb  5 07:08:11 UTC 2026
# keep awake Thu Feb  5 13:18:07 UTC 2026
# keep awake Thu Feb  5 18:58:14 UTC 2026
# keep awake Fri Feb  6 02:42:33 UTC 2026
# keep awake Fri Feb  6 07:03:15 UTC 2026
# keep awake Fri Feb  6 13:15:35 UTC 2026
# keep awake Fri Feb  6 19:00:19 UTC 2026
# keep awake Sat Feb  7 02:37:56 UTC 2026
# keep awake Sat Feb  7 06:53:15 UTC 2026
# keep awake Sat Feb  7 13:04:23 UTC 2026
# keep awake Sat Feb  7 18:46:37 UTC 2026
# keep awake Sun Feb  8 03:21:41 UTC 2026
# keep awake Sun Feb  8 07:00:12 UTC 2026
# keep awake Sun Feb  8 13:05:07 UTC 2026
# keep awake Sun Feb  8 18:48:17 UTC 2026
# keep awake Mon Feb  9 02:50:20 UTC 2026
# keep awake Mon Feb  9 07:14:33 UTC 2026
# keep awake Mon Feb  9 13:30:43 UTC 2026
# keep awake Mon Feb  9 19:25:18 UTC 2026
# keep awake Tue Feb 10 03:14:39 UTC 2026
# keep awake Tue Feb 10 07:15:33 UTC 2026
# keep awake Tue Feb 10 13:41:00 UTC 2026
# keep awake Tue Feb 10 19:15:09 UTC 2026
# keep awake Wed Feb 11 02:56:29 UTC 2026
# keep awake Wed Feb 11 07:12:19 UTC 2026
# keep awake Wed Feb 11 13:26:11 UTC 2026
# keep awake Wed Feb 11 19:12:28 UTC 2026
# keep awake Thu Feb 12 02:52:24 UTC 2026
# keep awake Thu Feb 12 07:11:19 UTC 2026
# keep awake Thu Feb 12 13:25:08 UTC 2026
# keep awake Thu Feb 12 19:12:01 UTC 2026
# keep awake Fri Feb 13 02:52:26 UTC 2026
# keep awake Fri Feb 13 07:07:32 UTC 2026
# keep awake Fri Feb 13 13:16:57 UTC 2026
# keep awake Fri Feb 13 19:00:19 UTC 2026
# keep awake Sat Feb 14 02:40:12 UTC 2026
# keep awake Sat Feb 14 06:56:10 UTC 2026
# keep awake Sat Feb 14 13:04:22 UTC 2026
# keep awake Sat Feb 14 18:45:57 UTC 2026
# keep awake Sun Feb 15 02:53:36 UTC 2026
# keep awake Sun Feb 15 06:58:57 UTC 2026
# keep awake Sun Feb 15 13:06:05 UTC 2026
# keep awake Sun Feb 15 18:47:31 UTC 2026
# keep awake Mon Feb 16 02:48:55 UTC 2026
# keep awake Mon Feb 16 07:13:46 UTC 2026
# keep awake Mon Feb 16 13:19:27 UTC 2026
# keep awake Mon Feb 16 18:54:54 UTC 2026
# keep awake Tue Feb 17 02:45:21 UTC 2026
# keep awake Tue Feb 17 07:08:07 UTC 2026
# keep awake Tue Feb 17 13:19:49 UTC 2026
# keep awake Tue Feb 17 19:06:28 UTC 2026
# keep awake Wed Feb 18 02:48:35 UTC 2026
# keep awake Wed Feb 18 07:10:21 UTC 2026
# keep awake Wed Feb 18 13:21:37 UTC 2026
# keep awake Wed Feb 18 19:04:46 UTC 2026
# keep awake Thu Feb 19 02:47:39 UTC 2026
# keep awake Thu Feb 19 07:09:21 UTC 2026
# keep awake Thu Feb 19 13:23:06 UTC 2026
# keep awake Thu Feb 19 19:00:10 UTC 2026
# keep awake Fri Feb 20 02:42:19 UTC 2026
# keep awake Fri Feb 20 07:05:19 UTC 2026
# keep awake Fri Feb 20 13:14:11 UTC 2026
# keep awake Fri Feb 20 18:58:04 UTC 2026
# keep awake Sat Feb 21 02:36:03 UTC 2026
# keep awake Sat Feb 21 06:53:14 UTC 2026
# keep awake Sat Feb 21 13:03:24 UTC 2026
# keep awake Sat Feb 21 18:46:31 UTC 2026
# keep awake Sun Feb 22 02:48:56 UTC 2026
# keep awake Sun Feb 22 06:57:40 UTC 2026
# keep awake Sun Feb 22 13:04:54 UTC 2026
# keep awake Sun Feb 22 18:47:02 UTC 2026
# keep awake Mon Feb 23 02:49:13 UTC 2026
# keep awake Mon Feb 23 07:15:13 UTC 2026
# keep awake Mon Feb 23 13:21:10 UTC 2026
# keep awake Mon Feb 23 19:14:33 UTC 2026
# keep awake Tue Feb 24 02:46:18 UTC 2026
# keep awake Tue Feb 24 07:09:48 UTC 2026
# keep awake Tue Feb 24 13:22:47 UTC 2026
# keep awake Tue Feb 24 19:11:34 UTC 2026
# keep awake Wed Feb 25 02:47:13 UTC 2026
# keep awake Wed Feb 25 07:10:43 UTC 2026
# keep awake Wed Feb 25 13:22:06 UTC 2026
# keep awake Wed Feb 25 19:14:29 UTC 2026
# keep awake Thu Feb 26 02:42:20 UTC 2026
# keep awake Thu Feb 26 07:09:05 UTC 2026
# keep awake Thu Feb 26 13:23:18 UTC 2026
# keep awake Thu Feb 26 19:01:08 UTC 2026
# keep awake Fri Feb 27 02:40:47 UTC 2026
# keep awake Fri Feb 27 07:03:23 UTC 2026
# keep awake Fri Feb 27 13:14:19 UTC 2026
# keep awake Fri Feb 27 18:55:11 UTC 2026
# keep awake Sat Feb 28 02:29:46 UTC 2026
# keep awake Sat Feb 28 06:50:05 UTC 2026
# keep awake Sat Feb 28 12:59:40 UTC 2026
# keep awake Sat Feb 28 18:42:15 UTC 2026
# keep awake Sun Mar  1 02:55:55 UTC 2026
# keep awake Sun Mar  1 06:55:34 UTC 2026
# keep awake Sun Mar  1 13:03:22 UTC 2026
# keep awake Sun Mar  1 18:43:46 UTC 2026
# keep awake Mon Mar  2 02:43:26 UTC 2026
# keep awake Mon Mar  2 07:07:33 UTC 2026
# keep awake Mon Mar  2 13:15:11 UTC 2026
# keep awake Mon Mar  2 18:58:25 UTC 2026
# keep awake Tue Mar  3 02:46:23 UTC 2026
# keep awake Tue Mar  3 07:02:06 UTC 2026
# keep awake Tue Mar  3 13:13:43 UTC 2026
# keep awake Wed Mar  4 02:39:16 UTC 2026
# keep awake Wed Mar  4 06:58:40 UTC 2026
# keep awake Wed Mar  4 13:12:41 UTC 2026
# keep awake Wed Mar  4 19:00:26 UTC 2026
# keep awake Thu Mar  5 02:42:10 UTC 2026
# keep awake Thu Mar  5 07:02:08 UTC 2026
# keep awake Thu Mar  5 13:16:24 UTC 2026
# keep awake Thu Mar  5 19:21:17 UTC 2026
# keep awake Fri Mar  6 02:39:38 UTC 2026
# keep awake Fri Mar  6 07:00:17 UTC 2026
# keep awake Fri Mar  6 13:12:09 UTC 2026
# keep awake Fri Mar  6 18:56:26 UTC 2026
# keep awake Sat Mar  7 02:32:32 UTC 2026
# keep awake Sat Mar  7 06:52:25 UTC 2026
# keep awake Sat Mar  7 13:01:54 UTC 2026
# keep awake Sat Mar  7 18:43:04 UTC 2026
# keep awake Sun Mar  8 02:45:55 UTC 2026
# keep awake Sun Mar  8 06:53:56 UTC 2026
# keep awake Sun Mar  8 13:03:31 UTC 2026
# keep awake Sun Mar  8 18:45:13 UTC 2026
# keep awake Mon Mar  9 02:47:31 UTC 2026
# keep awake Mon Mar  9 07:11:11 UTC 2026
# keep awake Mon Mar  9 13:20:20 UTC 2026
# keep awake Mon Mar  9 19:01:59 UTC 2026
# keep awake Tue Mar 10 02:38:45 UTC 2026
# keep awake Tue Mar 10 07:00:28 UTC 2026
# keep awake Tue Mar 10 13:17:35 UTC 2026
# keep awake Tue Mar 10 18:59:36 UTC 2026
# keep awake Wed Mar 11 02:37:51 UTC 2026
# keep awake Wed Mar 11 07:03:49 UTC 2026
# keep awake Wed Mar 11 13:17:04 UTC 2026
# keep awake Wed Mar 11 19:02:15 UTC 2026
# keep awake Thu Mar 12 02:43:39 UTC 2026
# keep awake Thu Mar 12 07:05:15 UTC 2026
# keep awake Thu Mar 12 13:17:31 UTC 2026
# keep awake Thu Mar 12 19:02:57 UTC 2026
# keep awake Fri Mar 13 02:41:34 UTC 2026
# keep awake Fri Mar 13 07:04:03 UTC 2026
# keep awake Fri Mar 13 13:15:24 UTC 2026
# keep awake Fri Mar 13 18:53:24 UTC 2026
# keep awake Sat Mar 14 02:40:07 UTC 2026
# keep awake Sat Mar 14 06:58:45 UTC 2026
# keep awake Sat Mar 14 13:07:33 UTC 2026
# keep awake Sat Mar 14 18:50:15 UTC 2026
# keep awake Sun Mar 15 03:18:09 UTC 2026
# keep awake Sun Mar 15 07:07:51 UTC 2026
# keep awake Sun Mar 15 13:08:45 UTC 2026
# keep awake Sun Mar 15 18:51:18 UTC 2026
# keep awake Mon Mar 16 03:20:17 UTC 2026
# keep awake Mon Mar 16 07:41:01 UTC 2026
# keep awake Mon Mar 16 13:42:35 UTC 2026
# keep awake Mon Mar 16 19:13:52 UTC 2026
# keep awake Tue Mar 17 02:43:38 UTC 2026
# keep awake Tue Mar 17 07:13:58 UTC 2026
# keep awake Tue Mar 17 13:40:19 UTC 2026
# keep awake Tue Mar 17 19:12:10 UTC 2026
# keep awake Wed Mar 18 02:50:17 UTC 2026
# keep awake Wed Mar 18 07:11:44 UTC 2026
# keep awake Wed Mar 18 13:41:46 UTC 2026
# keep awake Wed Mar 18 19:11:55 UTC 2026
# keep awake Thu Mar 19 02:51:39 UTC 2026
# keep awake Thu Mar 19 07:06:34 UTC 2026
# keep awake Thu Mar 19 13:22:08 UTC 2026
# keep awake Thu Mar 19 19:07:54 UTC 2026
# keep awake Fri Mar 20 02:42:32 UTC 2026
# keep awake Fri Mar 20 07:04:37 UTC 2026
# keep awake Fri Mar 20 13:15:33 UTC 2026
# keep awake Fri Mar 20 18:59:41 UTC 2026
# keep awake Sat Mar 21 02:35:56 UTC 2026
# keep awake Sat Mar 21 06:55:23 UTC 2026
# keep awake Sat Mar 21 13:04:42 UTC 2026
# keep awake Sat Mar 21 18:47:07 UTC 2026
# keep awake Sun Mar 22 02:52:24 UTC 2026
# keep awake Sun Mar 22 06:59:32 UTC 2026
# keep awake Sun Mar 22 13:06:54 UTC 2026
# keep awake Sun Mar 22 18:48:33 UTC 2026
# keep awake Mon Mar 23 02:52:28 UTC 2026
# keep awake Mon Mar 23 07:21:32 UTC 2026
# keep awake Mon Mar 23 13:24:06 UTC 2026
# keep awake Mon Mar 23 19:04:05 UTC 2026
# keep awake Tue Mar 24 02:44:03 UTC 2026
# keep awake Tue Mar 24 07:13:14 UTC 2026
# keep awake Tue Mar 24 13:39:59 UTC 2026
# keep awake Tue Mar 24 19:14:06 UTC 2026
# keep awake Wed Mar 25 02:49:04 UTC 2026
# keep awake Wed Mar 25 07:11:39 UTC 2026
# keep awake Wed Mar 25 13:26:55 UTC 2026
# keep awake Wed Mar 25 19:04:44 UTC 2026
# keep awake Thu Mar 26 02:55:45 UTC 2026
# keep awake Thu Mar 26 07:19:49 UTC 2026
# keep awake Thu Mar 26 13:44:26 UTC 2026
# keep awake Thu Mar 26 19:17:44 UTC 2026
# keep awake Fri Mar 27 03:17:01 UTC 2026
# keep awake Fri Mar 27 07:18:36 UTC 2026
# keep awake Fri Mar 27 13:22:00 UTC 2026
# keep awake Fri Mar 27 19:04:12 UTC 2026
# keep awake Sat Mar 28 02:48:03 UTC 2026
# keep awake Sat Mar 28 07:06:53 UTC 2026
# keep awake Sat Mar 28 13:10:31 UTC 2026
# keep awake Sat Mar 28 18:52:43 UTC 2026
# keep awake Sun Mar 29 03:21:32 UTC 2026
# keep awake Sun Mar 29 07:13:54 UTC 2026
# keep awake Sun Mar 29 13:11:17 UTC 2026
# keep awake Sun Mar 29 18:54:26 UTC 2026
# keep awake Mon Mar 30 03:24:40 UTC 2026
# keep awake Mon Mar 30 07:53:53 UTC 2026
# keep awake Mon Mar 30 13:50:03 UTC 2026
# keep awake Mon Mar 30 19:09:46 UTC 2026
# keep awake Tue Mar 31 03:16:04 UTC 2026
# keep awake Tue Mar 31 07:38:33 UTC 2026
# keep awake Tue Mar 31 13:49:26 UTC 2026
# keep awake Tue Mar 31 19:11:35 UTC 2026
# keep awake Wed Apr  1 03:28:39 UTC 2026
# keep awake Wed Apr  1 07:44:18 UTC 2026
# keep awake Wed Apr  1 13:52:52 UTC 2026
# keep awake Wed Apr  1 19:13:12 UTC 2026
# keep awake Thu Apr  2 02:53:43 UTC 2026
# keep awake Thu Apr  2 07:23:16 UTC 2026
# keep awake Thu Apr  2 13:43:15 UTC 2026
# keep awake Thu Apr  2 19:05:05 UTC 2026
# keep awake Fri Apr  3 02:56:01 UTC 2026
# keep awake Fri Apr  3 07:20:02 UTC 2026
# keep awake Fri Apr  3 13:19:14 UTC 2026
# keep awake Fri Apr  3 18:57:49 UTC 2026
# keep awake Sat Apr  4 02:46:50 UTC 2026
# keep awake Sat Apr  4 07:09:53 UTC 2026
# keep awake Sat Apr  4 13:11:20 UTC 2026
# keep awake Sat Apr  4 18:53:51 UTC 2026
# keep awake Sun Apr  5 03:23:33 UTC 2026
# keep awake Sun Apr  5 07:17:12 UTC 2026
# keep awake Sun Apr  5 13:13:05 UTC 2026
# keep awake Sun Apr  5 18:55:13 UTC 2026
# keep awake Mon Apr  6 03:26:49 UTC 2026
# keep awake Mon Apr  6 07:55:11 UTC 2026
# keep awake Mon Apr  6 13:25:12 UTC 2026
# keep awake Mon Apr  6 19:11:06 UTC 2026
# keep awake Tue Apr  7 03:17:09 UTC 2026
# keep awake Tue Apr  7 07:42:11 UTC 2026
# keep awake Tue Apr  7 13:47:08 UTC 2026
# keep awake Tue Apr  7 19:14:36 UTC 2026
# keep awake Wed Apr  8 03:19:05 UTC 2026
# keep awake Wed Apr  8 07:45:39 UTC 2026
# keep awake Wed Apr  8 13:49:13 UTC 2026
# keep awake Wed Apr  8 19:26:59 UTC 2026
# keep awake Thu Apr  9 02:54:49 UTC 2026
# keep awake Thu Apr  9 07:48:20 UTC 2026
# keep awake Thu Apr  9 13:59:59 UTC 2026
# keep awake Thu Apr  9 19:18:36 UTC 2026
# keep awake Fri Apr 10 03:27:31 UTC 2026
# keep awake Fri Apr 10 07:51:56 UTC 2026
# keep awake Fri Apr 10 13:26:36 UTC 2026
# keep awake Fri Apr 10 19:04:10 UTC 2026
# keep awake Sat Apr 11 02:51:09 UTC 2026
# keep awake Sat Apr 11 07:10:01 UTC 2026
# keep awake Sat Apr 11 13:14:07 UTC 2026
# keep awake Sat Apr 11 18:56:24 UTC 2026
# keep awake Sun Apr 12 03:31:18 UTC 2026
# keep awake Sun Apr 12 07:24:13 UTC 2026
# keep awake Sun Apr 12 13:16:23 UTC 2026
# keep awake Sun Apr 12 19:00:12 UTC 2026
# keep awake Mon Apr 13 03:37:39 UTC 2026
# keep awake Mon Apr 13 08:14:50 UTC 2026
# keep awake Mon Apr 13 13:57:26 UTC 2026
# keep awake Mon Apr 13 19:22:11 UTC 2026
# keep awake Tue Apr 14 03:27:27 UTC 2026
# keep awake Tue Apr 14 07:57:35 UTC 2026
# keep awake Tue Apr 14 14:01:01 UTC 2026
# keep awake Tue Apr 14 19:25:45 UTC 2026
# keep awake Wed Apr 15 03:25:57 UTC 2026
# keep awake Wed Apr 15 07:58:30 UTC 2026
# keep awake Wed Apr 15 13:54:29 UTC 2026
# keep awake Wed Apr 15 19:37:18 UTC 2026
# keep awake Thu Apr 16 03:33:20 UTC 2026
# keep awake Thu Apr 16 07:58:05 UTC 2026
# keep awake Thu Apr 16 14:03:33 UTC 2026
# keep awake Thu Apr 16 19:24:53 UTC 2026
# keep awake Fri Apr 17 03:29:08 UTC 2026
# keep awake Fri Apr 17 07:59:42 UTC 2026
# keep awake Fri Apr 17 13:46:19 UTC 2026
# keep awake Fri Apr 17 19:10:54 UTC 2026
# keep awake Sat Apr 18 03:17:13 UTC 2026
# keep awake Sat Apr 18 07:19:17 UTC 2026
# keep awake Sat Apr 18 13:16:55 UTC 2026
# keep awake Sat Apr 18 19:01:12 UTC 2026
# keep awake Sun Apr 19 03:35:54 UTC 2026
# keep awake Sun Apr 19 07:39:41 UTC 2026
# keep awake Sun Apr 19 13:17:03 UTC 2026
# keep awake Sun Apr 19 19:01:22 UTC 2026
# keep awake Mon Apr 20 03:37:36 UTC 2026
# keep awake Mon Apr 20 08:21:06 UTC 2026
# keep awake Mon Apr 20 14:00:14 UTC 2026
# keep awake Mon Apr 20 19:17:00 UTC 2026
# keep awake Tue Apr 21 03:30:09 UTC 2026
# keep awake Tue Apr 21 08:03:25 UTC 2026
# keep awake Tue Apr 21 13:59:30 UTC 2026
# keep awake Tue Apr 21 19:21:45 UTC 2026
# keep awake Wed Apr 22 03:28:03 UTC 2026
# keep awake Wed Apr 22 07:59:55 UTC 2026
# keep awake Wed Apr 22 13:59:38 UTC 2026
# keep awake Wed Apr 22 19:23:37 UTC 2026
# keep awake Thu Apr 23 03:31:38 UTC 2026
# keep awake Thu Apr 23 08:05:37 UTC 2026
# keep awake Thu Apr 23 14:01:26 UTC 2026
# keep awake Thu Apr 23 19:22:42 UTC 2026
# keep awake Fri Apr 24 03:33:46 UTC 2026
# keep awake Fri Apr 24 08:14:32 UTC 2026
# keep awake Fri Apr 24 13:54:20 UTC 2026
# keep awake Fri Apr 24 19:03:50 UTC 2026
# keep awake Sat Apr 25 03:18:05 UTC 2026
# keep awake Sat Apr 25 07:38:56 UTC 2026
# keep awake Sat Apr 25 13:20:00 UTC 2026
# keep awake Sat Apr 25 19:03:41 UTC 2026
# keep awake Sun Apr 26 03:40:37 UTC 2026
# keep awake Sun Apr 26 07:48:25 UTC 2026
# keep awake Sun Apr 26 13:21:09 UTC 2026
# keep awake Sun Apr 26 19:05:08 UTC 2026
# keep awake Mon Apr 27 03:46:04 UTC 2026
# keep awake Mon Apr 27 08:36:22 UTC 2026
# keep awake Mon Apr 27 14:12:33 UTC 2026
# keep awake Mon Apr 27 19:42:00 UTC 2026
# keep awake Tue Apr 28 03:51:01 UTC 2026
# keep awake Tue Apr 28 08:34:10 UTC 2026
# keep awake Tue Apr 28 19:49:44 UTC 2026
# keep awake Wed Apr 29 03:48:24 UTC 2026
# keep awake Wed Apr 29 08:27:52 UTC 2026
# keep awake Wed Apr 29 14:19:45 UTC 2026
# keep awake Wed Apr 29 19:45:02 UTC 2026
# keep awake Thu Apr 30 03:49:19 UTC 2026
# keep awake Thu Apr 30 08:30:47 UTC 2026
# keep awake Thu Apr 30 14:14:01 UTC 2026
# keep awake Thu Apr 30 19:42:29 UTC 2026
# keep awake Fri May  1 04:00:34 UTC 2026
# keep awake Fri May  1 08:21:13 UTC 2026
# keep awake Fri May  1 13:42:31 UTC 2026
# keep awake Fri May  1 19:20:39 UTC 2026
# keep awake Sat May  2 03:33:56 UTC 2026
# keep awake Sat May  2 07:53:03 UTC 2026
# keep awake Sat May  2 13:26:32 UTC 2026
# keep awake Sat May  2 19:10:06 UTC 2026
# keep awake Sun May  3 03:55:38 UTC 2026
# keep awake Sun May  3 08:08:18 UTC 2026
# keep awake Sun May  3 13:26:40 UTC 2026
# keep awake Sun May  3 19:09:24 UTC 2026
# keep awake Mon May  4 03:52:58 UTC 2026
# keep awake Mon May  4 08:38:29 UTC 2026
# keep awake Mon May  4 14:19:37 UTC 2026
# keep awake Mon May  4 19:48:49 UTC 2026
# keep awake Tue May  5 03:34:20 UTC 2026
# keep awake Tue May  5 08:20:19 UTC 2026
# keep awake Tue May  5 14:12:42 UTC 2026
# keep awake Tue May  5 19:43:53 UTC 2026
# keep awake Wed May  6 03:49:48 UTC 2026
# keep awake Wed May  6 08:36:05 UTC 2026
# keep awake Wed May  6 14:31:50 UTC 2026
# keep awake Wed May  6 19:54:45 UTC 2026
# keep awake Thu May  7 03:48:40 UTC 2026
# keep awake Thu May  7 08:43:19 UTC 2026
# keep awake Thu May  7 14:33:41 UTC 2026
# keep awake Thu May  7 19:48:32 UTC 2026
# keep awake Fri May  8 03:39:26 UTC 2026
# keep awake Fri May  8 07:49:22 UTC 2026
# keep awake Fri May  8 14:05:11 UTC 2026
# keep awake Fri May  8 19:38:38 UTC 2026
# keep awake Sat May  9 03:40:31 UTC 2026
# keep awake Sat May  9 08:03:41 UTC 2026
# keep awake Sat May  9 13:40:53 UTC 2026
# keep awake Sat May  9 19:13:13 UTC 2026
# keep awake Sun May 10 03:57:29 UTC 2026
# keep awake Sun May 10 08:17:56 UTC 2026
# keep awake Sun May 10 13:43:46 UTC 2026
# keep awake Sun May 10 19:14:06 UTC 2026
# keep awake Mon May 11 04:08:16 UTC 2026
# keep awake Mon May 11 09:53:34 UTC 2026
# keep awake Mon May 11 15:27:37 UTC 2026
# keep awake Mon May 11 19:56:47 UTC 2026
# keep awake Tue May 12 03:52:04 UTC 2026
# keep awake Tue May 12 08:48:43 UTC 2026
# keep awake Tue May 12 14:39:28 UTC 2026
# keep awake Tue May 12 20:02:01 UTC 2026
# keep awake Wed May 13 04:00:30 UTC 2026
# keep awake Wed May 13 08:52:36 UTC 2026
# keep awake Wed May 13 14:50:05 UTC 2026
# keep awake Wed May 13 20:03:19 UTC 2026
# keep awake Thu May 14 03:59:21 UTC 2026
# keep awake Thu May 14 08:45:31 UTC 2026
# keep awake Thu May 14 14:29:38 UTC 2026
# keep awake Thu May 14 19:54:38 UTC 2026
# keep awake Fri May 15 04:04:26 UTC 2026
# keep awake Fri May 15 09:18:12 UTC 2026
# keep awake Fri May 15 14:23:40 UTC 2026
# keep awake Fri May 15 19:47:29 UTC 2026
# keep awake Sat May 16 03:46:51 UTC 2026
# keep awake Sat May 16 08:12:12 UTC 2026
# keep awake Sat May 16 13:47:08 UTC 2026
# keep awake Sat May 16 19:15:17 UTC 2026
# keep awake Sun May 17 04:04:34 UTC 2026
# keep awake Sun May 17 08:28:33 UTC 2026
# keep awake Sun May 17 13:46:44 UTC 2026
# keep awake Sun May 17 19:18:14 UTC 2026
# keep awake Mon May 18 04:17:32 UTC 2026
# keep awake Mon May 18 10:18:11 UTC 2026
# keep awake Mon May 18 15:48:33 UTC 2026
# keep awake Mon May 18 19:53:11 UTC 2026
# keep awake Tue May 19 04:12:18 UTC 2026
# keep awake Tue May 19 09:58:03 UTC 2026
# keep awake Tue May 19 15:39:29 UTC 2026
# keep awake Tue May 19 19:58:57 UTC 2026
# keep awake Wed May 20 04:13:03 UTC 2026
# keep awake Wed May 20 09:48:09 UTC 2026
# keep awake Wed May 20 15:40:40 UTC 2026
# keep awake Wed May 20 20:25:38 UTC 2026
# keep awake Thu May 21 04:21:31 UTC 2026
# keep awake Thu May 21 09:54:48 UTC 2026
# keep awake Thu May 21 15:42:24 UTC 2026
# keep awake Thu May 21 20:01:56 UTC 2026
# keep awake Fri May 22 04:16:43 UTC 2026
# keep awake Fri May 22 09:40:10 UTC 2026
# keep awake Fri May 22 14:52:11 UTC 2026
# keep awake Fri May 22 19:57:36 UTC 2026
# keep awake Sat May 23 03:55:02 UTC 2026
# keep awake Sat May 23 08:27:07 UTC 2026
# keep awake Sat May 23 13:52:19 UTC 2026
# keep awake Sat May 23 19:20:45 UTC 2026
# keep awake Sun May 24 04:15:08 UTC 2026
# keep awake Sun May 24 08:39:04 UTC 2026
# keep awake Sun May 24 13:49:12 UTC 2026
# keep awake Sun May 24 19:24:41 UTC 2026
# keep awake Mon May 25 04:26:18 UTC 2026
# keep awake Mon May 25 10:22:15 UTC 2026
# keep awake Mon May 25 15:21:19 UTC 2026
# keep awake Mon May 25 19:50:35 UTC 2026
# keep awake Tue May 26 04:12:23 UTC 2026
# keep awake Tue May 26 10:10:02 UTC 2026
# keep awake Tue May 26 15:57:12 UTC 2026
# keep awake Tue May 26 20:20:28 UTC 2026
# keep awake Wed May 27 04:25:37 UTC 2026
# keep awake Wed May 27 10:07:43 UTC 2026
# keep awake Wed May 27 16:00:53 UTC 2026
# keep awake Wed May 27 20:25:33 UTC 2026
# keep awake Thu May 28 04:15:39 UTC 2026
# keep awake Thu May 28 10:15:54 UTC 2026
# keep awake Thu May 28 16:15:30 UTC 2026
# keep awake Thu May 28 20:32:11 UTC 2026
# keep awake Fri May 29 04:17:48 UTC 2026
# keep awake Fri May 29 10:06:00 UTC 2026
# keep awake Fri May 29 15:58:28 UTC 2026
# keep awake Fri May 29 20:33:35 UTC 2026
# keep awake Sat May 30 04:02:48 UTC 2026
# keep awake Sat May 30 08:35:09 UTC 2026
# keep awake Sat May 30 13:55:07 UTC 2026
# keep awake Sat May 30 19:35:52 UTC 2026
# keep awake Sun May 31 04:38:20 UTC 2026
# keep awake Sun May 31 08:55:34 UTC 2026
# keep awake Sun May 31 14:00:20 UTC 2026
# keep awake Sun May 31 19:26:00 UTC 2026
# keep awake Mon Jun  1 04:59:45 UTC 2026
# keep awake Mon Jun  1 11:48:38 UTC 2026
# keep awake Mon Jun  1 17:54:32 UTC 2026
# keep awake Mon Jun  1 21:39:47 UTC 2026
# keep awake Tue Jun  2 04:46:22 UTC 2026
# keep awake Tue Jun  2 10:47:35 UTC 2026
# keep awake Tue Jun  2 16:46:26 UTC 2026
# keep awake Tue Jun  2 21:16:59 UTC 2026
# keep awake Wed Jun  3 04:56:20 UTC 2026
# keep awake Wed Jun  3 11:13:11 UTC 2026
# keep awake Wed Jun  3 17:02:57 UTC 2026
# keep awake Wed Jun  3 21:22:46 UTC 2026
# keep awake Thu Jun  4 04:49:16 UTC 2026
# keep awake Thu Jun  4 10:06:18 UTC 2026
# keep awake Thu Jun  4 15:31:42 UTC 2026
# keep awake Thu Jun  4 20:12:15 UTC 2026
# keep awake Fri Jun  5 04:24:07 UTC 2026
# keep awake Fri Jun  5 10:03:16 UTC 2026
# keep awake Fri Jun  5 15:18:50 UTC 2026
# keep awake Fri Jun  5 20:02:40 UTC 2026
# keep awake Sat Jun  6 04:06:46 UTC 2026
# keep awake Sat Jun  6 08:41:40 UTC 2026
# keep awake Sat Jun  6 13:58:39 UTC 2026
# keep awake Sat Jun  6 19:40:19 UTC 2026
# keep awake Sun Jun  7 04:43:09 UTC 2026
# keep awake Sun Jun  7 09:26:15 UTC 2026
# keep awake Sun Jun  7 14:08:48 UTC 2026
# keep awake Sun Jun  7 19:41:07 UTC 2026
# keep awake Mon Jun  8 04:47:55 UTC 2026
# keep awake Mon Jun  8 11:09:50 UTC 2026
# keep awake Mon Jun  8 16:07:23 UTC 2026
# keep awake Mon Jun  8 20:26:21 UTC 2026
# keep awake Tue Jun  9 04:10:26 UTC 2026
# keep awake Tue Jun  9 09:56:48 UTC 2026
# keep awake Tue Jun  9 15:18:59 UTC 2026
# keep awake Tue Jun  9 20:13:59 UTC 2026
# keep awake Wed Jun 10 04:21:31 UTC 2026
# keep awake Wed Jun 10 10:14:11 UTC 2026
# keep awake Wed Jun 10 15:53:20 UTC 2026
# keep awake Wed Jun 10 20:39:35 UTC 2026
# keep awake Thu Jun 11 04:43:47 UTC 2026
# keep awake Thu Jun 11 10:49:16 UTC 2026
# keep awake Thu Jun 11 16:13:59 UTC 2026
# keep awake Thu Jun 11 20:29:50 UTC 2026
# keep awake Fri Jun 12 04:46:24 UTC 2026
# keep awake Fri Jun 12 10:23:55 UTC 2026
# keep awake Fri Jun 12 15:30:26 UTC 2026
# keep awake Fri Jun 12 20:19:25 UTC 2026
# keep awake Sat Jun 13 04:25:10 UTC 2026
# keep awake Sat Jun 13 09:26:12 UTC 2026
# keep awake Sat Jun 13 14:15:06 UTC 2026
# keep awake Sat Jun 13 19:45:15 UTC 2026
# keep awake Sun Jun 14 04:52:32 UTC 2026
# keep awake Sun Jun 14 09:47:55 UTC 2026
# keep awake Sun Jun 14 14:18:54 UTC 2026
# keep awake Sun Jun 14 19:46:07 UTC 2026
# keep awake Mon Jun 15 05:09:19 UTC 2026
# keep awake Mon Jun 15 12:31:57 UTC 2026
# keep awake Mon Jun 15 21:11:57 UTC 2026
# keep awake Tue Jun 16 05:11:56 UTC 2026
# keep awake Tue Jun 16 11:24:48 UTC 2026
# keep awake Tue Jun 16 17:08:59 UTC 2026
# keep awake Tue Jun 16 21:11:24 UTC 2026
# keep awake Wed Jun 17 04:54:38 UTC 2026
# keep awake Wed Jun 17 11:09:01 UTC 2026
# keep awake Wed Jun 17 15:52:31 UTC 2026
# keep awake Wed Jun 17 20:22:53 UTC 2026
# keep awake Thu Jun 18 04:45:49 UTC 2026
# keep awake Thu Jun 18 10:40:09 UTC 2026
# keep awake Thu Jun 18 15:41:14 UTC 2026
# keep awake Thu Jun 18 20:33:07 UTC 2026
# keep awake Fri Jun 19 05:05:28 UTC 2026
# keep awake Fri Jun 19 10:48:43 UTC 2026
# keep awake Fri Jun 19 15:34:10 UTC 2026
# keep awake Fri Jun 19 19:56:31 UTC 2026
# keep awake Sat Jun 20 04:18:04 UTC 2026
# keep awake Sat Jun 20 09:28:57 UTC 2026
# keep awake Sat Jun 20 14:19:25 UTC 2026
# keep awake Sat Jun 20 19:46:51 UTC 2026
# keep awake Sun Jun 21 05:01:58 UTC 2026
# keep awake Sun Jun 21 09:59:05 UTC 2026
# keep awake Sun Jun 21 14:25:01 UTC 2026
# keep awake Sun Jun 21 19:53:15 UTC 2026
# keep awake Mon Jun 22 05:10:12 UTC 2026
# keep awake Mon Jun 22 12:17:19 UTC 2026
# keep awake Mon Jun 22 20:57:14 UTC 2026
# keep awake Tue Jun 23 04:07:42 UTC 2026
# keep awake Tue Jun 23 09:55:49 UTC 2026
# keep awake Tue Jun 23 15:15:10 UTC 2026
# keep awake Tue Jun 23 20:14:14 UTC 2026
# keep awake Wed Jun 24 04:10:37 UTC 2026
# keep awake Wed Jun 24 09:42:35 UTC 2026
# keep awake Wed Jun 24 14:46:03 UTC 2026
# keep awake Wed Jun 24 19:55:51 UTC 2026
# keep awake Thu Jun 25 04:11:05 UTC 2026
# keep awake Thu Jun 25 09:34:42 UTC 2026
# keep awake Thu Jun 25 14:45:20 UTC 2026
# keep awake Thu Jun 25 20:10:56 UTC 2026
# keep awake Fri Jun 26 04:17:40 UTC 2026
# keep awake Fri Jun 26 09:40:11 UTC 2026
# keep awake Fri Jun 26 14:38:03 UTC 2026
# keep awake Fri Jun 26 19:59:20 UTC 2026
# keep awake Sat Jun 27 04:04:10 UTC 2026
# keep awake Sat Jun 27 08:44:55 UTC 2026
# keep awake Sat Jun 27 13:57:07 UTC 2026
# keep awake Sat Jun 27 19:38:03 UTC 2026
# keep awake Sun Jun 28 04:26:14 UTC 2026
# keep awake Sun Jun 28 09:21:39 UTC 2026
# keep awake Sun Jun 28 14:03:40 UTC 2026
# keep awake Sun Jun 28 19:39:29 UTC 2026
# keep awake Mon Jun 29 04:45:38 UTC 2026
# keep awake Mon Jun 29 11:14:49 UTC 2026
# keep awake Mon Jun 29 15:59:15 UTC 2026
# keep awake Mon Jun 29 20:07:31 UTC 2026
# keep awake Tue Jun 30 04:12:10 UTC 2026
# keep awake Tue Jun 30 09:53:09 UTC 2026
# keep awake Tue Jun 30 14:33:00 UTC 2026
# keep awake Tue Jun 30 20:06:20 UTC 2026
# keep awake Wed Jul  1 04:40:53 UTC 2026
# keep awake Wed Jul  1 10:01:52 UTC 2026
# keep awake Wed Jul  1 14:49:54 UTC 2026
# keep awake Wed Jul  1 20:03:35 UTC 2026
# keep awake Thu Jul  2 04:07:49 UTC 2026
# keep awake Thu Jul  2 09:24:50 UTC 2026
# keep awake Thu Jul  2 14:19:20 UTC 2026
# keep awake Thu Jul  2 19:41:36 UTC 2026
# keep awake Fri Jul  3 03:54:16 UTC 2026
# keep awake Fri Jul  3 09:29:06 UTC 2026
# keep awake Fri Jul  3 14:23:51 UTC 2026
# keep awake Fri Jul  3 19:38:55 UTC 2026
# keep awake Sat Jul  4 03:47:34 UTC 2026
# keep awake Sat Jul  4 08:46:03 UTC 2026
# keep awake Sat Jul  4 13:48:40 UTC 2026
# keep awake Sat Jul  4 19:23:29 UTC 2026
# keep awake Sun Jul  5 04:06:07 UTC 2026
# keep awake Sun Jul  5 09:15:38 UTC 2026
# keep awake Sun Jul  5 13:54:52 UTC 2026
# keep awake Sun Jul  5 19:33:51 UTC 2026
# keep awake Mon Jul  6 04:12:49 UTC 2026
# keep awake Mon Jul  6 10:51:48 UTC 2026
# keep awake Mon Jul  6 15:48:18 UTC 2026
# keep awake Mon Jul  6 20:06:34 UTC 2026
# keep awake Tue Jul  7 04:02:50 UTC 2026
# keep awake Tue Jul  7 09:53:36 UTC 2026
# keep awake Tue Jul  7 14:52:17 UTC 2026
# keep awake Tue Jul  7 20:05:31 UTC 2026
# keep awake Wed Jul  8 03:28:53 UTC 2026
# keep awake Wed Jul  8 08:38:30 UTC 2026
# keep awake Wed Jul  8 14:30:01 UTC 2026
# keep awake Wed Jul  8 19:43:40 UTC 2026
# keep awake Thu Jul  9 04:01:34 UTC 2026
# keep awake Thu Jul  9 09:51:02 UTC 2026
# keep awake Thu Jul  9 15:25:03 UTC 2026
# keep awake Thu Jul  9 19:55:21 UTC 2026
# keep awake Fri Jul 10 03:57:09 UTC 2026
# keep awake Fri Jul 10 09:47:04 UTC 2026
# keep awake Fri Jul 10 14:42:28 UTC 2026
# keep awake Fri Jul 10 19:41:59 UTC 2026
# keep awake Sat Jul 11 03:23:54 UTC 2026
# keep awake Sat Jul 11 08:05:08 UTC 2026
# keep awake Sat Jul 11 13:41:09 UTC 2026
# keep awake Sat Jul 11 19:13:47 UTC 2026
# keep awake Sun Jul 12 03:35:23 UTC 2026
# keep awake Sun Jul 12 08:26:18 UTC 2026
# keep awake Sun Jul 12 13:41:54 UTC 2026
# keep awake Sun Jul 12 19:14:21 UTC 2026
# keep awake Mon Jul 13 03:37:57 UTC 2026
# keep awake Mon Jul 13 09:39:26 UTC 2026
# keep awake Mon Jul 13 14:49:47 UTC 2026
# keep awake Mon Jul 13 19:38:12 UTC 2026
# keep awake Tue Jul 14 03:13:22 UTC 2026
# keep awake Tue Jul 14 08:19:20 UTC 2026
# keep awake Tue Jul 14 14:00:00 UTC 2026
# keep awake Tue Jul 14 19:35:42 UTC 2026
# keep awake Wed Jul 15 03:12:36 UTC 2026
# keep awake Wed Jul 15 08:25:06 UTC 2026
# keep awake Wed Jul 15 13:56:40 UTC 2026
# keep awake Wed Jul 15 19:21:19 UTC 2026
# keep awake Thu Jul 16 03:19:23 UTC 2026
# keep awake Thu Jul 16 08:23:48 UTC 2026
# keep awake Thu Jul 16 14:08:41 UTC 2026
# keep awake Thu Jul 16 19:21:37 UTC 2026
# keep awake Fri Jul 17 03:22:24 UTC 2026
# keep awake Fri Jul 17 08:20:14 UTC 2026
# keep awake Fri Jul 17 13:52:35 UTC 2026
# keep awake Fri Jul 17 19:19:42 UTC 2026
# keep awake Sat Jul 18 03:11:36 UTC 2026
# keep awake Sat Jul 18 08:02:05 UTC 2026
# keep awake Sat Jul 18 13:35:56 UTC 2026
# keep awake Sat Jul 18 19:12:37 UTC 2026
# keep awake Sun Jul 19 03:31:55 UTC 2026
# keep awake Sun Jul 19 08:27:44 UTC 2026
# keep awake Sun Jul 19 13:38:36 UTC 2026
# keep awake Sun Jul 19 19:14:50 UTC 2026
# keep awake Mon Jul 20 03:46:29 UTC 2026
# keep awake Mon Jul 20 09:27:19 UTC 2026
# keep awake Mon Jul 20 14:22:08 UTC 2026
# keep awake Mon Jul 20 19:49:45 UTC 2026
# keep awake Tue Jul 21 03:25:41 UTC 2026
# keep awake Tue Jul 21 08:38:33 UTC 2026
# keep awake Tue Jul 21 14:08:10 UTC 2026
# keep awake Tue Jul 21 19:40:42 UTC 2026
# keep awake Wed Jul 22 03:24:15 UTC 2026
# keep awake Wed Jul 22 08:37:48 UTC 2026
# keep awake Wed Jul 22 14:11:02 UTC 2026
# keep awake Wed Jul 22 19:35:40 UTC 2026
# keep awake Thu Jul 23 03:30:19 UTC 2026
# keep awake Thu Jul 23 08:39:34 UTC 2026
# keep awake Thu Jul 23 14:20:44 UTC 2026
# keep awake Thu Jul 23 19:35:09 UTC 2026
# keep awake Fri Jul 24 03:25:26 UTC 2026
# keep awake Fri Jul 24 08:35:27 UTC 2026
# keep awake Fri Jul 24 13:59:22 UTC 2026
# keep awake Fri Jul 24 19:38:38 UTC 2026
# keep awake Sat Jul 25 03:23:12 UTC 2026
# keep awake Sat Jul 25 08:16:06 UTC 2026
# keep awake Sat Jul 25 13:49:21 UTC 2026
# keep awake Sat Jul 25 19:17:29 UTC 2026
# keep awake Sun Jul 26 03:36:42 UTC 2026
# keep awake Sun Jul 26 08:34:23 UTC 2026
# keep awake Sun Jul 26 13:42:30 UTC 2026
# keep awake Sun Jul 26 19:20:27 UTC 2026
# keep awake Mon Jul 27 03:46:22 UTC 2026
# keep awake Mon Jul 27 10:02:21 UTC 2026
# keep awake Mon Jul 27 14:50:31 UTC 2026
# keep awake Mon Jul 27 19:44:07 UTC 2026
# keep awake Tue Jul 28 03:17:01 UTC 2026
# keep awake Tue Jul 28 08:44:56 UTC 2026
# keep awake Tue Jul 28 14:24:43 UTC 2026
# keep awake Tue Jul 28 19:41:28 UTC 2026
# keep awake Wed Jul 29 03:19:35 UTC 2026
# keep awake Wed Jul 29 08:49:36 UTC 2026
# keep awake Wed Jul 29 14:23:19 UTC 2026
# keep awake Wed Jul 29 19:26:17 UTC 2026
# keep awake Thu Jul 30 02:52:35 UTC 2026
# keep awake Thu Jul 30 08:38:52 UTC 2026
# keep awake Thu Jul 30 14:20:24 UTC 2026
# keep awake Thu Jul 30 19:42:02 UTC 2026
# keep awake Fri Jul 31 03:35:07 UTC 2026
# keep awake Fri Jul 31 09:10:33 UTC 2026
# keep awake Fri Jul 31 14:23:03 UTC 2026
# keep awake Fri Jul 31 19:43:41 UTC 2026
# keep awake Sat Aug  1 03:34:46 UTC 2026
# keep awake Sat Aug  1 08:29:03 UTC 2026
# keep awake Sat Aug  1 13:41:56 UTC 2026
# keep awake Sat Aug  1 19:18:35 UTC 2026
# keep awake Sun Aug  2 03:34:32 UTC 2026
# keep awake Sun Aug  2 08:31:49 UTC 2026
# keep awake Sun Aug  2 13:42:03 UTC 2026
# keep awake Sun Aug  2 19:19:36 UTC 2026
# keep awake Mon Aug  3 03:36:54 UTC 2026
# keep awake Mon Aug  3 09:58:03 UTC 2026
# keep awake Mon Aug  3 14:54:27 UTC 2026
# keep awake Mon Aug  3 19:45:45 UTC 2026
# keep awake Tue Aug  4 03:20:07 UTC 2026
# keep awake Tue Aug  4 08:47:58 UTC 2026
# keep awake Tue Aug  4 14:30:31 UTC 2026
# keep awake Tue Aug  4 19:45:00 UTC 2026
# keep awake Wed Aug  5 03:15:40 UTC 2026
# keep awake Wed Aug  5 08:45:21 UTC 2026
# keep awake Wed Aug  5 14:23:43 UTC 2026
# keep awake Wed Aug  5 19:44:09 UTC 2026
# keep awake Thu Aug  6 03:18:25 UTC 2026
# keep awake Thu Aug  6 08:46:45 UTC 2026
# keep awake Thu Aug  6 14:26:30 UTC 2026
# keep awake Fri Aug  7 00:07:42 UTC 2026
# keep awake Fri Aug  7 07:25:47 UTC 2026
# keep awake Fri Aug  7 13:17:24 UTC 2026
# keep awake Fri Aug  7 19:05:13 UTC 2026
# keep awake Sat Aug  8 02:03:14 UTC 2026
# keep awake Sat Aug  8 07:02:45 UTC 2026
# keep awake Sat Aug  8 13:01:22 UTC 2026
# keep awake Sat Aug  8 18:47:24 UTC 2026
# keep awake Sun Aug  9 02:11:12 UTC 2026
# keep awake Sun Aug  9 07:05:20 UTC 2026
# keep awake Sun Aug  9 13:03:56 UTC 2026
# keep awake Sun Aug  9 18:50:27 UTC 2026
# keep awake Mon Aug 10 02:19:55 UTC 2026
# keep awake Mon Aug 10 07:52:44 UTC 2026
# keep awake Mon Aug 10 13:22:43 UTC 2026
# keep awake Mon Aug 10 19:04:19 UTC 2026
# keep awake Tue Aug 11 02:09:54 UTC 2026
# keep awake Tue Aug 11 07:17:24 UTC 2026
# keep awake Tue Aug 11 13:19:25 UTC 2026
# keep awake Tue Aug 11 19:10:16 UTC 2026
# keep awake Wed Aug 12 02:27:52 UTC 2026
# keep awake Wed Aug 12 07:41:02 UTC 2026
# keep awake Wed Aug 12 13:23:15 UTC 2026
# keep awake Wed Aug 12 19:09:37 UTC 2026
# keep awake Thu Aug 13 02:29:36 UTC 2026
# keep awake Thu Aug 13 07:42:56 UTC 2026
# keep awake Thu Aug 13 13:25:02 UTC 2026
# keep awake Thu Aug 13 19:10:36 UTC 2026
# keep awake Fri Aug 14 02:27:53 UTC 2026
# keep awake Fri Aug 14 07:40:33 UTC 2026
# keep awake Fri Aug 14 13:19:59 UTC 2026
# keep awake Fri Aug 14 19:02:03 UTC 2026
# keep awake Sat Aug 15 01:38:30 UTC 2026
# keep awake Sat Aug 15 06:48:12 UTC 2026
# keep awake Sat Aug 15 12:49:22 UTC 2026
# keep awake Sat Aug 15 18:37:39 UTC 2026
# keep awake Sun Aug 16 01:45:36 UTC 2026
# keep awake Sun Aug 16 06:49:54 UTC 2026
# keep awake Sun Aug 16 12:51:01 UTC 2026
# keep awake Sun Aug 16 18:36:44 UTC 2026
# keep awake Mon Aug 17 01:43:17 UTC 2026
# keep awake Mon Aug 17 07:06:47 UTC 2026
# keep awake Mon Aug 17 12:55:30 UTC 2026
# keep awake Mon Aug 17 18:48:26 UTC 2026
# keep awake Tue Aug 18 01:38:42 UTC 2026
# keep awake Tue Aug 18 06:53:42 UTC 2026
# keep awake Tue Aug 18 12:57:06 UTC 2026
# keep awake Tue Aug 18 18:47:00 UTC 2026
# keep awake Wed Aug 19 01:40:47 UTC 2026
# keep awake Wed Aug 19 06:53:56 UTC 2026
# keep awake Wed Aug 19 12:58:09 UTC 2026
# keep awake Wed Aug 19 18:42:46 UTC 2026
# keep awake Thu Aug 20 01:39:18 UTC 2026
# keep awake Thu Aug 20 06:55:15 UTC 2026
# keep awake Thu Aug 20 13:00:35 UTC 2026
# keep awake Thu Aug 20 18:48:56 UTC 2026
# keep awake Fri Aug 21 01:44:19 UTC 2026
# keep awake Fri Aug 21 06:56:30 UTC 2026
# keep awake Fri Aug 21 12:59:29 UTC 2026
# keep awake Fri Aug 21 18:45:29 UTC 2026
# keep awake Sat Aug 22 01:37:37 UTC 2026
# keep awake Sat Aug 22 06:50:14 UTC 2026
# keep awake Sat Aug 22 12:50:43 UTC 2026
# keep awake Sat Aug 22 18:38:28 UTC 2026
# keep awake Sun Aug 23 01:47:30 UTC 2026
# keep awake Sun Aug 23 06:51:13 UTC 2026
# keep awake Sun Aug 23 12:51:52 UTC 2026
# keep awake Sun Aug 23 18:37:40 UTC 2026
# keep awake Mon Aug 24 01:45:06 UTC 2026
# keep awake Mon Aug 24 07:10:03 UTC 2026
# keep awake Mon Aug 24 13:02:11 UTC 2026
# keep awake Mon Aug 24 18:49:31 UTC 2026
# keep awake Tue Aug 25 01:39:34 UTC 2026
# keep awake Tue Aug 25 06:57:47 UTC 2026
# keep awake Tue Aug 25 12:58:20 UTC 2026
# keep awake Tue Aug 25 18:47:42 UTC 2026
# keep awake Wed Aug 26 01:45:02 UTC 2026
# keep awake Wed Aug 26 06:58:25 UTC 2026
# keep awake Wed Aug 26 13:03:53 UTC 2026
# keep awake Wed Aug 26 19:54:19 UTC 2026
# keep awake Thu Aug 27 08:48:02 UTC 2026
# keep awake Thu Aug 27 22:08:09 UTC 2026
# keep awake Fri Aug 28 11:00:54 UTC 2026
# keep awake Fri Aug 28 22:07:16 UTC 2026
# keep awake Sat Aug 29 06:46:12 UTC 2026
# keep awake Sat Aug 29 16:35:15 UTC 2026
# keep awake Sat Aug 29 20:48:08 UTC 2026
# keep awake Sun Aug 30 04:58:54 UTC 2026
# keep awake Sun Aug 30 11:43:07 UTC 2026
# keep awake Sun Aug 30 16:30:35 UTC 2026
# keep awake Sun Aug 30 20:53:01 UTC 2026
# keep awake Mon Aug 31 05:05:00 UTC 2026
# keep awake Mon Aug 31 13:21:36 UTC 2026
# keep awake Mon Aug 31 22:42:56 UTC 2026
# keep awake Tue Sep  1 04:42:13 UTC 2026
# keep awake Tue Sep  1 11:28:59 UTC 2026
# keep awake Tue Sep  1 16:26:07 UTC 2026
# keep awake Tue Sep  1 20:54:16 UTC 2026
# keep awake Wed Sep  2 04:03:56 UTC 2026
# keep awake Wed Sep  2 11:03:10 UTC 2026
# keep awake Wed Sep  2 16:24:16 UTC 2026
# keep awake Wed Sep  2 20:53:40 UTC 2026
# keep awake Thu Sep  3 04:02:24 UTC 2026
# keep awake Thu Sep  3 11:00:52 UTC 2026
# keep awake Thu Sep  3 16:14:33 UTC 2026
# keep awake Thu Sep  3 20:51:50 UTC 2026
# keep awake Fri Sep  4 04:05:35 UTC 2026
# keep awake Fri Sep  4 11:01:35 UTC 2026
# keep awake Fri Sep  4 16:09:52 UTC 2026
# keep awake Fri Sep  4 20:38:31 UTC 2026
# keep awake Sat Sep  5 04:01:47 UTC 2026
# keep awake Sat Sep  5 10:23:46 UTC 2026
# keep awake Sat Sep  5 14:58:21 UTC 2026
# keep awake Sat Sep  5 20:12:19 UTC 2026
# keep awake Sun Sep  6 04:09:30 UTC 2026
# keep awake Sun Sep  6 10:42:38 UTC 2026
