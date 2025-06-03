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
    local_model_path = "model/fine-tuned-indobert" # <--- INI BAGIAN YANG PENTING
    
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


st.sidebar.header("Input Data")
youtube_url_input = st.sidebar.text_input("Masukkan URL Video YouTube:")
analyze_button = st.sidebar.button("🚀 Ambil & Analisis Komentar")

# Inisialisasi session state untuk menyimpan DataFrame
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'raw_df' not in st.session_state: # Simpan juga raw df jika perlu
    st.session_state.raw_df = None

# Variabel untuk kolom yang akan digunakan di dashboard
# Ini sekarang ditetapkan karena sumber data dan pemrosesan sudah baku
TEXT_COLUMN_FOR_ANALYSIS = "cleanedText" # Teks yang sudah diproses untuk analisis
RAW_TEXT_COLUMN = "Text"
SENTIMENT_COLUMN = "sentimen"
LIKE_COUNT_COLUMN = "likeCount"
TIME_COLUMN = "Time"
SENTIMENT_COUNT_COLUMN = "sentimenCount"


if analyze_button and youtube_url_input:
    if not sentiment_model_pipeline:
        st.error("Model sentimen tidak berhasil dimuat. Tidak dapat melanjutkan analisis.")
    else:
        with st.spinner("Mengambil komentar dari YouTube... Ini mungkin memerlukan waktu ⏳"):
            video_id = None
            try:
                if "v=" in youtube_url_input:
                    video_id = youtube_url_input.split("v=")[-1].split("&")[0]
                elif "youtu.be/" in youtube_url_input:
                    video_id = youtube_url_input.split("youtu.be/")[-1].split("?")[0]
                else:
                    st.error("Format URL YouTube tidak valid.")
            except Exception:
                st.error("Format URL YouTube tidak valid.")

            if video_id:
                youtube_api_key = get_youtube_api_key()
                df_youtube = fetch_youtube_comments(video_id, youtube_api_key)
                st.session_state.raw_df = df_youtube # Simpan raw data

                if df_youtube is not None and not df_youtube.empty:
                    st.success(f"Berhasil mengambil {len(df_youtube)} komentar!")
                    
                    current_df = df_youtube.copy()

                    with st.spinner("Melakukan pra-pemrosesan teks... ⚙️"):
                        current_df[TEXT_COLUMN_FOR_ANALYSIS] = current_df[RAW_TEXT_COLUMN].apply(light_preprocess_text)
                    
                    with st.spinner("Melakukan analisis sentimen dengan model fine-tuned Anda... 🧠"):
                        current_df[SENTIMENT_COLUMN] = current_df[TEXT_COLUMN_FOR_ANALYSIS].apply(classify_sentiment_text)
                    
                    # Konversi tipe data dan perhitungan kolom lain
                    current_df[TIME_COLUMN] = pd.to_datetime(current_df[TIME_COLUMN], errors="coerce")
                    current_df["date"] = current_df[TIME_COLUMN].dt.date
                    current_df[LIKE_COUNT_COLUMN] = pd.to_numeric(current_df[LIKE_COUNT_COLUMN], errors="coerce").fillna(0).astype(int)
                    current_df[SENTIMENT_COUNT_COLUMN] = current_df[LIKE_COUNT_COLUMN] + 1 # Sesuai logika Colab
                    
                    # Ganti label sentimen jika perlu (misal dari model 'LABEL_0' -> 'negatif')
                    # Sesuaikan ini berdasarkan output aktual model Anda jika berbeda dari 'positive', 'negative', 'neutral'
                    # current_df[SENTIMENT_COLUMN] = current_df[SENTIMENT_COLUMN].replace({
                    #     "label_0": "negatif", 
                    #     "label_1": "netral",
                    #     "label_2": "positif"
                    # })
                    # Pastikan semua label sentimen menjadi lowercase untuk konsistensi
                    current_df[SENTIMENT_COLUMN] = current_df[SENTIMENT_COLUMN].str.lower()


                    st.session_state.processed_df = current_df # Simpan DataFrame yang sudah diproses
                    st.success("Pra-pemrosesan dan analisis sentimen selesai!")
                    if st.checkbox("Tampilkan pratinjau data hasil analisis", True):
                        st.dataframe(current_df[[RAW_TEXT_COLUMN, TEXT_COLUMN_FOR_ANALYSIS, SENTIMENT_COLUMN, LIKE_COUNT_COLUMN, SENTIMENT_COUNT_COLUMN]].head())
                elif df_youtube is None:
                    st.error("Gagal mengambil komentar. Periksa API Key atau URL.")
                else: # df_youtube is empty
                    st.warning("Tidak ada komentar yang ditemukan untuk video ini.")
                    st.session_state.processed_df = pd.DataFrame() # Set ke df kosong
elif analyze_button and not youtube_url_input:
    st.warning("Harap masukkan URL YouTube terlebih dahulu.")

# --- Mulai bagian Dashboard jika data sudah ada ---
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
        'nya', 'sih', 'dong', 'kok', 'deh', 'mah', 'tuh', 'nih', # partikel
        'url', 'user', # dari placeholder preprocessing
        # Tambahkan kata-kata dari slang_lexicon yang mungkin ingin tetap dihilangkan meskipun sudah dinormalisasi
        # Tambahkan kata-kata hasil demojize emoji yang umum jika dynamic_emoji_stopwords tidak menangkapnya (karena teks sudah diproses)
        'face', 'tears', 'joy', 'red', 'heart', 'blue', 'black', 'white', 'green', 'yellow', # komponen umum emoji
        'hand', 'hands', 'eyes', 'smiling', 'loudly', 'crying', 'rolling', 'floor', 'laughing',
        'thinking', 'pondering', 'pleading', 'point', 'right', 'left', 'up', 'down', 'backhand', 'index',
        'ok', 'flexed', 'biceps', 'folded', 'clapping', 'thumbs',
        'satu', 'dua', 'tiga', 'empat', 'lima', 'enam', 'tujuh', 'delapan', 'sembilan', 'sepuluh', # angka
        'januari', 'februari', 'maret', 'april', 'mei', 'juni', 'juli', 'agustus', 'september', 'oktober', 'november', 'desember', # bulan
        'senin', 'selasa', 'rabu', 'kamis', 'jumat', 'sabtu', 'minggu', # hari
         # Kata dari kode dashboard sebelumnya
        'yang', 'itu', 'dan', 'di', 'ke', 'dari', 'pada', 'untuk', 'oleh', 'dengan',
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
        'partai', 'politik', 'pemerintah', 'dpr', 'presiden', 'pilpres', 'pemilu'
    }
    stop_words_final = default_stopwords_id.union(custom_stopwords_list).union(dynamic_emoji_stopwords)
    st.sidebar.caption(f"Total stopwords (NLTK, kustom, emoji dinamis): {len(stop_words_final)}")


# --- EDA ---
    with st.expander("📊 Eksplorasi Data Awal", expanded=True):
        # (Kode EDA dari dashboard sebelumnya, disesuaikan untuk menggunakan df_dashboard dan nama kolom yang benar)
        # Contoh:
        col1_eda, col2_eda = st.columns(2)
        with col1_eda:
            st.subheader(f"📌 Distribusi '{LIKE_COUNT_COLUMN}' Komentar")
            fig_like = px.histogram(df_dashboard, x=LIKE_COUNT_COLUMN, nbins=50)
            st.plotly_chart(fig_like, use_container_width=True)
        with col2_eda:
            st.subheader(f"📊 Sebaran '{SENTIMENT_COLUMN}'")
            fig_sent = px.histogram(df_dashboard, x=SENTIMENT_COLUMN, color=SENTIMENT_COLUMN,
                                    color_discrete_map={"positif":"green", "negatif":"red", "netral":"grey"})
            st.plotly_chart(fig_sent, use_container_width=True)

        st.subheader(f"👍 Rata-rata '{LIKE_COUNT_COLUMN}' per Sentimen")
        like_avg = df_dashboard.groupby(SENTIMENT_COLUMN)[LIKE_COUNT_COLUMN].mean().reset_index().sort_values(by=LIKE_COUNT_COLUMN, ascending=False)
        fig_bar_like_sent = px.bar(like_avg, x=SENTIMENT_COLUMN, y=LIKE_COUNT_COLUMN, text_auto='.2s',
                                   color=SENTIMENT_COLUMN, color_discrete_map={"positif":"green", "negatif":"red", "netral":"grey"})
        st.plotly_chart(fig_bar_like_sent, use_container_width=True)

        st.subheader(f"🔢 Rata-rata '{SENTIMENT_COUNT_COLUMN}' per Sentimen")
        count_avg = df_dashboard.groupby(SENTIMENT_COLUMN)[SENTIMENT_COUNT_COLUMN].mean().reset_index().sort_values(by=SENTIMENT_COUNT_COLUMN, ascending=False)
        fig_count_sent = px.bar(count_avg, x=SENTIMENT_COLUMN, y=SENTIMENT_COUNT_COLUMN, text_auto='.2s',
                                color=SENTIMENT_COLUMN, color_discrete_map={"positif":"green", "negatif":"red", "netral":"grey"})
        st.plotly_chart(fig_count_sent, use_container_width=True)
        
        st.subheader("📅 Tren Sentimen Seiring Waktu")
        df_non_na_date_dashboard = df_dashboard.dropna(subset=['date'])
        if not df_non_na_date_dashboard.empty:
            time_series = df_non_na_date_dashboard.groupby(["date", SENTIMENT_COLUMN]).size().reset_index(name="jumlah_komentar")
            fig_line_trend = px.line(time_series, x="date", y="jumlah_komentar", color=SENTIMENT_COLUMN, markers=True,
                                     color_discrete_map={"positif":"green", "negatif":"red", "netral":"grey"})
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
                most_common_words_negative = word_counts_negative.most_common(20)
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
            # (Kode visualisasi topik dari dashboard sebelumnya)
            df_topic_info_bt = active_topic_model.get_topic_info()
            df_topic_info_filtered_bt = df_topic_info_bt[df_topic_info_bt["Topic"] != -1].head(15)

            if not df_topic_info_filtered_bt.empty:
                st.subheader("📈 Topik Komentar Negatif (BERTopic)")
                df_topic_info_filtered_bt["DisplayName"] = df_topic_info_filtered_bt["Name"].apply(lambda x: x[x.find("_")+1:].replace("_", " "))
                fig_topic_bar_bt = px.bar(df_topic_info_filtered_bt, x="DisplayName", y="Count", text_auto=True)
                st.plotly_chart(fig_topic_bar_bt, use_container_width=True)

                st.subheader("🧠 Rangkuman & WordCloud per Topik (BERTopic)")
                for _, row_bt in df_topic_info_filtered_bt.iterrows():
                    topic_id_bt_disp = row_bt["Topic"]
                    label_bt = row_bt["DisplayName"]
                    keywords_scores_bt = active_topic_model.get_topic(topic_id_bt_disp)
                    if keywords_scores_bt is None: continue
                    keywords_only_bt = [word for word, score in keywords_scores_bt[:10]]
                    with st.expander(f"Topik #{topic_id_bt_disp}: {label_bt} (Jumlah: {row_bt['Count']})"):
                        st.markdown(f"**🔑 Kata Kunci:** {', '.join(keywords_only_bt)}")
                        # WordCloud (menggunakan stop_words_final)
                        word_freq_wc_topic_bt = {w:s for w,s in keywords_scores_bt if w.lower() not in stop_words_final and len(w)>1}
                        if word_freq_wc_topic_bt:
                            wc_topic_bt = WordCloud(width=600, height=300, background_color='white', collocations=False).generate_from_frequencies(word_freq_wc_topic_bt)
                            fig_wc_bt, ax_wc_bt = plt.subplots(figsize=(8,4))
                            ax_wc_bt.imshow(wc_topic_bt, interpolation='bilinear'); ax_wc_bt.axis("off"); st.pyplot(fig_wc_bt)
                        else: st.info("Tidak ada kata kunci tersisa untuk WordCloud topik ini setelah filter stopwords.")
            else:
                st.info("Tidak ada topik signifikan ditemukan oleh BERTopic (selain outlier).")
            
            # --- Tombol Download Data dengan Topik BERTopic ---
            try:
                current_topics_bt, _ = active_topic_model.transform(processed_docs_for_bertopic_analysis)
                if len(df_negatif_bertopic) == len(current_topics_bt):
                    df_negatif_bertopic['BERTopic_ID'] = current_topics_bt
                    csv_data_bt_dl = df_negatif_bertopic.to_csv(index=False).encode('utf-8')
                    st.download_button(label="💾 Unduh Data Negatif dengan ID Topik BERTopic", data=csv_data_bt_dl,
                                       file_name='komentar_negatif_youtube_bertopic.csv', mime='text/csv')
            except Exception as e_transform_dl_bt:
                st.error(f"Gagal melakukan transformasi BERTopic untuk unduhan: {e_transform_dl_bt}")
    # ... (else jika model BERTopic tidak aktif) ...
elif st.session_state.processed_df is not None and st.session_state.processed_df.empty:
    st.info("Tidak ada data komentar yang diproses untuk ditampilkan di dashboard.")
else:
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
