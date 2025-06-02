import streamlit as st
import pandas as pd
import plotly.express as px
from bertopic import BERTopic
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import nltk
import collections # Untuk Counter
import emoji # Untuk identifikasi dan demojize emoji

# Unduh stopwords NLTK (jika belum ada) dan data emoji (opsional, tapi baik untuk kelengkapan)
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords

# Import untuk Coherence Score BERTopic (jika tersedia)
try:
    from bertopic.evaluation import coherence_model # Mencoba import yang lebih baru/spesifik
except ImportError:
    # Fallback atau penanganan jika tidak ada
    try:
        from bertopic.metrics import Coherence # Mungkin path lama atau berbeda
        coherence_model = Coherence # Untuk konsistensi nama variabel
    except ImportError:
        coherence_model = None # Jika tidak tersedia sama sekali

from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="Dashboard Analisis Sentimen & Topik", layout="wide")
st.title("📊 Dashboard Analisis Sentimen dan Topik Komentar YouTube")

# --------------------------
# Upload dan Baca File
# --------------------------
uploaded_file = st.file_uploader("Unggah file Excel atau CSV Anda", type=["xlsx", "csv"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            xls = pd.ExcelFile(uploaded_file)
            sheet_names = xls.sheet_names
            if not sheet_names:
                st.error("❌ File Excel tidak mengandung sheet.")
                st.stop()
            selected_sheet = st.selectbox("Pilih sheet dari file Excel:", sheet_names)
            if selected_sheet:
                df = pd.read_excel(xls, sheet_name=selected_sheet)
            else:
                st.error("⚠️ Silakan pilih sheet untuk diproses.")
                st.stop()
    except Exception as e:
        st.error(f"❌ Gagal memuat file: {e}")
        st.stop()

    # Tampilkan pratinjau data (opsional, bisa di-expander)
    if st.checkbox("Tampilkan pratinjau data"):
        st.dataframe(df.head())

    # Pilih kolom komentar, sentimen, like, dan waktu
    st.sidebar.header("Pengaturan Kolom Data")
    available_columns = df.columns.tolist()
    
    text_column = st.sidebar.selectbox("Pilih kolom berisi teks komentar:", available_columns, index=0 if available_columns else -1)
    sentimen_column = st.sidebar.selectbox("Pilih kolom berisi label sentimen:", available_columns, index=1 if len(available_columns) > 1 else -1)
    like_column = st.sidebar.selectbox("Pilih kolom berisi jumlah 'like':", available_columns, index=2 if len(available_columns) > 2 else -1)
    time_column = st.sidebar.selectbox("Pilih kolom berisi 'timestamp':", available_columns, index=3 if len(available_columns) > 3 else -1)
    
    # Asumsi 'sentimenCount' adalah kolom yang sudah ada atau akan dibuat.
    # Jika tidak ada, Anda mungkin perlu logika untuk menghitungnya atau membuatnya opsional.
    # Untuk saat ini, kita anggap ini wajib ada sesuai kode awal.
    sentimen_count_col_name = "sentimenCount"


    # Cek kolom wajib
    required_columns_to_check = []
    if text_column: required_columns_to_check.append(text_column)
    if sentimen_column: required_columns_to_check.append(sentimen_column)
    if like_column: required_columns_to_check.append(like_column)
    if time_column: required_columns_to_check.append(time_column)
    # Tambahkan 'sentimenCount' jika memang ini kolom yang ada di data asli
    if sentimen_count_col_name in df.columns:
         required_columns_to_check.append(sentimen_count_col_name)
    else:
        st.warning(f"Kolom '{sentimen_count_col_name}' tidak ditemukan. Beberapa visualisasi mungkin tidak berfungsi atau memerlukan pembuatan kolom ini.")


    all_selected_cols_valid = text_column and sentimen_column and like_column and time_column
    
    if not all_selected_cols_valid:
        st.warning("⚠️ Harap pilih semua kolom yang diperlukan dari sidebar.")
        st.stop()

    missing_cols = [col for col in required_columns_to_check if col not in df.columns]
    if missing_cols:
        st.error(f"❌ Kolom berikut tidak ditemukan di file Anda: {', '.join(missing_cols)}")
        st.stop()
    else:
        # Preprocessing dasar
        df[text_column] = df[text_column].astype(str).fillna("")
        df[sentimen_column] = df[sentimen_column].fillna("unknown").astype(str).str.lower()
        df[sentimen_column] = df[sentimen_column].replace({
            "negative": "negatif",
            "positive": "positif",
            "neutral": "netral"
        })
        df[time_column] = pd.to_datetime(df[time_column], errors="coerce")
        df["date"] = df[time_column].dt.date

        # Pastikan kolom 'like_column' numerik
        if like_column in df.columns:
            df[like_column] = pd.to_numeric(df[like_column], errors='coerce').fillna(0)
        
        # Pastikan kolom 'sentimenCount' numerik jika ada, atau buat jika tidak ada dan dibutuhkan
        if sentimen_count_col_name in df.columns:
             df[sentimen_count_col_name] = pd.to_numeric(df[sentimen_count_col_name], errors='coerce').fillna(0)
        # else:
            # Jika 'sentimenCount' TIDAK ADA dan Anda ingin menghitungnya berdasarkan jumlah kemunculan sentimen:
            # df[sentimen_count_col_name] = df.groupby(sentimen_column)[sentimen_column].transform('count')


        st.success("✅ Data berhasil dimuat dan diproses awal!")

        # --- PEMBUATAN STOPWORDS EMOJI DINAMIS ---
        dynamic_emoji_stopwords = set()
        if not df[text_column].empty:
            st.info("🔎 Memindai emoji untuk stopwords dinamis...")
            # Gabungkan teks unik untuk efisiensi, pastikan tidak ada NaN
            all_text_for_emoji_scan = " ".join(df[text_column].astype(str).dropna().unique())
            
            unique_emoji_chars = set()
            if all_text_for_emoji_scan.strip(): # Pastikan string tidak kosong
                emoji_data_list = emoji.emoji_list(all_text_for_emoji_scan)
                for item in emoji_data_list:
                    unique_emoji_chars.add(item['emoji'])

            if unique_emoji_chars:
                for char_emoji in unique_emoji_chars:
                    demojized_text = emoji.demojize(char_emoji, language='en')
                    cleaned_demojized_text = demojized_text.strip(':').replace('_', ' ').lower()
                    words_from_emoji = re.findall(r'\b\w+\b', cleaned_demojized_text)
                    for word in words_from_emoji:
                        if len(word) > 1: 
                            dynamic_emoji_stopwords.add(word)
                st.caption(f"💬 Stopwords dari emoji dinamis (kata komponen): {len(dynamic_emoji_stopwords)} kata. Contoh: {', '.join(sorted(list(dynamic_emoji_stopwords))[:10])}{'...' if len(dynamic_emoji_stopwords) > 10 else ''}")
            else:
                st.caption("Tidak ada emoji yang terdeteksi untuk stopwords dinamis.")
        # --- AKHIR PEMBUATAN STOPWORDS EMOJI DINAMIS ---

        # --- DEFINISI DAN KOMBINASI STOPWORDS ---
        default_stopwords_id = set(stopwords.words('indonesian'))
        custom_stopwords_list = {
            'yang', 'itu', 'dan', 'di', 'ke', 'dari', 'pada', 'untuk', 'oleh', 'dengan',
            'saat', 'kemarin', 'nanti', 'ada', 'adalah', 'baik', 'buruk', 'dll',
            'saya', 'kamu', 'dia', 'mereka', 'kita', 'kami', 'anda', 'juga',
            'ini', 'itu', 'nya', 'loh', 'sih', 'deh', 'mah', 'ga', 'gak', 'enggak', 'nggak',
            'tapi', 'namun', 'atau', 'ataupun', 'sebab', 'karena', 'jika', 'kalau',
            'supaya', 'biar', 'agar', 'ketika', 'setelah', 'sebelum', 'sampai', 'hingga',
            'satu', 'dua', 'tiga', 'empat', 'lima', 'enam', 'tujuh', 'delapan',
            'semua', 'beberapa', 'banyak', 'sedikit', 'orang', 'rumah', 'kota',
            'hal', 'masalah', 'sesuatu', 'merupakan', 'menjadi', 'terjadi',
            'berada', 'sedang', 'telah', 'yg', 'ya', 'tuh', 'ngga', 'bekasi', 'walikota',
            'wali', 'pak', 'daerah', 'bikin', 'tolong', 'lg', 'udah', 'org', 'semoga', 'klo',
            'jgn', 'udh', 'dah', 'karna', 'br', 'gk', 'gua', 'gue', 'gw', 'sy', 'aq',
            'video', 'channel', 'komen', 'komentar', 'konten', 'youtube', 'youtuber',
            'admin', 'kak', 'bang', 'mas', 'mbak', 'gan', 'bro', 'sis',
            'nonton', 'lihat', 'suka', 'banget', 'kali', 'aja', 'saja', 'pula',
            'belum', 'sudah', 'akan', 'selalu', 'sering', 'kadang', 'mungkin',
            'kenapa', 'gimana', 'bagaimana', 'apa', 'siapa', 'kapan', 'mana',
            'kok', 'sih', 'dong', 'kan', 'nih', 'pas', 'biar', 'ayo', 'mari',
            'bgt', 'tdk', 'gaes', 'guys', 'guys', 'mntp', 'mantap', 'keren',
            'partai', 'politik', 'pemerintah', 'dpr', 'presiden', 'pilpres', 'pemilu', 'face', 'with', 'tears', 'joy' # contoh domain politik
        }
        stop_words_final = default_stopwords_id.union(custom_stopwords_list).union(dynamic_emoji_stopwords)
        st.caption(f"Total stopwords yang digunakan (termasuk NLTK, kustom, dan dinamis dari emoji): {len(stop_words_final)} kata.")
        # --- AKHIR DEFINISI STOPWORDS ---
        # --------------------------
        # Eksplorasi Data Awal
        # --------------------------
        with st.expander("📊 Eksplorasi Data Awal", expanded=True):
            col1_eda, col2_eda = st.columns(2)

            # Histogram Like Count
            with col1_eda:
                st.subheader("📌 Distribusi Like Komentar")
                if like_column in df.columns and not df[like_column].empty:
                    fig_like = px.histogram(df, x=like_column, nbins=50, title="Distribusi Jumlah Like")
                    st.plotly_chart(fig_like, use_container_width=True)
                else:
                    st.info("Kolom 'like' tidak valid untuk histogram.")

            # Sebaran Sentimen
            with col2_eda:
                st.subheader("📊 Sebaran Sentimen")
                if sentimen_column in df.columns and not df[sentimen_column].empty:
                    fig_sent = px.histogram(df, x=sentimen_column, title="Sebaran Kategori Sentimen")
                    st.plotly_chart(fig_sent, use_container_width=True)
                else:
                    st.info("Kolom 'sentimen' tidak valid untuk histogram.")

            # Rata-rata Like per Sentimen
            st.subheader("👍 Rata-rata Like per Sentimen")
            if like_column in df.columns and sentimen_column in df.columns:
                like_avg = df.groupby(sentimen_column)[like_column].mean().reset_index().sort_values(by=like_column, ascending=False)
                fig_bar_like_sent = px.bar(like_avg, x=sentimen_column, y=like_column, text_auto='.2s', title="Rata-Rata Like per Kategori Sentimen")
                st.plotly_chart(fig_bar_like_sent, use_container_width=True)

            # Rata-rata SentimenCount per Sentimen (jika kolomnya ada)
            if sentimen_count_col_name in df.columns:
                st.subheader(f"🔢 Rata-rata '{sentimen_count_col_name}' per Sentimen")
                count_avg = df.groupby(sentimen_column)[sentimen_count_col_name].mean().reset_index().sort_values(by=sentimen_count_col_name, ascending=False)
                fig_count_sent = px.bar(count_avg, x=sentimen_column, y=sentimen_count_col_name, text_auto='.2s', title=f"Rata-Rata '{sentimen_count_col_name}' per Kategori Sentimen")
                st.plotly_chart(fig_count_sent, use_container_width=True)

            # Tren Sentimen Seiring Waktu
            st.subheader("📅 Tren Sentimen Seiring Waktu")
            if 'date' in df.columns and sentimen_column in df.columns:
                df_non_na_date = df.dropna(subset=['date']) # Pastikan tidak ada NaT di 'date'
                if not df_non_na_date.empty:
                    time_series = df_non_na_date.groupby(["date", sentimen_column]).size().reset_index(name="jumlah_komentar")
                    fig_line_trend = px.line(time_series, x="date", y="jumlah_komentar", color=sentimen_column, markers=True, title="Tren Jumlah Komentar per Sentimen Seiring Waktu")
                    st.plotly_chart(fig_line_trend, use_container_width=True)
                else:
                    st.info("Tidak ada data waktu yang valid untuk tren sentimen.")
            
            # --- WordCloud dan Top 20 Kata Komentar Negatif ---
            st.subheader("💬 Analisis Teks Komentar Negatif")
            df_negatif_eda = df[df[sentimen_column] == "negatif"]

            if not df_negatif_eda.empty:
                negative_comments_text_list = df_negatif_eda[text_column].dropna().astype(str).tolist()
                
                cleaned_negative_texts_for_wc = []
                all_negative_words_for_counter = []

                for comment_text in negative_comments_text_list:
                    # Pembersihan lebih detail: hapus URL, mention, hashtag sebelum tanda baca umum
                    comment_text = re.sub(r"http\S+|www\S+|https\S+", '', comment_text, flags=re.MULTILINE) # Hapus URL
                    comment_text = re.sub(r'\@\w+', '', comment_text) # Hapus mention
                    comment_text = re.sub(r'#\w+', '', comment_text) # Hapus hashtag
                    comment_text = re.sub(r'[^\w\s]', '', comment_text.lower()) # Hapus tanda baca & lowercase
                    
                    words = [word for word in comment_text.split() if word not in stop_words_final and len(word) > 2] # Filter stopwords & kata pendek
                    cleaned_negative_texts_for_wc.append(" ".join(words))
                    all_negative_words_for_counter.extend(words)

                full_negative_text_cleaned = " ".join(cleaned_negative_texts_for_wc)

                # WordCloud
                st.markdown("☁️ **WordCloud Komentar Negatif** (setelah stopwords removal)")
                if full_negative_text_cleaned.strip():
                    try:
                        wc = WordCloud(width=800, height=400, background_color='white', collocations=False).generate(full_negative_text_cleaned)
                        fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
                        ax_wc.imshow(wc, interpolation='bilinear')
                        ax_wc.axis("off")
                        st.pyplot(fig_wc)
                    except Exception as e_wc:
                        st.error(f"Gagal membuat WordCloud: {e_wc}")
                else:
                    st.info("Tidak ada kata tersisa untuk WordCloud komentar negatif setelah filtering.")

                # Visualisasi 20 Kata Teratas
                st.markdown("🔝 **20 Kata Teratas Komentar Negatif** (setelah stopwords removal)")
                if all_negative_words_for_counter:
                    word_counts_negative = collections.Counter(all_negative_words_for_counter)
                    most_common_words_negative = word_counts_negative.most_common(20)

                    if most_common_words_negative:
                        df_most_common_negative = pd.DataFrame(most_common_words_negative, columns=['Kata', 'Frekuensi'])
                        fig_top_words_neg = px.bar(df_most_common_negative, x='Frekuensi', y='Kata', orientation='h',
                                                   title="20 Kata Teratas dalam Komentar Negatif", text_auto=True)
                        fig_top_words_neg.update_layout(yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig_top_words_neg, use_container_width=True)
                    else:
                        st.info("Tidak ada kata yang cukup untuk menampilkan 20 kata teratas komentar negatif.")
                else:
                    st.info("Tidak ada teks negatif yang tersisa untuk dianalisis kata teratas.")
            else:
                st.info("Tidak ada komentar negatif untuk dianalisis dalam EDA.")

        # ----------------------------------------------------
        # Analisis BERTopic (HANYA untuk SENTIMEN NEGATIF)
        # ----------------------------------------------------
        st.header("🔬 Analisis Topik Komentar Negatif dengan BERTopic")

        df_negatif_topic = df[df[sentimen_column] == "negatif"].copy() # Gunakan .copy() untuk menghindari SettingWithCopyWarning

        if len(df_negatif_topic) < 5: # BERTopic butuh cukup dokumen
            st.warning("⚠️ Jumlah komentar negatif terlalu sedikit (kurang dari 5) untuk analisis topik yang bermakna.")
        else:
            if st.button("🔍 Proses dan Analisis Topik dari Komentar Negatif (BERTopic)"):
                with st.spinner("Melatih model BERTopic pada komentar negatif... Ini mungkin memerlukan waktu."):
                    # Ambil teks komentar negatif
                    docs_negatif = df_negatif_topic[text_column].astype(str).dropna().tolist()
                    
                    # Preprocessing tambahan sebelum ke BERTopic (opsional, tapi bisa membantu)
                    # Hapus URL, mention, hashtag, angka, dan karakter non-alfanumerik kecuali spasi, lalu lowercase
                    # stopwords akan ditangani oleh BERTopic atau CountVectorizer di dalamnya jika dikonfigurasi
                    processed_docs_negatif = []
                    for doc in docs_negatif:
                        doc = re.sub(r"http\S+|www\S+|https\S+", '', doc, flags=re.MULTILINE)
                        doc = re.sub(r'\@\w+', '', doc)
                        doc = re.sub(r'#\w+', '', doc)
                        doc = re.sub(r'\d+', '', doc) # Hapus angka
                        doc = re.sub(r'[^\w\s]', '', doc) # Hapus tanda baca
                        doc = doc.lower().strip()
                        processed_docs_negatif.append(doc)
                    
                    # Hapus string kosong yang mungkin muncul setelah preprocessing
                    processed_docs_negatif = [doc for doc in processed_docs_negatif if doc]

                    if not processed_docs_negatif or len(processed_docs_negatif) < 5:
                        st.error("Tidak ada dokumen yang valid tersisa untuk BERTopic setelah preprocessing atau jumlahnya terlalu sedikit.")
                        st.stop()

                    try:
                        # Inisialisasi BERTopic
                        # Anda bisa menambahkan CountVectorizer dengan stopwords_final jika ingin BERTopic juga menggunakannya
                        # from sklearn.feature_extraction.text import CountVectorizer
                        # vectorizer_model = CountVectorizer(stop_words=list(stop_words_final))
                        # topic_model = BERTopic(language="multilingual", verbose=True, vectorizer_model=vectorizer_model)
                        # Untuk kesederhanaan, kita gunakan setting default dengan "indonesian" jika data mayoritas B.Indonesia
                        # Jika campuran, "multilingual" lebih aman.
                        
                        topic_model = BERTopic(language="indonesian", verbose=True, min_topic_size=3, nr_topics="auto") # Sesuaikan min_topic_size
                        topics, probs = topic_model.fit_transform(processed_docs_negatif)
                        
                        st.success("✅ Model BERTopic berhasil dilatih!")

                        # --- Perhitungan Coherence Score ---
                        st.subheader("💯 Coherence Score Topik")
                        try:
                            # Metode 1: Menggunakan fungsi coherence_model dari bertopic.evaluation (jika tersedia)
                            if coherence_model and hasattr(topic_model, 'topic_representations_'): # Cek atribut yang dibutuhkan
                                score_cv = coherence_model(topic_model, documents=pd.Series(processed_docs_negatif), topics=topics, coherence='c_v')
                                st.metric(label="Coherence Score (c_v)", value=f"{score_cv:.4f}")
                                score_umass = coherence_model(topic_model, documents=pd.Series(processed_docs_negatif), topics=topics, coherence='u_mass')
                                st.metric(label="Coherence Score (u_mass)", value=f"{score_umass:.4f}")
                            else:
                                raise ImportError("Fungsi coherence_model tidak tersedia atau model tidak siap.")
                        except Exception as e_coh_internal:
                            st.warning(f"⚠️ Gagal menghitung Coherence dengan metode internal BERTopic ({e_coh_internal}). Mencoba metode manual dengan Gensim...")
                            try:
                                tokenized_docs_for_coherence = [doc.split() for doc in processed_docs_negatif]
                                dictionary_gensim = Dictionary(tokenized_docs_for_coherence)
                                corpus_gensim = [dictionary_gensim.doc2bow(doc) for doc in tokenized_docs_for_coherence]
                                
                                # Dapatkan topik dari BERTopic
                                keywords_per_topic_gensim = []
                                # Pastikan hanya topik valid (bukan outlier -1)
                                valid_topic_ids = sorted([tid for tid in topic_model.get_topics().keys() if tid != -1])

                                for topic_id in valid_topic_ids:
                                    topic_words_scores = topic_model.get_topic(topic_id)
                                    if topic_words_scores: # Pastikan tidak None
                                        keywords_per_topic_gensim.append([word for word, score in topic_words_scores[:10]]) # Ambil top 10 kata

                                if keywords_per_topic_gensim:
                                    cm_cv = CoherenceModel(topics=keywords_per_topic_gensim, texts=tokenized_docs_for_coherence, dictionary=dictionary_gensim, coherence='c_v')
                                    coherence_cv_manual = cm_cv.get_coherence()
                                    st.metric(label="Coherence Score (c_v - manual Gensim)", value=f"{coherence_cv_manual:.4f}")

                                    cm_umass = CoherenceModel(topics=keywords_per_topic_gensim, texts=tokenized_docs_for_coherence, dictionary=dictionary_gensim, coherence='u_mass')
                                    coherence_umass_manual = cm_umass.get_coherence()
                                    st.metric(label="Coherence Score (u_mass - manual Gensim)", value=f"{coherence_umass_manual:.4f}")
                                else:
                                    st.info("Tidak ada topik yang diekstrak untuk perhitungan koherensi manual.")
                            except Exception as e_coh_manual:
                                st.error(f"❌ Gagal menghitung Coherence Score secara manual: {e_coh_manual}")
                        
                        # --- Tampilkan Informasi Topik ---
                        df_topic_info = topic_model.get_topic_info()
                        # Filter outlier topic (-1) jika tidak ingin ditampilkan
                        df_topic_info_filtered = df_topic_info[df_topic_info["Topic"] != -1].head(15) # Tampilkan top 15 topik

                        if not df_topic_info_filtered.empty:
                            st.subheader("📈 Jumlah Komentar Negatif per Topik Teridentifikasi")
                            # Ganti nama default dari BERTopic jika perlu
                            df_topic_info_filtered["DisplayName"] = df_topic_info_filtered["Name"].apply(lambda x: x[x.find("_")+1:].replace("_", " "))
                            
                            fig_topic_bar = px.bar(df_topic_info_filtered, 
                                                   x="DisplayName",  # Gunakan DisplayName atau Name
                                                   y="Count", 
                                                   text_auto=True,
                                                   title="Distribusi Komentar per Topik (Top 15)",
                                                   labels={"DisplayName": "Topik", "Count": "Jumlah Komentar"})
                            fig_topic_bar.update_layout(xaxis_title="Representasi Topik", yaxis_title="Jumlah Komentar")
                            st.plotly_chart(fig_topic_bar, use_container_width=True)

                            st.subheader("🧠 Rangkuman Topik Negatif & WordCloud per Topik")
                            for index, row in df_topic_info_filtered.iterrows():
                                topic_id = row["Topic"]
                                label = row["DisplayName"] # atau row["Name"]
                                keywords_scores = topic_model.get_topic(topic_id) # List of tuples (word, score)
                                
                                if keywords_scores is None: continue # Skip jika topik tidak punya keywords

                                keywords_only = [word for word, score in keywords_scores[:10]] # Ambil 10 kata kunci teratas

                                with st.expander(f"Topik #{topic_id}: {label} (Jumlah: {row['Count']})"):
                                    st.markdown(f"**🔑 Kata Kunci Utama:** {', '.join(keywords_only)}")

                                    # Contoh Komentar Representatif
                                    try:
                                        rep_docs = topic_model.get_representative_docs(topic_id)
                                        if rep_docs:
                                            st.markdown("**💬 Contoh Komentar Negatif Representatif:**")
                                            for doc_sample in rep_docs[:3]: # Tampilkan 3 contoh
                                                st.markdown(f"> {doc_sample}")
                                        else:
                                            st.markdown("> Tidak ada contoh komentar representatif untuk topik ini.")
                                    except Exception as e_rep_doc:
                                        st.caption(f"Info: Tidak bisa mengambil contoh dokumen representatif ({e_rep_doc})")


                                    # WordCloud Kata Kunci Topik (menggunakan stop_words_final)
                                    st.markdown("**☁️ WordCloud Kata Kunci Topik** (setelah stopwords global removal):")
                                    # Buat dictionary frekuensi dari keywords_scores untuk WordCloud
                                    # Filter lagi dengan stop_words_final untuk WordCloud spesifik topik ini
                                    word_freq_for_wc_topic = {
                                        word: score for word, score in keywords_scores 
                                        if word.lower() not in stop_words_final and len(word) > 1
                                    }
                                    
                                    if word_freq_for_wc_topic:
                                        try:
                                            wc_topic = WordCloud(width=600, height=300, background_color='white', collocations=False).generate_from_frequencies(word_freq_for_wc_topic)
                                            fig_wc_topic, ax_wc_topic = plt.subplots(figsize=(8, 4))
                                            ax_wc_topic.imshow(wc_topic, interpolation='bilinear')
                                            ax_wc_topic.axis("off")
                                            st.pyplot(fig_wc_topic)
                                        except Exception as e_wc_topic:
                                            st.error(f"Gagal membuat WordCloud untuk topik {topic_id}: {e_wc_topic}")
                                    else:
                                        st.info("Tidak ada kata kunci tersisa untuk WordCloud topik ini setelah filtering stopwords global.")
                        else:
                            st.info("Tidak ada topik yang signifikan ditemukan oleh BERTopic selain outlier.")

                        # Tombol Download Data Negatif dengan Label Topik
                        # Tambahkan kolom topik ke df_negatif_topic
                        # Pastikan panjang 'topics' sesuai dengan 'processed_docs_negatif'
                        # Jika ada pemfilteran di processed_docs_negatif, perlu pemetaan balik yang hati-hati
                        # Cara aman: Buat DataFrame baru dari processed_docs_negatif dan topics
                        
                        if len(processed_docs_negatif) == len(topics):
                            df_negatif_with_topics_assigned = pd.DataFrame({'ProcessedText': processed_docs_negatif, 'AssignedTopicID': topics})
                            
                            # Jika Anda ingin menggabungkan kembali dengan data asli (df_negatif_topic)
                            # Ini memerlukan indeks yang cocok atau cara lain untuk join.
                            # Untuk sederhana, kita unduh processed text dan topiknya.
                            # Atau, jika df_negatif_topic tidak difilter setelah docs_negatif dibuat, panjangnya harus sama.
                            if len(df_negatif_topic) == len(topics):
                                df_negatif_topic['BERTopic_ID'] = topics
                                csv_data_bertopic = df_negatif_topic.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="💾 Unduh Data Komentar Negatif dengan ID Topik BERTopic",
                                    data=csv_data_bertopic,
                                    file_name='komentar_negatif_dengan_topik_bertopic.csv',
                                    mime='text/csv',
                                )
                            else:
                                st.caption("Tidak dapat mencocokkan topik dengan DataFrame asli secara langsung karena perbedaan panjang setelah preprocessing. Unduhan akan berisi teks terproses dan ID topik.")
                                csv_data_processed_bertopic = df_negatif_with_topics_assigned.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                label="💾 Unduh Teks Terproses & ID Topik BERTopic",
                                data=csv_data_processed_bertopic,
                                file_name='teks_negatif_terproses_dengan_id_topik.csv',
                                mime='text/csv',
                            )

                        else:
                             st.warning("Perbedaan panjang antara dokumen yang diproses dan hasil topik. Tidak dapat membuat file unduhan topik.")


                    except Exception as e_bertopic:
                        st.error(f"❌ Terjadi kesalahan saat menjalankan BERTopic: {e_bertopic}")
                        st.exception(e_bertopic) # Tampilkan traceback untuk debug
else:
    # Pesan jika tidak ada file yang diunggah
    st.info("👋 Selamat datang! Silakan unggah file data (Excel atau CSV) untuk memulai analisis.")

st.markdown("---")
st.markdown("Dibuat dengan Streamlit & Python | Analisis Sentimen & Topik Komentar")
