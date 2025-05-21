import streamlit as st
import pandas as pd
import plotly.express as px
from bertopic import BERTopic
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

st.set_page_config(page_title="Dashboard Analisis BERTopic", layout="wide")
st.title("📊 Dashboard Analisis Sentimen dan Topik Komentar YouTube")

# --------------------------
# Upload dan Baca File
# --------------------------
uploaded_file = st.file_uploader("Unggah file Excel atau CSV", type=["xlsx", "csv"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        xls = pd.ExcelFile(uploaded_file)
        sheet_names = xls.sheet_names
        selected_sheet = st.selectbox("Pilih sheet:", sheet_names)
        df = pd.read_excel(xls, sheet_name=selected_sheet)

    # Tetapkan nama kolom secara otomatis
    text_column = st.selectbox("Pilih kolom komentar:", df.columns)
    sentimen_column = "sentimen"
    like_column = "likeCount"
    time_column = "Time"

    # Pastikan semua kolom yang dibutuhkan tersedia
    required_columns = [text_column, sentimen_column, like_column, time_column, "sentimenCount"]
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        st.error(f"❌ Kolom berikut tidak ditemukan di file: {', '.join(missing_cols)}")
    else:
        # Preprocessing dasar
        df[text_column] = df[text_column].astype(str)
        df[sentimen_column] = df[sentimen_column].fillna("unknown").str.lower()
        
        # Ubah label sentimen ke Bahasa Indonesia
        df[sentimen_column] = df[sentimen_column].replace({
            "negative": "negatif",
            "positive": "positif",
            "neutral": "netral"
        })

        # Waktu
        df[time_column] = pd.to_datetime(df[time_column], errors="coerce")
        df["date"] = df[time_column].dt.date

        st.success("✅ Data berhasil dimuat!")

        # --------------------------
        # Eksplorasi Awal
        # --------------------------
        with st.expander("📊 Eksplorasi Data Awal"):
            col1, col2 = st.columns(2)

            # Histogram Like Count
            with col1:
                st.subheader("📌 Distribusi Like Komentar")
                fig_like = px.histogram(df, x=like_column, nbins=50)
                st.plotly_chart(fig_like, use_container_width=True)

            # Sebaran Sentimen
            with col2:
                st.subheader("📊 Sebaran Sentimen")
                fig_sent = px.histogram(df, x=sentimen_column)
                st.plotly_chart(fig_sent, use_container_width=True)

            # Rata-rata like per sentimen
            st.subheader("👍 Rata-rata Like per Sentimen")
            like_avg = df.groupby(sentimen_column)[like_column].mean().reset_index()
            fig_bar = px.bar(like_avg, x=sentimen_column, y=like_column, text_auto='.2s')
            st.plotly_chart(fig_bar, use_container_width=True)

            # Rata-rata sentimenCount per sentimen
            st.subheader("🔢 Rata-rata SentimenCount per Sentimen")
            count_avg = df.groupby(sentimen_column)["sentimenCount"].mean().reset_index()
            fig_count = px.bar(count_avg, x=sentimen_column, y="sentimenCount", text_auto='.2s')
            st.plotly_chart(fig_count, use_container_width=True)

            # Tren sentimen seiring waktu
            st.subheader("📅 Tren Sentimen Seiring Waktu")
            time_series = df.groupby(["date", sentimen_column]).size().reset_index(name="count")
            fig_line = px.line(time_series, x="date", y="count", color=sentimen_column, markers=True)
            st.plotly_chart(fig_line, use_container_width=True)

            # WordCloud komentar negatif dengan stopwords
            st.subheader("☁️ WordCloud Komentar Negatif (Tanpa Stopwords)")

            # Load stopwords default Bahasa Indonesia
            default_stopwords = set(stopwords.words('indonesian'))

            # Tambahkan custom stopwords
            custom_stopwords = {
                'yang', 'itu', 'dan', 'di', 'ke', 'dari', 'pada', 'untuk', 'oleh', 'dengan',
                'saat', 'kemarin', 'nanti', 'ada', 'adalah', 'baik', 'buruk', 'dll',
                'saya', 'kamu', 'dia', 'mereka', 'kita', 'kami', 'anda', 'juga',
                'ini', 'itu', 'nya', 'loh', 'sih', 'deh', 'mah', 'ga', 'gak', 'enggak',
                'tapi', 'namun', 'atau', 'ataupun', 'sebab', 'karena', 'jika', 'kalau',
                'supaya', 'biar', 'agar', 'ketika', 'setelah', 'sebelum', 'sampai', 'hingga',
                'satu', 'dua', 'tiga', 'empat', 'lima', 'enam', 'tujuh', 'delapan',
                'semua', 'beberapa', 'banyak', 'sedikit', 'orang', 'rumah', 'kota',
                'hal', 'masalah', 'sesuatu', 'ada', 'merupakan', 'menjadi', 'terjadi',
                'berada', 'sedang', 'telah', 'yg', 'ya', 'tuh', 'ngga', 'bekasi', 'walikota',
                'wali', 'pak', 'daerah', 'bikin', 'tolong', 'lg', 'udah', 'org', 'semoga', 'klo',
                'jgn', 'udh', 'dah', 'karna', 'br'
            }

            # Gabungkan stopwords
            stop_words = default_stopwords.union(custom_stopwords)

            # Filter komentar negatif
            negative_comments = df[df[sentimen_column] == "negatif"][text_column].dropna()

            # Bersihkan teks: hapus tanda baca, angka, dan konversi ke lowercase
            cleaned_texts = []
            for comment in negative_comments:
                clean_text = re.sub(r'[^\w\s]', '', str(comment).lower())
                words = [word for word in clean_text.split() if word not in stop_words]
                cleaned_texts.append(" ".join(words))

            # Gabungkan semua teks bersih menjadi satu string
            negative_text_cleaned = " ".join(cleaned_texts)

            if negative_text_cleaned:
                wc = WordCloud(
                    width=800,
                    height=400,
                    background_color='white',
                    stopwords=stop_words  # Gunakan stopwords
                ).generate(negative_text_cleaned)

                fig_wc, ax = plt.subplots(figsize=(10, 4))
                ax.imshow(wc, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig_wc)
            else:
                st.info("Tidak ada kata setelah penghapusan stopwords.")

        # --------------------------
        # Analisis BERTopic (HANYA untuk SENTIMEN NEGATIF)
        # --------------------------
        if st.button("🔍 Proses dan Analisis Topik dari Komentar Negatif"):
            df_negatif = df[df[sentimen_column] == "negatif"]

            if len(df_negatif) == 0:
                st.warning("⚠️ Tidak ada komentar negatif untuk dianalisis.")
            else:
                with st.spinner("Melatih BERTopic hanya pada komentar negatif..."):
                    docs_negatif = df_negatif[text_column].astype(str).tolist()
                    topic_model = BERTopic(language="indonesian", verbose=True)
                    topics, probs = topic_model.fit_transform(docs_negatif)

                df_topic_info = topic_model.get_topic_info()
                df_topic_info = df_topic_info[df_topic_info["Topic"] != -1]

                st.subheader("📈 Jumlah Komentar Negatif per Topik")
                fig = px.bar(df_topic_info.head(10), x="Name", y="Count", text_auto=True)
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("🧠 Rangkuman Topik Negatif & WordCloud")
                for index, row in df_topic_info.iterrows():
                    topic_id = row["Topic"]
                    label = row["Name"]
                    keywords = topic_model.get_topic(topic_id)
                    rep_docs = topic_model.get_representative_docs(topic_id)

                    with st.expander(f"Topik #{topic_id} - {label}"):
                        st.markdown("**🔑 Kata Kunci Utama:** " + ", ".join([w[0] for w in keywords[:10]]))
                        st.markdown("**💬 Contoh Komentar Negatif:**")
                        for doc in rep_docs[:3]:
                            st.markdown(f"> {doc}")

                        # WordCloud dari kata kunci topik (tanpa stopwords)
                        st.markdown("**☁️ WordCloud Kata Kunci Topik (Tanpa Stopwords):**")
                        word_freq = dict(keywords)

                        # Hapus stopwords dari word_freq
                        filtered_word_freq = {
                            word: freq for word, freq in word_freq.items() if word.lower() not in stop_words
                        }

                        if filtered_word_freq:
                            wc = WordCloud(width=800, height=400, background_color='white')
                            wc_img = wc.generate_from_frequencies(filtered_word_freq)
                            fig_wc, ax = plt.subplots(figsize=(8, 4))
                            ax.imshow(wc_img, interpolation='bilinear')
                            ax.axis("off")
                            st.pyplot(fig_wc)
                        else:
                            st.info("Semua kata dalam frekuensi termasuk stopwords.")

                # Opsi download hasil analisis topik negatif
                df_negatif["topic"] = topics
                st.download_button(
                    label="💾 Unduh Data Komentar Negatif dengan Topik",
                    data=df_negatif.to_csv(index=False),
                    file_name='analisis_topik_negatif.csv',
                    mime='text/csv'
                )
