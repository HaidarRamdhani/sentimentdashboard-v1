import streamlit as st
import pandas as pd
import plotly.express as px
from bertopic import BERTopic
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from datetime import datetime

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
# -------------------------------
# Deteksi otomatis kolom
# -------------------------------
def auto_detect_column(possible_names, df_columns):
    for name in possible_names:
        for col in df_columns:
            if name.lower() in col.lower():
                return col
    return "(tidak ada)"

# Kolom otomatis
auto_sentimen = auto_detect_column(["sentimen", "sentiment"], df.columns)
auto_like = auto_detect_column(["likeCount", "like", "jumlah_like"], df.columns)
auto_time = auto_detect_column(["Time", "timestamp", "waktu", "tanggal"], df.columns)

# Dropdown dengan preselect
sentimen_column = st.selectbox("Pilih kolom sentimen (opsional):", ["(tidak ada)"] + list(df.columns),
                               index=(["(tidak ada)"] + list(df.columns)).index(auto_sentimen))
like_column = st.selectbox("Pilih kolom like (opsional):", ["(tidak ada)"] + list(df.columns),
                           index=(["(tidak ada)"] + list(df.columns)).index(auto_like))
time_column = st.selectbox("Pilih kolom waktu (opsional):", ["(tidak ada)"] + list(df.columns),
                           index=(["(tidak ada)"] + list(df.columns)).index(auto_time))

# Alihkan sentimen ke bahasa Indonesia jika dipilih
if sentimen_column != "(tidak ada)":
    df[sentimen_column] = df[sentimen_column].str.lower().map({
        "positive": "positif",
        "negative": "negatif",
        "neutral": "netral"
    }).fillna(df[sentimen_column])

    # Deteksi kolom teks & sentimen
    text_column = st.selectbox("Pilih kolom komentar:", df.columns)

    # Preprocessing dasar
    df[text_column] = df[text_column].astype(str)

    if sentimen_column != "(tidak ada)":
        df[sentimen_column] = df[sentimen_column].fillna("unknown")

    if like_column != "(tidak ada)":
        df[like_column] = pd.to_numeric(df[like_column], errors="coerce").fillna(0)

    if time_column != "(tidak ada)":
        df[time_column] = pd.to_datetime(df[time_column], errors="coerce")
        df["date"] = df[time_column].dt.date

    st.success("✅ Data berhasil dimuat!")

    # --------------------------
    # Eksplorasi Awal
    # --------------------------
    with st.expander("📊 Eksplorasi Data Awal"):
        col1, col2 = st.columns(2)
        if like_column != "(tidak ada)":
            with col1:
                st.subheader("📌 Distribusi Like Komentar")
                fig_like = px.histogram(df, x=like_column, nbins=50)
                st.plotly_chart(fig_like, use_container_width=True)

        if sentimen_column != "(tidak ada)":
            with col2:
                st.subheader("📊 Sebaran Sentimen")
                fig_sent = px.histogram(df, x=sentimen_column)
                st.plotly_chart(fig_sent, use_container_width=True)

        if sentimen_column != "(tidak ada)" and like_column != "(tidak ada)":
            st.subheader("👍 Rata-rata Like per Sentimen")
            like_avg = df.groupby(sentimen_column)[like_column].mean().reset_index()
            fig_bar = px.bar(like_avg, x=sentimen_column, y=like_column, text_auto='.2s')
            st.plotly_chart(fig_bar, use_container_width=True)

        if sentimen_column != "(tidak ada)" and time_column != "(tidak ada)":
            st.subheader("📅 Tren Sentimen Seiring Waktu")
            time_series = df.groupby(["date", sentimen_column]).size().reset_index(name="count")
            fig_line = px.line(time_series, x="date", y="count", color=sentimen_column, markers=True)
            st.plotly_chart(fig_line, use_container_width=True)

        # WordCloud komentar negatif
        if sentimen_column != "(tidak ada)":
            st.subheader("☁️ WordCloud Komentar Negatif")
            negative_text = " ".join(df[df[sentimen_column] == "negative"][text_column].dropna())
            if negative_text:
                wc = WordCloud(width=800, height=400, background_color='white').generate(negative_text)
                fig_wc, ax = plt.subplots(figsize=(10, 4))
                ax.imshow(wc, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig_wc)
            else:
                st.info("Tidak ada komentar negatif.")

    # --------------------------
    # Analisis BERTopic
    # --------------------------
    if st.button("🔍 Proses dan Analisis Topik"):
        with st.spinner("Melatih BERTopic..."):
            docs = df[text_column].astype(str).tolist()
            topic_model = BERTopic(language="indonesian", verbose=True)
            topics, probs = topic_model.fit_transform(docs)

        df_topic_info = topic_model.get_topic_info()
        df_topic_info = df_topic_info[df_topic_info["Topic"] != -1]

        st.subheader("📈 Jumlah Komentar per Topik")
        fig = px.bar(df_topic_info.head(10), x="Name", y="Count", text_auto=True)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🧠 Rangkuman Topik & WordCloud")
        for index, row in df_topic_info.iterrows():
            topic_id = row["Topic"]
            label = row["Name"]
            keywords = topic_model.get_topic(topic_id)
            rep_docs = topic_model.get_representative_docs(topic_id)

            with st.expander(f"Topik #{topic_id} - {label}"):
                st.markdown("**🔑 Kata Kunci Utama:** " + ", ".join([w[0] for w in keywords[:10]]))
                st.markdown("**💬 Contoh Komentar:**")
                for doc in rep_docs[:3]:
                    st.markdown(f"> {doc}")

                # WordCloud
                st.markdown("**☁️ WordCloud Kata Kunci:**")
                word_freq = dict(keywords)
                wc = WordCloud(width=800, height=400, background_color='white')
                wc_img = wc.generate_from_frequencies(word_freq)
                fig_wc, ax = plt.subplots(figsize=(8, 4))
                ax.imshow(wc_img, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig_wc)
