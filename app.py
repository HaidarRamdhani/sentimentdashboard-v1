import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from io import BytesIO

st.set_page_config(page_title="Dashboard Sentimen", layout="wide")
st.title("📊 Dashboard Analisis Sentimen Komentar YouTube")

# Upload file
uploaded_file = st.file_uploader("Unggah file CSV atau Excel", type=["csv", "xlsx"])

if uploaded_file:
    # Load data
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Preprocessing
    df['publishedAt'] = pd.to_datetime(df.get('Time', pd.NaT), errors='coerce')
    df['likeCount'] = pd.to_numeric(df.get('likeCount', 0), errors='coerce').fillna(0)
    df['sentimen'] = df.get('sentimen', 'unknown').fillna('unknown')

    st.success("✅ Data berhasil dimuat!")

    # Fungsi untuk membuat WordCloud
    def generate_wordcloud(sentiment):
        text = " ".join(df[df["sentimen"] == sentiment]["cleanedText"].dropna())
        wc = WordCloud(width=800, height=400, background_color='white').generate(text)

        buffer = BytesIO()
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(buffer, format="png")
        plt.close()
        buffer.seek(0)
        return buffer

    # Layout dashboard
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📌 Distribusi Like Komentar")
        fig_like = px.histogram(df, x='likeCount', nbins=50, title="Distribusi Like Komentar")
        st.plotly_chart(fig_like, use_container_width=True)

    with col2:
        st.subheader("📊 Sebaran Sentimen")
        fig_sent = px.histogram(df, x='sentimen', title='Jumlah Sentimen')
        st.plotly_chart(fig_sent, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("👍 Rata-rata Like per Sentimen")
        like_avg = df.groupby('sentimen')['likeCount'].mean().reset_index()
        fig_bar = px.bar(like_avg, x='sentimen', y='likeCount',
                         title="Rata-rata Jumlah Like per Sentimen",
                         color='sentimen', text_auto='.2s')
        st.plotly_chart(fig_bar, use_container_width=True)

    with col4:
        st.subheader("📅 Tren Sentimen dari Waktu ke Waktu")
        df['date'] = df['publishedAt'].dt.date
        time_series = df.groupby(['date', 'sentimen']).size().reset_index(name='count')
        fig_line = px.line(
            time_series, x='date', y='count', color='sentimen', markers=True,
            title='Sentimen Seiring Waktu'
        )
        fig_line.update_layout(xaxis_title='Tanggal', yaxis_title='Jumlah Komentar')
        st.plotly_chart(fig_line, use_container_width=True)

    st.subheader("☁️ Wordcloud Komentar Negatif")
    if not df[df["sentimen"] == "negative"].empty:
        img_data = generate_wordcloud("negative")
        st.image(img_data, use_column_width=True)
    else:
        st.info("Tidak ada komentar negatif untuk ditampilkan.")
else:
    st.info("Silakan unggah file CSV atau Excel terlebih dahulu.")
