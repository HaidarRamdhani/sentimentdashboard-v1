import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from io import BytesIO

# BERTopic dan dependencies
from bertopic import BERTopic

st.set_page_config(page_title="Dashboard Sentimen + Topic Clustering", layout="wide")
st.title("📊 Dashboard Analisis Sentimen Komentar YouTube")

uploaded_file = st.file_uploader("Unggah file CSV atau Excel", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Preprocessing
    df['publishedAt'] = pd.to_datetime(df.get('Time', pd.NaT), errors='coerce')
    df['likeCount'] = pd.to_numeric(df.get('likeCount', 0), errors='coerce').fillna(0)
    df['sentimen'] = df.get('sentimen', 'unknown').fillna('unknown')

    st.success("✅ Data berhasil dimuat!")

    # Fungsi WordCloud
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

    # Visualisasi dasar
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

    # ====== BERTopic untuk clustering komentar negatif ======
    st.subheader("🗂️ Topic Clustering Komentar Negatif")

    negatif_df = df[df['sentimen'] == 'negative'].copy()
    negatif_texts = negatif_df['cleanedText'].dropna().tolist()

    if len(negatif_texts) < 10:
        st.warning("Data komentar negatif kurang dari 10, clustering tidak dilakukan.")
    else:
        with st.spinner("Sedang melakukan topic clustering..."):
            topic_model = BERTopic(language="indonesian", verbose=False)
            topics, probs = topic_model.fit_transform(negatif_texts)

        # Tampilkan ringkasan topik
        topic_info = topic_model.get_topic_info()
        st.write(topic_info.head(10))

        # Visualisasi bar chart topik
        fig_topic = topic_model.visualize_barchart(top_n_topics=10)
        st.plotly_chart(fig_topic, use_container_width=True)

        # Contoh komentar dari topik teratas
        top_topic = topic_info.iloc[1]['Topic']  # biasanya topik 0 itu outlier
        st.markdown(f"**Contoh komentar untuk Topik #{top_topic}:**")
        example_texts = [text for i, text in enumerate(negatif_texts) if topics[i] == top_topic][:5]
        for idx, txt in enumerate(example_texts, 1):
            st.write(f"{idx}. {txt}")

else:
    st.info("Silakan unggah file CSV atau Excel terlebih dahulu.")
