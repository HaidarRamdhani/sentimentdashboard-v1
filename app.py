import streamlit as st
import pandas as pd
import plotly.express as px
from bertopic import BERTopic
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# -------------------------------
# 1. Upload File dan Pilih Sheet
# -------------------------------
st.title("📊 Dashboard Analisis Topik Komentar dengan BERTopic")
uploaded_file = st.file_uploader("Unggah file Excel atau CSV", type=["xlsx", "csv"])

if uploaded_file:
    # Baca file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        xls = pd.ExcelFile(uploaded_file)
        sheet_names = xls.sheet_names
        selected_sheet = st.selectbox("Pilih sheet:", sheet_names)
        df = pd.read_excel(xls, sheet_name=selected_sheet)

    # Deteksi kolom teks
    text_column = st.selectbox("Pilih kolom komentar:", df.columns)
    if st.button("🔍 Proses dan Analisis Topik"):

        # -------------------------------
        # 2. Training BERTopic
        # -------------------------------
        st.info("Melatih model BERTopic... tunggu sebentar.")
        docs = df[text_column].astype(str).tolist()
        topic_model = BERTopic(language="indonesian", verbose=True)
        topics, probs = topic_model.fit_transform(docs)

        # Ambil info topik
        df_topic_info = topic_model.get_topic_info()
        df_topic_info = df_topic_info[df_topic_info['Topic'] != -1]  # Exclude outliers

        # -------------------------------
        # 3. Bar Chart Topik
        # -------------------------------
        st.subheader("📈 Jumlah Komentar per Topik")
        fig = px.bar(df_topic_info.head(10),
                     x="Name", y="Count",
                     title="Top 10 Topik Terbanyak",
                     text_auto=True)
        st.plotly_chart(fig, use_container_width=True)

        # -------------------------------
        # 4. Ringkasan Topik + Wordcloud
        # -------------------------------
        st.subheader("🧠 Detail Topik")
        for index, row in df_topic_info.iterrows():
            topic_id = row["Topic"]
            label = row["Name"]
            keywords = topic_model.get_topic(topic_id)
            representative_doc = topic_model.get_representative_docs(topic_id)[0]

            with st.expander(f"Topik #{topic_id} - {label}"):
                st.markdown("**🔑 Kata Kunci Utama:** " +
                            ", ".join([w[0] for w in keywords[:10]]))
                st.markdown("**💬 Contoh Komentar:**")
                st.markdown(f"> {representative_doc}")

                # WordCloud
                st.markdown("**☁️ WordCloud Kata Kunci:**")
                word_freq = dict(keywords)
                wc = WordCloud(width=800, height=400, background_color='white')
                wc_img = wc.generate_from_frequencies(word_freq)

                fig_wc, ax = plt.subplots(figsize=(8, 4))
                ax.imshow(wc_img, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig_wc)
