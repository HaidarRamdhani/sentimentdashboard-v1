import streamlit as st
import pandas as pd
import numpy as np
import re
import emoji
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
from umap import UMAP
import hdbscan
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# --- AMBIL API KEY ---
try:
    YOUTUBE_API_KEY = st.secrets.youtube.api_key
except Exception as e:
    st.error("API Key tidak ditemukan di secrets.toml")
    st.stop()

# --- YOUTUBE SCRAPING ---
def scrape_youtube_comments(video_url):
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    video_id = video_url.split("v=")[-1].split("&")[0]
    
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100
    )
    
    comments = []
    while request:
        response = request.execute()
        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
        request = youtube.commentThreads().list_next(request, response)
    
    return pd.DataFrame({"comments": comments})

# --- PREPROCESSING ---
def preprocess_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = emoji.demojize(text, delimiters=(" :", ": "))
    text = re.sub(r"[^\w\s:]", "", text)
    factory = StopWordRemoverFactory()
    stopword = factory.create_stop_word_remover()
    return stopword.remove(text)

# --- SENTIMENT ANALYSIS DENGAN MODEL BAHASA INGGRIS (fallback) ---
try:
    from transformers import pipeline
    sentiment_analyzer = pipeline("text-classification", model="cahya/bert-base-indonesian-1.5G-sentiment-analysis", device=-1)
except Exception as e:
    st.warning(f"Model sentimen Bahasa Indonesia gagal dimuat: {e}. Menggunakan model fallback Bahasa Inggris.")
    sentiment_analyzer = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english", device=-1)

def analyze_sentiment(texts):
    return [result['label'] for result in sentiment_analyzer(texts)]

# --- EMBEDDING DENGAN TF-IDF + UMAP ---
def get_tfidf_embeddings(texts):
    st.info("Membuat embedding dengan TF-IDF + UMAP...")
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(texts)
    umap_model = UMAP(n_components=768, n_neighbors=15, random_state=42)
    embeddings = umap_model.fit_transform(tfidf.toarray())
    st.info("Embedding selesai.")
    return embeddings

# --- CLUSTERING ---
def cluster_comments(embeddings):
    st.info("Memulai klasterisasi dengan HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
    clusters = clusterer.fit_predict(embeddings)
    st.info("Klasterisasi selesai.")
    return clusters

# --- VISUALIZATION ---
def plot_sentiment_distribution(df):
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x="sentiment")
    plt.title("Distribusi Sentimen")
    st.pyplot(plt)

def plot_clusters(embeddings, clusters):
    st.info("Visualisasi klaster...")
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], hue=clusters, palette="viridis", legend="full")
    plt.title("Cluster Komentar")
    st.pyplot(plt)

def generate_wordcloud(df, clusters):
    for cluster in np.unique(clusters):
        if cluster == -1:
            continue
        cluster_comments = df[df["cluster"] == cluster]["cleaned"].str.cat(sep=" ")
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(cluster_comments)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)

# --- STREAMLIT APP ---
st.title("📊 Dashboard Analisis Komentar YouTube")
st.markdown("Masukkan URL video YouTube untuk menganalisis komentar.")

video_url = st.text_input("URL Video YouTube")

if st.button("Analisis"):
    if not video_url:
        st.error("Harap masukkan URL video!")
    else:
        with st.spinner("1/5 Mengambil komentar dari YouTube..."):
            comments_df = scrape_youtube_comments(video_url)
            comments_df["cleaned"] = comments_df["comments"].apply(preprocess_text)

        st.write("🔍 Komentar Bersih:")
        st.write(comments_df[["comments", "cleaned"]].head(10))

        with st.spinner("2/5 Membuat embedding dengan TF-IDF..."):
            embeddings = get_tfidf_embeddings(comments_df["cleaned"].tolist())

        with st.spinner("3/5 Analisis sentimen..."):
            comments_df["sentiment"] = analyze_sentiment(comments_df["cleaned"].tolist())

        with st.spinner("4/5 Klasterisasi dengan HDBSCAN..."):
            comments_df["cluster"] = cluster_comments(embeddings)

        st.write("📈 Hasil Klasterisasi:")
        st.write(comments_df[["cleaned", "sentiment", "cluster"]])

        with st.spinner("5/5 Visualisasi hasil..."):
            plot_sentiment_distribution(comments_df)
            plot_clusters(embeddings, comments_df["cluster"])
            generate_wordcloud(comments_df, comments_df["cluster"])
