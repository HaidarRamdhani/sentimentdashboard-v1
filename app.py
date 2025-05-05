import streamlit as st
import pandas as pd
import numpy as np
import torch
import hdbscan
import umap
import re
import emoji
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from youtube_comment_downloader import YoutubeCommentDownloader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

# Load tokenizer and models
@st.cache_resource
def load_models():
    tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobertweet-base")
    embed_model = AutoModel.from_pretrained("indobenchmark/indobertweet-base")
    sent_model = AutoModelForSequenceClassification.from_pretrained("indobenchmark/indobertweet-base")
    return tokenizer, embed_model, sent_model

tokenizer, embed_model, sent_model = load_models()

# Step 1: Scrape YouTube comments
def scrape_comments(video_url, max_comments=300):
    downloader = YoutubeCommentDownloader()
    comments = downloader.get_comments_from_url(video_url, sort_by="top")
    data = [c['text'] for i, c in zip(range(max_comments), comments)]
    return pd.DataFrame({"comment": data})

# Step 2: Preprocessing

def clean_text(text):
    text = emoji.demojize(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = text.lower()
    return text.strip()

def preprocess(df):
    df['clean'] = df['comment'].apply(clean_text)
    return df

# Step 3: Embedding
@torch.no_grad()
def get_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    outputs = embed_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

# Step 4: Sentiment Analysis
@torch.no_grad()
def get_sentiments(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    outputs = sent_model(**inputs)
    preds = torch.argmax(outputs.logits, dim=1).numpy()
    return preds

# Step 5: Clustering

def cluster_texts(embeddings):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.0, metric='cosine')
    embedding_2d = reducer.fit_transform(embeddings)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10, metric='euclidean')
    labels = clusterer.fit_predict(embedding_2d)
    return embedding_2d, labels

# Step 6: WordCloud

def plot_wordcloud(texts, title):
    vec = CountVectorizer(stop_words='english').fit(texts)
    bag_of_words = vec.transform(texts)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    wc = WordCloud(width=800, height=400).generate_from_frequencies(dict(words_freq))
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    st.pyplot(plt)

# Streamlit App
st.title("📺 YouTube Komentar Analyzer - Sentimen & Topik")

video_url = st.text_input("Masukkan URL video YouTube")

if st.button("Scrape & Analyze") and video_url:
    with st.spinner("Mengambil komentar..."):
        df = scrape_comments(video_url)
        df = preprocess(df)

    with st.spinner("Embedding dan Analisis Sentimen..."):
        embeddings = get_embeddings(df['clean'].tolist())
        sentiments = get_sentiments(df['clean'].tolist())
        df['sentiment'] = sentiments

    with st.spinner("Clustering dengan HDBSCAN..."):
        emb_2d, clusters = cluster_texts(embeddings)
        df['cluster'] = clusters

    st.success("Analisis selesai!")

    st.subheader("📊 Distribusi Sentimen")
    sent_map = {0: "Negatif", 1: "Netral", 2: "Positif"}
    df['sent_label'] = df['sentiment'].map(sent_map)
    st.bar_chart(df['sent_label'].value_counts())

    st.subheader("🧠 Clustering Komentar")
    st.write("UMAP Scatter plot dan Wordcloud per cluster")
    cluster_select = st.selectbox("Pilih cluster:", sorted(df['cluster'].unique()))
    cluster_df = df[df['cluster'] == cluster_select]
    plot_wordcloud(cluster_df['clean'], title=f"Wordcloud Cluster {cluster_select}")

    st.subheader("📃 Komentar pada Cluster Ini")
    st.dataframe(cluster_df[['comment', 'sent_label']])
