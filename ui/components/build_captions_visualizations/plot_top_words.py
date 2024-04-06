# IMPORTS
import streamlit as st
import pandas as pd
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
import string
import matplotlib.pyplot as plt
import nltk

nltk.download('stopwords')
nltk.download('punkt')

def plot_top_words(file_path, title):
    df = pd.read_csv(file_path)

    stop_words = set(stopwords.words('english'))
    tokenized_captions = word_tokenize(' '.join(df['image_caption']).lower())

    filtered_captions = [word for word in tokenized_captions if word not in stop_words and word not in string.punctuation]

    word_freq = Counter(filtered_captions)

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Word Cloud of Most Frequent Words in {title} Captions (excluding stopwords and punctuation)')
    plt.axis('off')
    st.pyplot(plt)

    most_common_words = word_freq.most_common(20)

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(most_common_words)), [val[1] for val in most_common_words], tick_label=[val[0] for val in most_common_words])
    plt.title(f'Bar Chart of Most Frequent Words in {title} Captions (excluding stopwords)')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(plt)
