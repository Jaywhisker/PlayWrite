# IMPORTS
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import string
from nltk.corpus import stopwords
from wordcloud import WordCloud

def clean_and_split(text):
    """
    Clean and split the text into individual words, removing punctuation and stopwords.
    Args:
        text (str): Input text to be cleaned and split.
    Returns:
        list: List of cleaned and split words.
    """
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    return [word.lower() for word in text.split() if word.lower() not in stop_words]

def analyze_dataset(file_path):
    """
    Analyze the dataset.
    Args:
        file_path (string): Input file_path for csv containing data.
    Returns:
        tuple: A tuple containing two Counters, one for all words and one for cleaned words.
    """
    df = pd.read_csv(file_path)
    caption_col = df.columns[df.columns.str.contains('caption', case=False)][0]
    all_words = df[caption_col].str.cat(sep=' ').lower().split()
    all_words_clean = clean_and_split(df[caption_col].str.cat(sep=' '))

    unique_words = set(all_words)
    unique_words_clean = set(all_words_clean)

    st.write(f"Unique Words: {len(unique_words)}")
    st.write(f"Unique Words w/o Stopwords: {len(unique_words_clean)}")

    return Counter(all_words), Counter(all_words_clean)

def analyze_selected_datasets(dataset_files, report_label, num_datasets=None):
    """
    Analyze selected datasets based on the number of datasets specified.
    Args:
        dataset_files (dict): Dictionary containing DataFrames, where keys are dataset labels and values are DataFrames.
        report_label (str): Label for the report.
        num_datasets (int): Number of datasets to analyze. If None, analyze all.
    Returns:
        tuple: A tuple containing two Counters, one for all words and one for cleaned words across selected datasets.
    """
    selected_files = dict(list(dataset_files.items())[:num_datasets]) if num_datasets is not None else dataset_files

    all_counter = Counter()
    clean_counter = Counter()

    for label, df in selected_files.items():
        counter, clean = analyze_dataset(df, label)
        all_counter += counter
        clean_counter += clean

    st.write(f"Total unique words: {len(all_counter)}")
    st.write(f"Total unique words w/o stopwords: {len(clean_counter)}")

    return all_counter, clean_counter

def plot_top_words(counter, title):
    """
    Plot the top words from the given counter.
    Args:
        counter (Counter): Counter object containing word frequencies.
        title (str): Title for the plot.
    """
    top_words = counter.most_common(10)
    words, counts = zip(*top_words)
    custom_color = (0.1, 0.4, 0.8)
    plt.figure(figsize=(5, 3))
    plt.barh(words, counts, color=custom_color, height=0.7)
    plt.xlabel('Frequency')
    plt.title(title)
    plt.gca().invert_yaxis()
    st.pyplot(plt)

def word_cloud(counter, title):
    """
    Generate and display a word cloud from the given counter.

    Args:
        counter (Counter): Counter object containing word frequencies.
        title (str): Title for the word cloud.
    """
    wordcloud = WordCloud(width=800, height=400, background_color ='white').generate_from_frequencies(counter)
    plt.figure(figsize=(6, 3), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.title(title, fontsize=24)
    st.pyplot(plt)
