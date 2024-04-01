import pandas as pd
import os

from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
import string
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
import streamlit as st


def sample_function(file_path):
  st.markdown(f'This is a sample {file_path}')

# Overall Length Statistics
def analyze_caption_lengths(file_path, title):
    df = pd.read_csv(file_path)

    if 'image_caption' in df.columns:
        caption_col = 'image_caption'
    elif 'caption' in df.columns:
        caption_col = 'caption'
    else:
        print(f"No known caption column found in {title}.")
        return

    caption_lengths = df[caption_col].apply(lambda x: len(x.split()))
    mean_length = round(caption_lengths.mean(), 2)
    min_length = caption_lengths.min()
    max_length = caption_lengths.max()

    st.markdown(f"Mean caption length: {mean_length}")
    st.markdown(f"Minimum caption length: {min_length}")
    st.markdown(f"Maximum caption length: {max_length}")

    unique_captions = df[caption_col].nunique()
    st.markdown(f"Number of unique captions: {unique_captions}")

def analyze_caption_lengths_by_class(file_path):
    df = pd.read_csv(file_path)

    for image_class, group in df.groupby('image_class'):
      caption_lengths = group['image_caption'].apply(lambda x: len(x.split()))
      mean_length = round(caption_lengths.mean(), 2)
      min_length = caption_lengths.min()
      max_length = caption_lengths.max()

      st.markdown(f"CLASS: {image_class}")
      st.markdown(f"Mean caption length: {mean_length}")
      st.markdown(f"Minimum caption length: {min_length}")
      st.markdown(f"Maximum caption length: {max_length}")

      # Count the number of unique captions within the class
      unique_captions = group['image_caption'].nunique()
      st.markdown(f"Number of unique captions: {unique_captions}")
      
def find_repeated_captions(file_path):
    df = pd.read_csv(file_path)
    
    # Identifying all duplicated captions
    duplicated_df = df[df.duplicated('image_caption', keep=False)]
    # Count occurrences of each caption and sort them in descending order
    caption_counts = duplicated_df['image_caption'].value_counts().sort_values(ascending=False)

    total_repeated = caption_counts.size
    st.markdown(f"Total number of unique captions repeated: {total_repeated}\n")

# for file, title in output_files.items():
#     find_repeated_captions(file)

def visualize_top_words(file_path, title):
    nltk.download('punkt')

    # Read the CSV file
    df = pd.read_csv(file_path)
  

    # Tokenize captions
    stop_words = set(stopwords.words('english'))
    tokenized_captions = word_tokenize(' '.join(df['image_caption']).lower())  # Assuming captions are space-separated sentences

    # Remove stopwords and punctuation
    filtered_captions = [word for word in tokenized_captions if word not in stop_words and word not in string.punctuation]

    # Calculate word frequencies
    word_freq = Counter(filtered_captions)

    # Visualize most frequent words using word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

    # Displaying the word cloud
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Word Cloud of Most Frequent Words in {title} Captions (excluding stopwords and punctuation)')
    plt.axis('off')
    st.pyplot(plt)  # Use st.pyplot() to display matplotlib plots

    # Visualize most frequent words using bar chart
    most_common_words = word_freq.most_common(20)  # Change 20 to adjust the number of words to display

    # Preparing the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(most_common_words)), [val[1] for val in most_common_words], tick_label=[val[0] for val in most_common_words])
    plt.title(f'Bar Chart of Most Frequent Words in {title} Captions (excluding stopwords)')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(plt)

def visualize_top_words_by_class(file_path, title):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Group by 'image_class' and then analyze captions for each class
    for image_class, group in df.groupby('image_class'):
        # Tokenize captions
        stop_words = set(stopwords.words('english'))
        tokenized_captions = word_tokenize(' '.join(group['image_caption']).lower())  # Assuming captions are space-separated sentences

        # Remove stopwords and punctuation
        filtered_captions = [word for word in tokenized_captions if word not in stop_words and word not in string.punctuation]

        # Calculate word frequencies
        word_freq = Counter(filtered_captions)

        # Visualize most frequent words using word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'Word Cloud of Most Frequent Words in {title} - Class: {image_class} Captions (excluding stopwords and punctuation)')
        plt.axis('off')
        plt.show()

        # Visualize most frequent words using bar chart
        most_common_words = word_freq.most_common(20)  # Change 20 to adjust the number of words to display

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(most_common_words)), [val[1] for val in most_common_words], tick_label=[val[0] for val in most_common_words])
        plt.title(f'Bar Chart of Most Frequent Words in {title} - Class: {image_class} Captions (excluding stopwords and punctuation)')
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45, ha='right')
        plt.show()

def extract_unique_words_from_files(output_files):
    stop_words = set(stopwords.words('english'))

    unique_landscape_words = set()

    def filter_landscape_words(words):
        landscape_words = set()
        for word in words:
            synsets = wn.synsets(word)
            for synset in synsets:
                hypernyms = synset.hypernyms()
                for hypernym in hypernyms:
                    if 'landscape' in hypernym.definition():
                        landscape_words.add(word)
                        break  # Stop searching hypernyms if found
                if word in landscape_words:
                    break  # Stop searching synsets if found
        return landscape_words

    for file in output_files:
        with open(file, 'r', encoding='utf-8') as f:
            text = f.read()

        words = word_tokenize(text)

        words = [word.lower() for word in words if word.isalpha()]

        words = [word for word in words if word not in stop_words]

        landscape_words = filter_landscape_words(words)

        unique_landscape_words.update(landscape_words)

    return list(unique_landscape_words)
