import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import string
from nltk.corpus import stopwords

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

def analyze_dataset(df, dataset_label):
    """
    Analyze the dataset.

    Args:
        df (DataFrame): Input DataFrame containing the dataset.
        dataset_label (str): Label for the dataset.

    Returns:
        tuple: A tuple containing two Counters, one for all words and one for cleaned words.
    """
    caption_col = df.columns[df.columns.str.contains('caption', case=False)][0]

    all_words = df[caption_col].str.cat(sep=' ').lower().split()
    all_words_clean = clean_and_split(df[caption_col].str.cat(sep=' '))
    
    unique_words = set(all_words)
    unique_words_clean = set(all_words_clean)
    
    print(f"Unique words of {dataset_label}: {len(unique_words)}")
    print(f"Unique words w/o stopwords of {dataset_label}: {len(unique_words_clean)}")

    return Counter(all_words), Counter(all_words_clean)

def analyze_all_datasets(dataset_files, report_label):
    """
    Analyze all datasets.

    Args:
        dataset_files (dict): Dictionary containing DataFrames, where keys are dataset labels and values are DataFrames.
        report_label (str): Label for the report.

    Returns:
        tuple: A tuple containing two Counters, one for all words and one for cleaned words across all datasets.
    """
    all_counter = Counter()
    clean_counter = Counter()
    
    for label, df in dataset_files.items():
        print(f"\nAnalyzing dataset: {label}")
        counter, clean = analyze_dataset(df, label)
        all_counter += counter
        clean_counter += clean

    print(f"\n{report_label}:")
    print(f"Total unique words: {len(all_counter)}")
    print(f"Total unique words w/o stopwords: {len(clean_counter)}")

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
    
    plt.figure(figsize=(5, 3))  # Adjusted the size for better visibility
    plt.barh(words, counts, color=custom_color, height=0.7)
    plt.xlabel('Frequency')
    plt.title(title)
    plt.gca().invert_yaxis() 
    plt.show()

# Analyze BLIP2 and Kosmos2 Datasets 
landscape_path = dict(list(file_path.items())[:2])
all_counter_first_two, clean_counter_first_two = analyze_all_datasets(landscape_path, "For Landscape Dataset")
plot_top_words(clean_counter_first_two, "Word Count for Both BLIP2 and Kosmos2 Datasets")

# Analyze All Data
all_counter_all, clean_counter_all = analyze_all_datasets(file_path, "For All Datasets")
plot_top_words(clean_counter_all, "Word Count for All Datasets")