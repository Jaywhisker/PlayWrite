# IMPORTS
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def analyze_class_distribution(file_path):
    """
    Analyze the class distribution of the DataFrame.

    Args:
        file_path (string): Input a file_path to the csv containing the data.

    Returns:
        Series: Class distribution.
        int: Total number of images.
    """
    df = pd.read_csv(file_path)

    class_distribution = df['classified_label'].value_counts()
    total_images = class_distribution.sum()
    return class_distribution, total_images

def plot_class_distribution(file_path):
    """
    Plot a pie chart showing the distribution of image classes.

    Args:
        class_distribution (Series): Class distribution.
        total_images (int): Total number of images.
    """
    class_distribution, total_images = analyze_class_distribution(file_path)

    labels = class_distribution.index
    colors = ['#8EA4D2', '#F7B2AD', '#FFD6A5', '#BEE1E6', '#C3ACEE']

    def func(pct, allvals):
        absolute = int(pct/100.*total_images)
        return "{:d}\n({:.1f}%)".format(absolute, pct)

    plt.figure(figsize=(6, 6))
    plt.pie(class_distribution, labels=labels, colors=colors, startangle=140,
            autopct=lambda pct: func(pct, class_distribution))
    plt.title('Distribution of Image Classes', fontsize=16)
    plt.tight_layout() 
    st.pyplot(plt)
