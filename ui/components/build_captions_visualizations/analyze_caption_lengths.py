# IMPORTS
import streamlit as st
import pandas as pd

def analyze_caption_lengths(df, title):
    """
    Analyze caption lengths within the provided DataFrame.

    Args:
        df (DataFrame): The DataFrame to analyze.
        title (str): The label of the dataset for display purposes.
    """
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

    print(f"Mean caption length for {title}: {mean_length}")
    print(f"Minimum caption length for {title}: {min_length}")
    print(f"Maximum caption length for {title}: {max_length}")

    unique_captions = df[caption_col].nunique()
    print(f"Number of unique captions for {title}: {unique_captions}\n")

def analyze_caption_lengths_by_class(df, title):
    """
    Analyze caption lengths by class within the provided DataFrame.

    Args:
        df (DataFrame): The DataFrame to analyze.
        title (str): The label of the dataset for display purposes.
    """
    if 'image_class' in df.columns:
        class_col = 'image_class'
    elif 'classified_label' in df.columns:
        class_col = 'classified_label'
    else:
        print(f"No known class column found in {title}.")
        return

    caption_col = 'image_caption' if 'image_caption' in df.columns else 'caption'

    for image_class, group in df.groupby(class_col):
        caption_lengths = group[caption_col].apply(lambda x: len(x.split()))
        mean_length = round(caption_lengths.mean(), 2)
        min_length = caption_lengths.min()
        max_length = caption_lengths.max()

        print(f"CLASS: {image_class} in {title}")
        print(f"Mean caption length: {mean_length}")
        print(f"Minimum caption length: {min_length}")
        print(f"Maximum caption length: {max_length}")

        unique_captions = group[caption_col].nunique()
        print(f"Number of unique captions: {unique_captions}")
        print("---")

for label, df in file_path.items():
    analyze_caption_lengths(df, label)

for label, df in file_path.items():
    analyze_caption_lengths_by_class(df, label)