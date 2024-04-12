# IMPORTS
import streamlit as st
import pandas as pd

def analyze_caption_lengths(file_path):
    """
    Analyze caption lengths within the provided CSV file.

    Args:
        file_path (str): The path to the CSV file to analyze.
    """
    df = pd.read_csv(file_path)
    title = file_path.split('/')[-1]

    if 'image_caption' in df.columns:
        caption_col = 'image_caption'
    elif 'caption' in df.columns:
        caption_col = 'caption'
    else:
        st.write(f"No known caption column found in {title}.")
        return

    caption_lengths = df[caption_col].apply(lambda x: len(x.split()))
    mean_length = round(caption_lengths.mean(), 2)
    min_length = caption_lengths.min()
    max_length = caption_lengths.max()

    st.write(f"**Mean Caption Length:** {mean_length}")
    st.write(f"**Minimum Caption Length:** {min_length}")
    st.write(f"**Maximum Caption Length:** {max_length}")

def analyze_caption_lengths_by_class(file_path):
    """
    Analyze caption lengths by class within the provided CSV file.

    Args:
        file_path (str): The path to the CSV file to analyze.
    """
    df = pd.read_csv(file_path)
    title = file_path.split('/')[-1]

    if 'image_class' in df.columns:
        class_col = 'image_class'
    elif 'classified_label' in df.columns:
        class_col = 'classified_label'
    else:
        st.write(f"No known class column found in {title}.")
        return

    caption_col = 'image_caption' if 'image_caption' in df.columns else 'caption'

    for image_class, group in df.groupby(class_col):
        caption_lengths = group[caption_col].apply(lambda x: len(x.split()))
        mean_length = round(caption_lengths.mean(), 2)
        min_length = caption_lengths.min()
        max_length = caption_lengths.max()

        st.write(f"**CLASS: {image_class}**")
        st.write(f"Mean Caption Length: {mean_length}")
        st.write(f"Minimum Caption Length: {min_length}")
        st.write(f"Maximum Caption Length: {max_length}")

        unique_captions = group[caption_col].nunique()
        st.write(f"Number of Unique Captions: {unique_captions}")
        st.write("---")
