# IMPORTS
import streamlit as st
import pandas as pd

def find_repeated_captions(df, title):
    """
    Finds and reports repeated captions within the provided DataFrame.

    Args:
        df (DataFrame): The DataFrame to search for repeated captions.
        title (str): The label of the dataset for display purposes.
    """
    if 'image_caption' in df.columns:
        caption_col = 'image_caption'
    elif 'caption' in df.columns:
        caption_col = 'caption'
    else:
        print(f"No known caption column found in {title}.")
        return
    
    # Identifying all duplicated captions
    duplicated_df = df[df.duplicated(caption_col, keep=False)]
    # Count occurrences of each caption and sort them in descending order
    caption_counts = duplicated_df[caption_col].value_counts().sort_values(ascending=False)

    print(f"Dataset: {title}")
    total_repeated = caption_counts.size
    print(f"Total number of unique captions repeated: {total_repeated}\n")

    for caption, count in caption_counts.items():
        print(f'Caption "{caption}" repeated {count} times')
    print("")

for label, df in file_path.items():
    find_repeated_captions(df, label)