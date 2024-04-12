# IMPORTS
import streamlit as st
import pandas as pd

def analyze_caption_repetition(file_path):
    """
    Finds and reports repeated captions within the provided file path.

    Args:
        file_path (string): The file path to search for repeated captions.
    """
    df = pd.read_csv(file_path)

    if 'image_caption' in df.columns:
        caption_col = 'image_caption'
    elif 'caption' in df.columns:
        caption_col = 'caption'
    else:
        return
    
    # Identifying all duplicated captions
    duplicated_df = df[df.duplicated(caption_col, keep=False)]
    # Count occurrences of each caption and sort them in descending order
    caption_counts = duplicated_df[caption_col].value_counts().sort_values(ascending=False)
    unique_captions = df[caption_col].nunique()
    total_repeated = caption_counts.size

    st.write(f"**Number of Unique Captions:** {unique_captions}")
    st.write(f"**Total Number of Unique Captions Repeated**: {total_repeated}\n")
    st.write("---")
    
    st.markdown("### Repeated Captions")
    st.markdown("###")

    # Display the repeated captions along with their counts in a table format
    for caption, count in caption_counts.items():
        st.write(f'Caption "{caption}" repeated {count} times')
    if total_repeated == 0:
        st.write("No repeated captions found.")
