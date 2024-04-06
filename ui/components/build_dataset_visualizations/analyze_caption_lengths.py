# IMPORTS
import streamlit as st
import pandas as pd

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

    unique_captions = group['image_caption'].nunique()
    st.markdown(f"Number of unique captions: {unique_captions}")
    st.markdown("---")
