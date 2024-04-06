# IMPORTS
import streamlit as st
import pandas as pd

def analyze_caption_repetition(file_path):
  df = pd.read_csv(file_path)
  
  duplicated_df = df[df.duplicated('image_caption', keep=False)]
  caption_counts = duplicated_df['image_caption'].value_counts().sort_values(ascending=False)

  total_repeated = caption_counts.size
  st.markdown(f"Total number of unique captions repeated: {total_repeated}\n")
