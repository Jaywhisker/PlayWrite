# IMPORTS
import streamlit as st
import os
from components.build_captions_visualizations.analyze_caption_lengths import analyze_caption_lengths, analyze_caption_lengths_by_class
from components.build_captions_visualizations.analyze_caption_repetition import analyze_caption_repetition
from components.build_captions_visualizations.analyze_flickr30k_classification import plot_class_distribution
# from components.build_captions_visualizations.analyze_unique_word import plot_top_words

# CONFIGURE FILE PATHS
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))

def build_captions_visualizations():
  file_path_with_label = [
    { 'title': 'Blip', 'file_path': os.path.join(project_root, 'src', 'data_visualisation', 'output', 'Blip_Label.csv') },
    { 'title': 'Kosmos', 'file_path': os.path.join(project_root, 'src', 'data_visualisation', 'output', 'Kosmos_Label.csv') },
    { 'title': 'Flicker30k', 'file_path': os.path.join(project_root, 'src', 'data_visualisation', 'output', 'flick30k_filtered_result.csv') }
  ]

  st.markdown("### Captions Analysis")

  col_container = st.container()

  with col_container:
    cols = st.columns(2)
    
    for i, item in enumerate(file_path_with_label):
      column = cols[i % 2]
      
      with column:
        st.subheader(item['title'])

        with st.expander("Analyze Caption Lengths"):
          st.markdown("###")
          analyze_caption_lengths(item['file_path'])

        with st.expander("Analyze Caption Lengths by Class"):
          st.markdown("###")
          analyze_caption_lengths_by_class(item['file_path'])
            
        with st.expander("Analyze Caption Repetition"):
          st.markdown("###")
          analyze_caption_repetition(item['file_path'])       

        if item['title'] != 'Flicker30k':
          st.markdown("###")
          st.markdown("Top Words in Captions")

        if item['title'] == 'Flicker30k':
          st.markdown("###")
          st.markdown("Analyze Class Distribution")
          plot_class_distribution(item['file_path'])
