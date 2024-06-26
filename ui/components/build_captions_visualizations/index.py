# IMPORTS
import streamlit as st
import os
from components.build_captions_visualizations.analyze_caption_lengths import analyze_caption_lengths, analyze_caption_lengths_by_class
from components.build_captions_visualizations.analyze_caption_repetition import analyze_caption_repetition
from components.build_captions_visualizations.analyze_caption_accuracy import analyze_caption_accuracy
from components.build_captions_visualizations.analyze_unique_words import analyze_dataset, plot_top_words, word_cloud
from components.build_captions_visualizations.analyze_flickr30k_class_distribution import plot_class_distribution
from components.build_captions_visualizations.analyze_flickr30k_image_distribution import plot_image_distribution

# CONFIGURE FILE PATHS
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))

def build_captions_visualizations():
  file_path_with_label = [
    { 'title': 'Blip', 'file_path': os.path.join(project_root, 'resources', 'data_visualisation', 'Blip_Label.csv') },
    { 'title': 'Kosmos', 'file_path': os.path.join(project_root, 'resources', 'data_visualisation', 'Kosmos_Label.csv') },
    { 'title': 'Flicker30k', 'file_path': os.path.join(project_root, 'resources', 'data_visualisation', 'flick30k_filtered_result.csv') }
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

        st.markdown('###')
        st.markdown("**Caption Accuracy**")
        analyze_caption_accuracy(item['file_path']) 

        st.markdown("###")
        st.markdown("**Top Words in Captions**")
        _, clean_counter = analyze_dataset(item['file_path'])
        plot_top_words(clean_counter, f"Top Word")
        word_cloud(clean_counter, f"Word Cloud")

        if item['title'] != 'Flicker30k':
          st.markdown("###")

        if item['title'] == 'Flicker30k':
          st.markdown("###")
          st.markdown("**Analyze Class Distribution**")
          plot_class_distribution(item['file_path'])

          st.markdown("###")
          st.markdown("**Analyze Image Distribution**")
          plot_image_distribution(item['file_path'])
