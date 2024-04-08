# IMPORTS
import streamlit as st
import os
import sys
from PIL import Image

from components.sidebar.index import build_sidebar
from components.build_music_generation_section.index import build_music_generation_section
from components.build_show_generation_processes_section.index import build_show_generation_processes_section
from components.build_music_player_section.index import build_music_player_section
from components.build_captions_visualizations.index import build_captions_visualizations

from src.data.dataloader import Vocabulary
from src.models.finetuned_cnn import PreTrainedCNNModels
from src.models.image_caption_attention import *
from src.utils.llama_prompt_template import PromptTemplate


# PAGE CONFIGURATIONS
st.set_page_config(
  page_title="PlayWrite",
  page_icon="images/logo.png",
  layout="wide",
  initial_sidebar_state="expanded",
  menu_items={
    "About": """
      ## PlayWrite: 

      ### An AI-powered application to generate ambient game music using images and text inputs.
      
      **GitHub**: https://github.com/Jaywhisker/PlayWrite
      
      Video game development tools are getting more and more accessible but music creation has remained largely unchanged over the last decade. 
      Buying music licenses is not an option as it is not only expensive but its not permanent. This is a large hurdle that indie developers have to overcome, thus we have created PlayWrite.
    """
  }
)

tab1, tab2 = st.tabs(["Music Generation", "Visualizations"])

# MUSIC GENERATION TAB
with tab1:
  def main():
    st.title("PlayWrite: AI-Powered Ambient Game Music Generator 🎵🎮")

    if 'is_music_generation_processing' not in st.session_state:
      st.session_state.is_music_generation_processing = False

    is_music_generated = False
    audio_file_name = None

    steps = build_sidebar(disabled=st.session_state.is_music_generation_processing)

    left_col, right_col = st.columns([2, 1], gap="large")

    with left_col:
      supporting_text, uploaded_image, start_music_generation = build_music_generation_section(disabled=st.session_state.is_music_generation_processing)

      if start_music_generation:
        is_music_generated, audio_file_name = build_show_generation_processes_section(supporting_text, uploaded_image, steps)
      
    with right_col:
      build_music_player_section(is_music_generated, audio_file_name)

  if __name__ == "__main__":
    main()

# VISUALIZATIONS TAB
with tab2:
  st.title("Data Visualizations 📊")

  build_captions_visualizations()
  