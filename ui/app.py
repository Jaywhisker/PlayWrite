# IMPORT LIBRARIES
import streamlit as st
from PIL import Image

# IMPORT SECTIONS
from components.sidebar.index import build_sidebar
from components.build_music_generation_section.index import build_music_generation_section
from components.build_show_processes_section.index import build_show_processes_section
from components.build_music_player_section.index import build_music_player_section

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
      
      **GitHub**: 
      
      Video game development tools are getting more and more accessible but music creation has remained largely unchanged over the last decade. 
      Buying music licenses is not an option as it is not only expensive but its not permanent. This is a large hurdle that indie developers have to overcome, thus we have created PlayWrite.
    """
  }
)

tab1, tab2 = st.tabs(["Home", "Visualizations"])

# HOME TAB
with tab1:
  st.title("PlayWrite: AI-Powered Ambient Game Music Generator 🎵🎮")

  def main():
    epochs, steps = build_sidebar()

    left_col, right_col = st.columns([2, 1], gap="large")
    is_music_generated = False

    with left_col:
      uploaded_image, supporting_text, generate_music = build_music_generation_section()

      if generate_music:
        is_music_generated = build_show_processes_section()
      
    with right_col:
      build_music_player_section(is_music_generated)

  if __name__ == "__main__":
    main()

# VISUALIZATIONS TAB
with tab2:
  st.title("Data Visualization 📊")
  
  