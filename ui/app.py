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

st.title("PlayWrite: AI-Powered Ambient Game Music Generator 🎵🎮")

# MAIN FUNCTION
def main():
  build_sidebar()
  uploaded_image, supporting_text, generate_music = build_music_generation_section()
  
  if generate_music:
    is_music_generated = build_show_processes_section()
    build_music_player_section(is_music_generated)

if __name__ == "__main__":
  main()
