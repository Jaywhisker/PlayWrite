# IMPORT LIBRARIES
import streamlit as st

# IMPORT COMPONENTS
from .upload_image import upload_image
from .input_supporting_text import input_supporting_text
from .generate_music_button import generate_music_button

def build_music_generation_section():
  st.markdown("###")
  uploaded_image = upload_image()
  st.markdown("###")
  supporting_text = input_supporting_text()
  st.markdown("###")
  generate_music = generate_music_button(uploaded_image)
  st.markdown("---")

  return uploaded_image, supporting_text, generate_music
