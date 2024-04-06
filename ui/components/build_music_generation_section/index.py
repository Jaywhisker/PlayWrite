# IMPORTS
import streamlit as st
from .input_supporting_text import input_supporting_text
from .upload_image import upload_image
from .generate_music_button import generate_music_button

def build_music_generation_section(disabled=False):
  st.markdown("### Music Generation Input")
  st.markdown("###")

  supporting_text = input_supporting_text(disabled=disabled)
  st.markdown("###")

  uploaded_image = upload_image(disabled=disabled)
  st.markdown("###")

  start_music_generation = generate_music_button(supporting_text, uploaded_image)

  return supporting_text, uploaded_image, start_music_generation
