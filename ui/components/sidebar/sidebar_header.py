# IMPORTS
import streamlit as st
from utils.display_image import display_image

def sidebar_header():
  st.markdown(display_image("images/logo.png", width=280, height=280), unsafe_allow_html=True)
  st.markdown("---")
  st.markdown('# User Controls')
