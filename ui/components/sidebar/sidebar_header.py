# IMPORT LIBRARIES
import streamlit as st

# IMPORT UTILITY FUNCTIONS
from utils.display_image import display_image

def sidebar_header():
  st.markdown(display_image("images/logo.png"), unsafe_allow_html=True)
  st.markdown("---")
  st.markdown('# User Controls')
