# IMPORT LIBRARIES
import streamlit as st

# IMPORT COMPONENTS
from .sidebar_header import sidebar_header
from .generate_music_params import generate_music_params

def build_sidebar():
  with st.sidebar:
    sidebar_header()
    st.markdown("###")

    num_epochs, steps = generate_music_params()
    st.markdown("###")

    st.markdown('---')

    return num_epochs, steps
