# IMPORT LIBRARIES
import streamlit as st

# IMPORT COMPONENTS
from .sidebar_header import sidebar_header
from .caption_image_params import caption_image_params
from .generate_music_params import generate_music_params

def build_sidebar():
  with st.sidebar:
    sidebar_header()
    st.markdown("###")

    sample_param_1, sample_param_2, sample_param_3 = caption_image_params()
    st.markdown("###")

    num_epochs = generate_music_params()
    st.markdown("###")

    st.markdown('---')

    return sample_param_1, sample_param_2, sample_param_3, num_epochs
