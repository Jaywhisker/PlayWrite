# IMPORTS
import streamlit as st
from .sidebar_header import sidebar_header
from .mustango_params import mustango_params

def build_sidebar(disabled=False):
  with st.sidebar:
    sidebar_header()
    st.markdown("###")

    steps = mustango_params(disabled=disabled)
    st.markdown("###")

    st.markdown('---')

    return steps
