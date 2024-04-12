# IMPORTS
import streamlit as st

def input_supporting_text(disabled=False):
  return st.text_input("Supporting Image Text (Required)", placeholder="Enter image supporting text here...", disabled=disabled)
