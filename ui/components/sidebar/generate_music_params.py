# IMPORT LIBRARIES
import streamlit as st

def generate_music_params():
  st.write('### Parameters for Music Generating Model (Mustango)')

  num_epochs = st.slider('Number of Epochs', min_value=1, max_value=100, value=10)

  with st.expander("What are epochs?"):
    st.write("An epoch is one complete pass through the entire dataset during training. "
      "For example, 10 epochs mean the model will see the entire dataset 10 times.")
    
  return num_epochs
