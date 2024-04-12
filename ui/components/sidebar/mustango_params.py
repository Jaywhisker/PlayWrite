# IMPORTS
import streamlit as st

def mustango_params(disabled=False):
  st.write('### Parameters for Music Generating Model (Mustango)')
    
  steps = st.slider('Number of Steps', min_value=1, max_value=200, value=150, disabled=disabled)

  with st.expander("What are steps?"):
    st.write("A step is the number of batches of data the model processes before making an update to its weights. "
      "Each step processes a subset of the dataset, called a batch, and makes predictions based on the current state of the model, "
      "then compares these predictions to the actual outcomes to adjust the model's weights. "
      "For instance, if your dataset consists of 1000 samples and you use a batch size of 100, "
      "it will take 10 steps to complete one epoch.")
    
  return steps
