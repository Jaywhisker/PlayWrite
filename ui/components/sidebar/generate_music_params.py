# IMPORT LIBRARIES
import streamlit as st

def generate_music_params():
  st.write('### Parameters for Music Generating Model (Mustango)')

  num_epochs = st.slider('Number of Epochs', min_value=1, max_value=100, value=10)

  with st.expander("What are epochs?"):
    st.write("An epoch is one complete pass through the entire dataset during training. "
      "For example, 10 epochs mean the model will see the entire dataset 10 times.")
  st.markdown("###")
    
  steps = st.slider('Number of Steps', min_value=1, max_value=100, value=10)

  with st.expander("What are steps?"):
    st.write("A step is the number of batches of data the model processes before making an update to its weights. "
      "Each step processes a subset of the dataset, called a batch, and makes predictions based on the current state of the model, "
      "then compares these predictions to the actual outcomes to adjust the model's weights. "
      "For instance, if your dataset consists of 1000 samples and you use a batch size of 100, "
      "it will take 10 steps to complete one epoch.")
    
  return num_epochs, steps
