# IMPORT LIBRARIES
import streamlit as st

def caption_image_params():
  st.write('### Parameters for Image Captioning Model')
  
  sample_param_1 = st.checkbox('Sample Parameter 01')
  sample_param_2 = st.checkbox('Sample Parameter 02')
  sample_param_3 = st.radio('Choose a Sample Parameter 03:', ('Parameter A', 'Parameter B', 'Parameter C'))

  return sample_param_1, sample_param_2, sample_param_3
