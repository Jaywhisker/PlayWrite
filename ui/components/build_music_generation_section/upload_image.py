# IMPORT LIBRARIES
import streamlit as st
from PIL import Image

def upload_image():
  uploaded_file = st.file_uploader("Upload Image for Ambient Sound Generation (Required)", type=["jpg", "jpeg", "png"], help="Drag and drop or click to upload an image")

  if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, use_column_width=True)

  return uploaded_file
