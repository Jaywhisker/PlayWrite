# IMPORTS
import streamlit as st
from PIL import Image

def upload_image(disabled=False):
  uploaded_image = st.file_uploader("Image Upload (Required)", type=["jpg", "jpeg", "png"], help="Drag and drop or click to upload an image", disabled=disabled)

  if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
      st.image(image, width=300)

  return uploaded_image
