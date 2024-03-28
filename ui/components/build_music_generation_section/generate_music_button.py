# IMPORT LIBRARIES
import streamlit as st

def generate_music_button(supporting_text, uploaded_image):
  if 'disable_generate_music' not in st.session_state:
    st.session_state.disable_generate_music = False

  def change_disabled_state():
    st.session_state.disable_generate_music = True

  if (supporting_text.strip() != '') and (uploaded_image is not None):
    st.session_state.disable_generate_music = False
  else:
    st.session_state.disable_generate_music = True

  generate_music = st.button("Generate Music", disabled=st.session_state.disable_generate_music, use_container_width=True, key="generate_music_btn", on_click=change_disabled_state, type="primary")
  return generate_music
