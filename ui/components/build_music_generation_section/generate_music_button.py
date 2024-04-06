# IMPORTS
import streamlit as st

def generate_music_button(supporting_text, uploaded_image):
  def change_disabled_state():
    st.session_state.is_music_generation_processing = True
  
  if (supporting_text.strip() == '' or uploaded_image is None):
    button_disabled = True
  else:
    button_disabled = False

  start_music_generation = st.button("Generate Music",
                                    disabled=button_disabled or st.session_state.is_music_generation_processing,
                                    use_container_width=True,
                                    key="generate_music_btn",
                                    on_click=change_disabled_state,
                                    type="primary")
  return start_music_generation
