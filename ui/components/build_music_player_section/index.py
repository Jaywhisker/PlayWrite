# IMPORT LIBRARIES
import streamlit as st

def build_music_player_section(is_music_generated):
  st.markdown("### Music Player 🎵")
  st.markdown("###")

  if is_music_generated:
    audio_file = 'placeholder_audio.mp3'

    st.audio(audio_file, format='audio/mp3', start_time=0)
    st.markdown("Enjoy the Music!")
  else:
    st.markdown("Fill in the 'Music Generation Input' Section to Start Creating Music! ")
