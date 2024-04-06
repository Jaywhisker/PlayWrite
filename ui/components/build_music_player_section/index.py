# IMPORTS
import streamlit as st
import os

# CONFIGURE FILE PATHS
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
audio_files_path = os.path.join(project_root, 'ui', 'audio_files')

def build_music_player_section(is_music_generated, audio_file_name):
  st.markdown("### Music Player 🎵")
  st.markdown("###")

  if is_music_generated:
    audio_file_path = os.path.join(audio_files_path, audio_file_name)
    st.audio(audio_file_path, format='audio/mp3', start_time=0)
    st.markdown("Enjoy the Music!")
  else:
    st.markdown("Fill in the 'Music Generation Input' Section to Start Creating Music! ")
