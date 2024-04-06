# IMPORTS
import streamlit as st
import time
import sys
import os
from src.main import playWrite, generate
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
from utils.generate_audio_file_name import generate_audio_file_name

def build_show_generation_processes_section(supporting_text, uploaded_image):
  process_steps = [
    "Captioning Image...",
    "Creating Music Generation Prompt...",
    "Generating Music..."
  ]

  process_steps_success = [
    "Image Successfully Captioned!",
    "Music Generation Prompt Successfully Created!",
    "Music Successfully Generated!"
  ]

  dynamic_input = []

  # INTEGRATION OF MODEL
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  PlayWrite = playWrite(device=device,
                        vocab_path='resources/Vocabulary.pkl',
                        image_caption_path='models/image_captioning/model.pt',
                        hg_access_token=None,
                        llama_model_path='models/llama/model',
                        llama_tokenizer_path='models/llama/tokenizer')

  st.markdown("---")
  st.markdown("## The Music Generation Process")
  st.markdown("###")

  for index, (step, success_message) in enumerate(zip(process_steps, process_steps_success)):
    status_text = st.empty()
    progress_bar = st.empty()

    for percent_complete in range(101):
      status_text.markdown(f"{step}")
      progress_bar.progress(percent_complete)
      time.sleep(0.05)

    status_text.empty()
    progress_bar.empty()
    
    if index < len(dynamic_input):
      st.success(f"{success_message}\n\nOutput: {dynamic_input[index]}")
    else:
      st.success(f"{success_message}")

    time.sleep(1)

  st.markdown("---")
  st.markdown(f"SHOWWW {generate_audio_file_name()}")

  is_music_generated = True
  audio_file_name = 'placeholder_audio.mp3'

  return is_music_generated, audio_file_name
