# IMPORTS
import streamlit as st
import time
import sys
import os
import gc
import torch
import soundfile as sf
import pickle

original_sys = sys.path
# CONFIGURE FILE PATHS
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
models_path = os.path.join(project_root, 'models')
resources_path = os.path.join(project_root, 'resources')
audio_files_path = os.path.join(project_root, 'resources', 'music_results')

sys.path.append(project_root)
from src.data.dataloader import Vocabulary
from src.main import playWrite

sys.path = original_sys
from utils.generate_audio_file_name import generate_audio_file_name

def empty_cuda_cache(playwrite, index):
  if index == 0:
    del playwrite.image_caption
  elif index == 1:
    del playwrite.llama_model

  torch.cuda.empty_cache()
  gc.collect()


def build_show_generation_processes_section(supporting_text, uploaded_image, steps):
  st.markdown("---")
  st.markdown("## The Music Generation Process")
  st.markdown("###")

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

  text_prompt = supporting_text
  byte_image = uploaded_image.getvalue()
  is_music_generated = False
  audio_file_name = ''
  image_caption = ''
  music_prompt = ''
  
  # INTEGRATION OF MODEL
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  PlayWrite = playWrite(device=device,
                        vocab_path=os.path.join(resources_path, 'Vocabulary.pkl'),
                        image_caption_path=os.path.join(models_path, 'image_captioning', 'model.pt'),
                        hg_access_token=None,
                        llama_model_path=os.path.join(models_path, 'llama', 'model'),
                        llama_tokenizer_path=os.path.join(models_path, 'llama', 'tokenizer'))
  # PROCESSING STEPS
  for index, (step, success_message) in enumerate(zip(process_steps, process_steps_success)):
    with st.spinner(f"{step}"):
      if index == 0:
        image_caption = PlayWrite.caption_image(byte_image, model=PlayWrite.image_caption, vocab=PlayWrite.vocab, device=PlayWrite.device, max_length=50)
        output_message = f"{success_message}\n\nOutput: {image_caption}"

      elif index == 1:
        music_prompt = PlayWrite.generate_music_prompt(caption=image_caption, text_prompt=text_prompt, llama_model=PlayWrite.llama_model, llama_tokenizer=PlayWrite.llama_tokenizer, device=PlayWrite.device)
        output_message = f"{success_message}\n\nOutput: {music_prompt}"

      elif index == 2:
        """
          If we were to have more computing power, then the commented out code is used to connect and load the music from Mustango.
          As for demonstration purposes and the usability of this UI, we are simply simulating how this process would work using a placehlder mp3 file.
        """
        # music = PlayWrite.generate_music(music_prompt=music_prompt, model=PlayWrite.mustango, steps=steps, guidance=3)
        # audio_file_name = generate_audio_file_name()
        # audio_file_path = os.path.join(audio_files_path, audio_file_name)
        # sf.write(audio_file_path, music, samplerate=16000)
        audio_file_name = 'placeholder.mp3'
        output_message = success_message
        is_music_generated = True

      st.success(output_message)
      empty_cuda_cache(PlayWrite, index)

  st.markdown("---")

  return is_music_generated, audio_file_name


if __name__ == "__main__":
  print("import complete")