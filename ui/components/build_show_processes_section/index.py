# IMPORT LIBRARIES
import streamlit as st
import time

def build_show_processes_section():
  process_steps = [
    "Captioning Image...",
    "Combining Caption & Supporting Text...",
    "Generating Music..."
  ]

  process_steps_success = [
    "Image Successfully Captioned!",
    "Image Caption & Supporting Text Successfully Combined!",
    "Music Successfully Generated!"
  ]

  placeholders = [
    "Caption Example",
    "Combined Text Example"
  ]

  st.markdown("---")
  st.markdown("## Logging Music Generation Process")
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
    
    if index < len(placeholders):
      st.success(f"{success_message}\n\nOutput: {placeholders[index]}")
    else:
      st.success(f"{success_message}")

    time.sleep(1)

  st.markdown("---")

  is_music_generated = True

  return is_music_generated
