import random
import string

def generate_audio_file_name(length=16):
  characters = string.ascii_letters + string.digits
  randomize_characters = ''.join(random.choices(characters, k=length))

  return f'{randomize_characters}.wav'
