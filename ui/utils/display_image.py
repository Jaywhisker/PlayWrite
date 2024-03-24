import base64

def convert_image_to_base64(image_path):
  with open(image_path, "rb") as img_file:
    return base64.b64encode(img_file.read()).decode()

def display_image(image_path):
  img_base64 = convert_image_to_base64(image_path)
  return f'<img src="data:image/png;base64,{img_base64}">'
