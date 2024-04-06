import base64

def convert_image_to_base64(image_path):
  with open(image_path, "rb") as img_file:
    return base64.b64encode(img_file.read()).decode()

def display_image(image_path, width=None, height=None):
  img_base64 = convert_image_to_base64(image_path)
  size_style = ''
  if width or height:
    size_parts = []
    if width:
        size_parts.append(f'width: {width}px;')
    if height:
        size_parts.append(f'height: {height}px;')
    size_style = 'style="' + ' '.join(size_parts) + '"'
  return f'<img src="data:image/png;base64,{img_base64}" {size_style}>'
