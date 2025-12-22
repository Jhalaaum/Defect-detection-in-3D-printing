from PIL import Image
import os

input_folder = "Picture/"
output_folder = "Picture2"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".tif", ".tiff")):
        tif_path = os.path.join(input_folder, filename)
        img = Image.open(tif_path)
        jpg_filename = os.path.splitext(filename)[0] + ".jpg"
        jpg_path = os.path.join(output_folder, jpg_filename)
        img.convert("RGB").save(jpg_path, "JPEG")
        print(f"Converted: {filename} -> {jpg_filename}")