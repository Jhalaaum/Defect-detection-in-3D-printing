import os
import shutil

mixed_folder = "models/train/"

images_folder = os.path.join(mixed_folder, "images")
masks_folder = os.path.join(mixed_folder, "masks")

os.makedirs(images_folder, exist_ok=True)
os.makedirs(masks_folder, exist_ok=True)

for filename in os.listdir(mixed_folder):
    filepath = os.path.join(mixed_folder, filename)

    if os.path.isfile(filepath):
        if filename.lower().endswith(".jpg"):
            shutil.move(filepath, os.path.join(images_folder, filename))
        elif filename.lower().endswith(".png"):
            shutil.move(filepath, os.path.join(masks_folder, filename))

