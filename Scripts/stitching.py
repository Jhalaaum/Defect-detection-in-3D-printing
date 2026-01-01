import os
from PIL import Image
import re

folder = "Scripts/3D Voxel/masks"
output_folder = "./stitched_masks"
os.makedirs(output_folder, exist_ok=True)

images_by_z = {}

for filename in os.listdir(folder):
    if filename.endswith(".png"):
        match = re.match(r"(\d+)_cropped_(\d+)_(\d+)\.png", filename)
        if match:
            z, x_idx, y_idx = map(int, match.groups())
            img = Image.open(os.path.join(folder, filename))
            images_by_z.setdefault(z, []).append((x_idx, y_idx, img))

for z, imgs in images_by_z.items():
    max_x = max(x for x, y, _ in imgs)
    max_y = max(y for x, y, _ in imgs)

    tile_w, tile_h = imgs[0][2].size
    full_w = max_x * tile_w
    full_h = max_y * tile_h

    stitched = Image.new("RGB", (full_w, full_h))

    for x_idx, y_idx, img in imgs:
        stitched.paste(img, ((x_idx-1)*tile_w, (y_idx-1)*tile_h))

    output_path = os.path.join(output_folder, f"{z}.png")
    stitched.save(output_path)
