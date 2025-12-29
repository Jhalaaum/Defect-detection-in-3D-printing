import json
import numpy as np
import cv2

# -------- CONFIG --------
json_path = "Scripts/3D Voxel/annotation.json"   # your JSON file
mask_out = "mask.png"           # output mask
# ------------------------

LABEL_MAP = {
    "crack": 1,
    "LoF": 2,
    "incomplete melting induced porosities": 3,
    "Entrapped gas porosity": 4
}

# Load annotation
with open(json_path, "r") as f:
    data = json.load(f)

height = data["height"]
width = data["width"]
boxes = data["boxes"]

# Create empty mask
mask = np.zeros((height, width), dtype=np.uint8)

# Fill polygons
for obj in boxes:
    label = obj["label"]
    if label not in LABEL_MAP:
        continue

    class_id = LABEL_MAP[label]

    pts = np.array(obj["points"], dtype=np.int32)
    pts = pts.reshape((-1, 1, 2))  # OpenCV format

    cv2.fillPoly(mask, [pts], class_id)

# Save mask
cv2.imwrite(mask_out, mask)

print("Mask saved to:", mask_out)
print("Unique values in mask:", np.unique(mask))
