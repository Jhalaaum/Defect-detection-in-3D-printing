import cv2
import numpy as np

label_mask = cv2.imread('stitched_masks/22.png', cv2.IMREAD_GRAYSCALE)

colors = {
    0: (0, 0, 0),
    1: (0, 255, 0),
    2: (0, 165, 255),
    3: (0, 0, 0),
    4: (0, 0, 255),
    5: (255, 255, 255),
}

color_vis = np.zeros((label_mask.shape[0], label_mask.shape[1], 3), dtype=np.uint8)

for label_id, color in colors.items():
    color_vis[label_mask == label_id] = color

cv2.imshow('Label Mask Visualization', color_vis)
cv2.waitKey(0)
cv2.destroyAllWindows()
