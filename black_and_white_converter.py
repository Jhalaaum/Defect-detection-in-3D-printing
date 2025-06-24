import cv2
import os

input_folder = 'Picture'
output_folder = 'Black_and_white_pictures'

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        _, bw = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
        cv2.imwrite(os.path.join(output_folder, filename), bw)

print("Conversion done.")
