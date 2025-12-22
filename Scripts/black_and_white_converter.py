import cv2
import os

input_folder = 'Picture2'
output_folder = 'Grayscale_pictures'

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        img_path = os.path.join(input_folder, filename)

        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, img_gray)

print("Grayscale conversion done.")