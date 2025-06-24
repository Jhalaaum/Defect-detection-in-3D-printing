import os
import cv2
import numpy as np

input_folder = "Picture"
output_folder = "CSVs"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        img_path = os.path.join(input_folder, filename)
        
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Warning: Could not read {img_path}, skipping.")
            continue
        
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        
        binary_array = (binary == 255).astype(np.uint8)
        
        csv_filename = os.path.splitext(filename)[0] + '.csv'
        csv_path = os.path.join(output_folder, csv_filename)
        
        np.savetxt(csv_path, binary_array, fmt='%d', delimiter=',')
        print(f"Processed {filename} -> {csv_path}")