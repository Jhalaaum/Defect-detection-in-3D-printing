import os

folder = "Scripts/3D Voxel/masks"  # Replace with your folder path

for filename in os.listdir(folder):
    if filename.endswith(".png"):
        # Remove the unwanted suffix
        new_name = filename.split("_png.rf")[0] + ".png"
        old_path = os.path.join(folder, filename)
        new_path = os.path.join(folder, new_name)
        os.rename(old_path, new_path)
        print(f"{filename} -> {new_name}")
