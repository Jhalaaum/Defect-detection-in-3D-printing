import h5py
from scipy.ndimage import label
import numpy as np
from skimage.measure import regionprops
import matplotlib.pyplot as plt

with h5py.File("voxel_volume.h5", "r") as f:
    volume = f["binary_volume"][:]

structure = np.ones((3, 3, 3), dtype=np.uint8)
labeled_volume, num_pores = label(volume, structure=structure)

props = regionprops(labeled_volume)

pore_volumes = [p.area for p in props]
centroids = [p.centroid for p in props]
total_voxels = volume.size
voxels_of_pores = volume.sum()

fraction_porosity = voxels_of_pores / total_voxels

print("Porosity fraction:", fraction_porosity)
print("Number of pores detected:", num_pores)

plt.hist(pore_volumes, bins=3000)
plt.xlabel("Pore volume (voxels)")
plt.yscale("log")
plt.xlim(0, 2000)
plt.ylabel("Frequency")
plt.title("Pore Size Distribution")

plt.savefig('Pore Size Distribution.png')

plt.show()

