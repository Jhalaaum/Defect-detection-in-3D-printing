import numpy as np
import h5py
from skimage import measure
import trimesh

with h5py.File("voxel_volume.h5", "r") as f:
    binary_volume = f["binary_volume"][:]
    voxel_size_mm = f.attrs["voxel_size_mm"]

binary_volume = np.transpose(binary_volume, (1, 2, 0))  # (H, W, Z)

verts, faces, normals, values = measure.marching_cubes(binary_volume, level=0.5)

verts *= voxel_size_mm
target_reduction = 0.95  # remove 95% of faces

mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
mesh = mesh.simplify_quadric_decimation(target_reduction)

print(len(mesh.faces))

mesh.export("porosity_mesh.stl", file_type='stl')

