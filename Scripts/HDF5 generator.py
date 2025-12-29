import os
import cv2
import numpy as np
import h5py
import json

def sortByNumber(files):
    return sorted(files, key=lambda x: int(os.path.splitext(x)[0]))

input_dir = "stitched_masks"
output_dir = "./binary_masks"
os.makedirs(output_dir, exist_ok=True)

label_to_binary = {
    0: 0,
    1: 1,
    2: 1,
    3: 0,
    4: 1,
    5: 1,
}
label = {
    0: 0,
    1: 1,
    2: 2,
    3: 0,
    4: 3,
    5: 4,
}

files = sortByNumber([f for f in os.listdir(input_dir) if f.endswith(".png")])
binaryVolume = []
multiClassVolume=[]

for fname in files:
    if fname.endswith(".png"):
        mask = cv2.imread(os.path.join(input_dir, fname), cv2.IMREAD_GRAYSCALE)
        remapped = np.zeros_like(mask, dtype=np.uint8)
        for src, dst in label.items():
            remapped[mask == src] = dst
        multiClassVolume.append(remapped)

        binary = np.zeros_like(mask, dtype=np.uint8)
        for src, dst in label_to_binary.items():
            binary[mask == src] = dst
        binaryVolume.append(binary)
        # binary_mask = np.zeros(mask.shape, dtype=np.uint8)

        # for label_id, binary_val in label_to_binary.items():
        #     binary_mask[mask == label_id] = binary_val
        # cv2.imwrite(
        #     os.path.join(output_dir, fname),
        #     binary_mask * 255
        # )


# for fname in sorted(os.listdir(output_dir)):
#     if fname.endswith(".png"):
#         img = cv2.imread(os.path.join(output_dir, fname), cv2.IMREAD_GRAYSCALE)
#         volume.append((img > 0).astype(np.uint8))


volume = np.stack(binaryVolume, axis=0)
multiClassVolume = np.stack(multiClassVolume, axis=0)


with h5py.File("voxel_volume.h5", "w") as f:
    f.create_dataset(
        "binary_volume",
        data=volume,
        dtype=np.uint8,
        compression="gzip",
        compression_opts=4
    )

    f.create_dataset(
        "multiclass_volume",
        data=multiClassVolume,
        dtype=np.uint8,
        compression="gzip",
        compression_opts=4
    )

    f.attrs["voxel_size_mm"] = 0.005
    f.attrs["material"] = "316L_stainless_steel"
    f.attrs["build_direction"] = "z"
    f.attrs["porosity_pipeline"] = "U-Net segmentation"

    f.attrs["multiclass_definition"] = json.dumps({
        0: "background",
        1: "Entrapped gas porosity",
        2: "LoF",
        3: "crack",
        4: "incomplete melting induced porosities"
    })

    f.attrs["binary_definition"] = json.dumps({
        0: "solid",
        1: "void"
    })
