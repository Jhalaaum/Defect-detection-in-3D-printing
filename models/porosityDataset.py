import torch
from torch.utils.data import Dataset
import cv2
import os
from PIL import Image
import torchvision.transforms as T

class porosity_Dataset(Dataset):
    def __init__(self, image_path, mask_path, x, mean, std, transform=None, patch=False,
                 image_ext=".jpg", mask_ext=".png"):
        self.img_path = image_path
        self.mask_path = mask_path
        self.x = x  # list of base filenames (without extension)
        self.mean = mean
        self.std = std
        self.transform = transform
        self.patch = patch
        self.image_ext = image_ext
        self.mask_ext = mask_ext
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        image_filename = self.x[idx] + self.image_ext
        mask_filename = self.x[idx] + "_mask" + self.mask_ext

        img_path = os.path.join(self.img_path, image_filename)
        mask_path = os.path.join(self.mask_path, mask_filename)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        # else img is numpy array from cv2 already

        img = Image.fromarray(img)
        t = T.Compose([
            T.ToTensor(),
            T.Normalize(self.mean, self.std)
        ])
        img = t(img)

        mask = torch.tensor(mask, dtype=torch.long)  # convert to tensor explicitly

        if self.patch:
            img, mask = self.tiles(img, mask)

        return img, mask

    def tiles(self, img, mask, threshold=0.01):
        size = 256
        img_patches = img.unfold(1, size, size).unfold(2, size, size)
        img_patches = img_patches.contiguous().view(3, -1, size, size)
        img_patches = img_patches.permute(1, 0, 2, 3)

        mask_patches = mask.unfold(0, size, size).unfold(1, size, size)
        mask_patches = mask_patches.contiguous().view(-1, size, size)

        keep_indices = []
        for i, patch in enumerate(mask_patches):
            fg = (patch != 0).sum().item()
            ratio = fg / (size * size)
            if ratio > threshold:
                keep_indices.append(i)

        if not keep_indices:
            keep_indices = [0]  # Always return at least one tile

        return img_patches[keep_indices], mask_patches[keep_indices]
