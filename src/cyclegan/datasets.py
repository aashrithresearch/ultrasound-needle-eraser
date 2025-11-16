# -*- coding: utf-8 -*-

import os
from glob import glob
from PIL import Image
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CycleGANDataset(Dataset):
    def __init__(self, root_dir, img_size=256):
        self.paths = []
        for sub in ["1", "2", "3", "4"]:
            self.paths.extend(
                sorted(glob(os.path.join(root_dir, sub, "*.png")))
            )
            self.paths.extend(
                sorted(glob(os.path.join(root_dir, sub, "*.jpg")))
            )

        assert len(self.paths) > 0, "No images found in dataset."

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # grayscale [-1,1]
        ])

        self.A_paths = self.paths
        self.B_paths = self.paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        #random a
        img_A_path = random.choice(self.A_paths)
        img_A = Image.open(img_A_path).convert("L")

        #random b
        img_B_path = random.choice(self.B_paths)
        img_B = Image.open(img_B_path).convert("L")

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B}
