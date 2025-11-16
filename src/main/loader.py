# -*- coding: utf-8 -*-

import os
from glob import glob
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T

def get_preprocess_transform(img_size=(256, 256)):
    return T.Compose([
        T.Resize(img_size),
        T.CenterCrop(img_size),
        T.ToTensor(),
    ])

class UltrasoundNeedleDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, transform=None):
        self.image_paths = sorted(glob(os.path.join(image_dir, "**", "*.png"), recursive=True))
        self.image_paths += sorted(glob(os.path.join(image_dir, "**", "*.jpg"), recursive=True))
        self.image_paths += sorted(glob(os.path.join(image_dir, "**", "*.jpeg"), recursive=True))

        if mask_dir:
            self.mask_paths = sorted(glob(os.path.join(mask_dir, "**", "*.png"), recursive=True))
            self.mask_paths += sorted(glob(os.path.join(mask_dir, "**", "*.jpg"), recursive=True))
            self.mask_paths += sorted(glob(os.path.join(mask_dir, "**", "*.jpeg"), recursive=True))
            assert len(self.image_paths) == len(self.mask_paths), "Images and masks must match 1-to-1"
            
        self.mask_dir = mask_dir
        self.transform = transform

        if mask_dir:
            self.mask_paths = sorted(glob(os.path.join(mask_dir, "*.png")))
            assert len(self.image_paths) == len(self.mask_paths), \
                "Images and masks must match 1-to-1"
        else:
            self.mask_paths = [None] * len(self.image_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("L")  #grayscale ultrasound

        #load mask
        mask_path = self.mask_paths[idx]
        if mask_path is not None:
            mask = Image.open(mask_path).convert("L")
        else:
            mask = None

        #apply transforms
        if self.transform:
            img = self.transform(img)
            if mask is not None:
                mask = T.ToTensor()(mask)

        return {
            "image": img,
            "mask": mask,
            "path": self.image_paths[idx]
        }

def create_dataloaders(
    image_dir,
    mask_dir=None,
    img_size=(256, 256),
    batch_size=8,
    val_split=0.15,
    test_split=0.15,
):
    transform = get_preprocess_transform(img_size)
    dataset = UltrasoundNeedleDataset(image_dir, mask_dir, transform)

    total = len(dataset)
    test_size = int(total * test_split)
    val_size = int(total * val_split)
    train_size = total - val_size - test_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_dl, val_dl, test_dl = create_dataloaders(
        image_dir="/content/simus/images",
        mask_dir="/content/simus/masks",
        batch_size=4
    )
    print("Train batches:", len(train_dl))
    print("Val batches:", len(val_dl))
    print("Test batches:", len(test_dl))
