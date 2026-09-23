"""
src/dataset.py - PyTorch Dataset and Data Augmentation Pipeline for 7 Rice Field States (Dry, Water, Wet, Green rice, Green weed, Straw, Others).
"""

import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from src.model import CLASSES, CLASS_TO_IDX


class RiceFieldDataset(Dataset):
    """
    Dataset that loads satellite crops across the 7 classes:
    - Dry
    - Water
    - Wet
    - Green rice
    - Green weed
    - Straw
    - Others
    """
    def __init__(self, dataset_dir="Dataset", image_size=224, num_samples=960, is_train=True):
        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.num_samples = num_samples
        self.is_train = is_train
        
        self.class_images = {cls: [] for cls in CLASSES}
        
        # Support case-insensitive class folder matching and known aliases
        class_map = {cls.lower(): cls for cls in CLASSES}
        class_map.update({
            "green_rice": "Green rice",
            "green_weed": "Green weed",
            "flood": "Water",
            "flooded": "Water",
            "planted": "Green rice",
            "other": "Others"
        })
        
        if os.path.exists(dataset_dir):
            for dir_name in os.listdir(dataset_dir):
                target_cls = class_map.get(dir_name.lower())
                if target_cls in self.class_images:
                    dir_path = os.path.join(dataset_dir, dir_name)
                    if os.path.isdir(dir_path):
                        for f in os.listdir(dir_path):
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                                img_path = os.path.join(dir_path, f)
                                try:
                                    img = Image.open(img_path).convert("RGB")
                                    self.class_images[target_cls].append(img)
                                except Exception as e:
                                    print(f"[!] Warning loading {img_path}: {e}")
                                    
        for cls_name, imgs in self.class_images.items():
            print(f"[*] Class '{cls_name:7s}': Loaded {len(imgs)} source image(s) from '{dataset_dir}'.")
            if len(imgs) == 0:
                print(f"[!] Warning: No images found for class '{cls_name}'.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        cls_idx = idx % len(CLASSES)
        cls_name = CLASSES[cls_idx]
        
        source_imgs = self.class_images[cls_name]
        if len(source_imgs) == 0:
            from src.model import CLASS_COLORS
            color = CLASS_COLORS.get(cls_name, (128, 128, 128))
            src_img = Image.new("RGB", (self.image_size, self.image_size), color=color)
        else:
            src_img = random.choice(source_imgs)
        
        w, h = src_img.size
        sz = self.image_size
        
        if w >= sz and h >= sz:
            if self.is_train:
                crop_scale = random.uniform(0.6, 1.0)
                crop_sz = int(min(w, h) * crop_scale)
                crop_sz = max(sz, crop_sz)
                
                left = random.randint(0, max(0, w - crop_sz))
                top = random.randint(0, max(0, h - crop_sz))
                patch = src_img.crop((left, top, left + crop_sz, top + crop_sz))
                patch = patch.resize((sz, sz), Image.Resampling.BILINEAR)
            else:
                left = (w - sz) // 2
                top = (h - sz) // 2
                patch = src_img.crop((left, top, left + sz, top + sz))
        else:
            patch = src_img.resize((sz, sz), Image.Resampling.BILINEAR)
            
        arr = np.array(patch, dtype=np.float32) / 255.0
        
        if self.is_train:
            if random.random() > 0.5:
                arr = np.fliplr(arr)
            if random.random() > 0.5:
                arr = np.flipud(arr)
            k = random.randint(0, 3)
            if k > 0:
                arr = np.rot90(arr, k)
                
            brightness = random.uniform(0.85, 1.15)
            contrast = random.uniform(0.85, 1.15)
            arr = np.clip((arr - 0.5) * contrast + 0.5, 0.0, 1.0)
            arr = np.clip(arr * brightness, 0.0, 1.0)
            
            noise = np.random.normal(0, 0.015, arr.shape)
            arr = np.clip(arr + noise, 0.0, 1.0)
            
        arr = np.ascontiguousarray(arr)
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        
        tensor = torch.tensor(arr, dtype=torch.float32).permute(2, 0, 1).contiguous()
        label = torch.tensor(cls_idx, dtype=torch.long)
        
        return tensor, label


def create_dataloaders(dataset_dir="Dataset", batch_size=16, num_train_samples=800, num_val_samples=160):
    train_ds = RiceFieldDataset(dataset_dir=dataset_dir, image_size=224, num_samples=num_train_samples, is_train=True)
    val_ds = RiceFieldDataset(dataset_dir=dataset_dir, image_size=224, num_samples=num_val_samples, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = create_dataloaders()
    images, labels = next(iter(train_loader))
    print("Dataset batch test successful!")
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
    print("Labels in batch:", [CLASSES[idx.item()] for idx in labels[:8]])
