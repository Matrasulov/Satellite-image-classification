"""
Dataset and DataLoader utilities for satellite image classification.
"""

import os
import torch
from glob import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split


class SatelliteDataset(Dataset):
    """Custom dataset for satellite image classification."""
    
    def __init__(self, root, transformations=None):
        """
        Args:
            root (str): Root directory containing class folders
            transformations: torchvision transforms to apply
        """
        self.transformations = transformations
        self.im_paths = sorted(glob(f"{root}/*/*"))
        
        self.cls_names = {}
        self.cls_counts = {}
        
        count = 0
        for im_path in self.im_paths:
            class_name = self._get_class(im_path)
            if class_name not in self.cls_names:
                self.cls_names[class_name] = count
                self.cls_counts[class_name] = 1
                count += 1
            else:
                self.cls_counts[class_name] += 1
    
    def _get_class(self, path):
        """Extract class name from path."""
        return os.path.dirname(path).split("/")[-1]
    
    def __len__(self):
        return len(self.im_paths)
    
    def __getitem__(self, idx):
        im_path = self.im_paths[idx]
        im = Image.open(im_path).convert("RGB")
        gt = self.cls_names[self._get_class(im_path)]
        
        if self.transformations is not None:
            im = self.transformations(im)
        
        return im, gt


def get_dataloaders(root, transformations, batch_size=32, 
                   split=[0.9, 0.05, 0.05], num_workers=4):
    """
    Create train, validation, and test dataloaders.
    
    Args:
        root (str): Root directory of dataset
        transformations: Transforms to apply
        batch_size (int): Batch size for dataloaders
        split (list): Train/val/test split ratios
        num_workers (int): Number of workers for data loading
    
    Returns:
        tuple: (train_dl, val_dl, test_dl, class_names_dict)
    """
    dataset = SatelliteDataset(root=root, transformations=transformations)
    
    total_len = len(dataset)
    train_len = int(total_len * split[0])
    val_len = int(total_len * split[1])
    test_len = total_len - (train_len + val_len)
    
    train_ds, val_ds, test_ds = random_split(
        dataset=dataset, 
        lengths=[train_len, val_len, test_len]
    )
    
    train_dl = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    val_dl = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    test_dl = DataLoader(
        test_ds, 
        batch_size=1, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    return train_dl, val_dl, test_dl, dataset.cls_names
