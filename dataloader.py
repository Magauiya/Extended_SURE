
import os
import glob
import random
import numpy as np
from PIL import Image

from scipy.io.matlab.mio import savemat, loadmat

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision import transforms

class BSD500_Dataset(Dataset):
    def __init__(self, cfg, img_list):
        self.img_list = img_list
        self.cfg = cfg
        self.to_tensor = transforms.ToTensor()
        self.augment = transforms.Compose([
            transforms.RandomResizedCrop(size=(cfg.img_size, cfg.img_size), 
                                         scale=(0.75, 1.0), 
                                         ratio=(1.0, 1.0), 
                                         interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5)
        ])
        
    def __len__(self):
        return len(self.img_list)

    def generate(self, clean):
        # Random crop
        clean = self.augment(clean)
        sigma = np.random.uniform(self.cfg.min_std, self.cfg.max_std, (1, )).astype(np.float32)
        noisy = clean + np.random.normal(0, sigma, np.shape(clean))
        
        clean = self.to_tensor(clean) * 255 # because ToTensor normalized to [0-1]
        noisy = self.to_tensor(noisy)   # [0-255]
        sigma = torch.from_numpy(sigma) # [0-255]
        
        return noisy, clean, sigma

    def __getitem__(self, idx):
        img_path = self.img_list[idx]             # Select sample
        clean = Image.open(img_path).convert('L') # Load input img
        return self.generate(clean)
