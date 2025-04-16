import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import os
import random

class DIV2KDataset(Dataset):
    def __init__(self, hr_dir, lr_dir):
        super().__init__()

        self.hr_dir = hr_dir
        self.lr_dir = lr_dir

        self.hr_images = sorted(os.listdir(hr_dir))
        self.lr_images = sorted(os.listdir(lr_dir))

        assert len(self.hr_images) == len(self.lr_images)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, index):
        hr_path = os.path.join(self.hr_dir, self.hr_images[index])
        lr_path = os.path.join(self.lr_dir, self.lr_images[index])

        hr_img = Image.open(hr_path).convert("RGB")
        lr_img = Image.open(lr_path).convert("RGB")

        hr_img = self.transform(hr_img)
        lr_img = self.transform(lr_img)

        return lr_img, hr_img
    
class PatchDataset(Dataset):
    def __init__(self, pt_src):
        data = torch.load(pt_src)
        self.inputs = data['inputs']
        self.targets = data['targets']

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]