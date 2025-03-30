from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import os
import random

class DIV2KDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, patch_size=480, scale=2):
        super().__init__()

        self.hr_dir = hr_dir
        self.lr_dir = lr_dir

        self.hr_images = sorted(os.listdir(hr_dir))
        self.lr_images = sorted(os.listdir(lr_dir))

        self.patch_size = patch_size
        self.scale = scale

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

        lr_patch, hr_patch = self.random_crop(hr_img, lr_img)

        hr_patch = self.transform(hr_patch)
        lr_patch = self.transform(lr_patch)

        return lr_patch, hr_patch

    def random_crop(self, hr_img, lr_img):
        h, w = hr_img.size[1], lr_img.size[0]

        top = random.randint(0, h - self.patch_size)
        left = random.randint(0, w - self.patch_size)

        hr_patch = TF.crop(hr_img, top, left, self.patch_size, self.patch_size)
        lr_patch = TF.crop(lr_img, top // self.scale, left // self.scale,
                           self.patch_size // self.scale, self.patch_size // self.scale)
        
        return lr_patch, hr_patch