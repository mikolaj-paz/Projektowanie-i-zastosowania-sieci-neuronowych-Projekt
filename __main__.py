import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
import cv2

from dataset import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = DIV2KDataset("dataset/DIV2K_train_HR", "dataset/DIV2K_train_LR_bicubic/X2")
valid_dataset = DIV2KDataset("dataset/DIV2K_valid_HR", "dataset/DIV2K_valid_LR_bicubic/X2")

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=True)

lr, hr = next(iter(train_loader))
print("LR size", lr.unsqueeze(0).shape)
print("HR size", hr.shape)

# img = 

# plt.imshow(img)
# plt.axis("off")
# plt.show()