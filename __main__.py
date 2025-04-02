import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt

from dataset import *
from models.srcnn import SRCNN
from models.srgan import SRGANgenerator, SRGANdiscriminator
from models.swinir import SwinIR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

print('Loading datasets...')
train_dataset = DIV2KDataset("patches/DIV2K_train_HR_480", "patches/DIV2K_train_LR_240")
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

lr, hr = next(iter(train_loader))

print('Loading models...')
srcnn = SRCNN().to(device)
preprocess = torch.nn.Upsample(scale_factor=2, mode='bicubic')
srgan_generator = SRGANgenerator().to(device)
srgan_discriminator = SRGANdiscriminator().to(device)
swinir = SwinIR().to(device)

with torch.no_grad():
    print('Calculating SRCNN output...')
    out_srcnn = srcnn(preprocess(lr))
    print('Calculating SRGAN output...')
    out_srgan = srgan_generator(lr)
    out_srgan_disc = srgan_discriminator(out_srgan)
    print('Discriminator output:', out_srgan_disc.squeeze(0).cpu().numpy())
    print('Calculating SwinIR output...')
    out_swinir = swinir(lr)

print('Visualizing output...')
hr_img = ToPILImage()(hr.squeeze(0).cpu())
srcnn_img = ToPILImage()(out_srcnn.squeeze(0).cpu())
srgan_img = ToPILImage()(out_srgan.squeeze(0).cpu())
swinir_img = ToPILImage()(out_swinir.squeeze(0).cpu())
plt.subplot(2, 2, 1)
plt.title('High Resolution Image')
plt.imshow(hr_img)
plt.axis('off')
plt.subplot(2, 2, 2)
plt.title('SRCNN Output Image')
plt.imshow(srcnn_img)
plt.axis('off')
plt.subplot(2, 2, 3)
plt.title('SRGAN Output Image')
plt.imshow(srgan_img)
plt.axis('off')
plt.subplot(2, 2, 4)
plt.title('SwinIR Output Image')
plt.imshow(swinir_img)
plt.axis('off')
plt.show()