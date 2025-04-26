import torch
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from PIL import Image

from models.srcnn import SRCNN
from models.srgan import SRGANgenerator, SRGANdiscriminator
from models.swinir import SwinIR

def model_from_file(model: nn.Module, path: str):
    try:
        model.load_state_dict(torch.load(path, weights_only=True))
        print(f'Loaded model from {path}')
    except FileNotFoundError:
        print(f'Couldn\'t load the model from {path}')

def image_output_from_model(model: nn.Module, device: torch.device, img: Image):
    model = model.to(device)
    tensor: torch.Tensor = transforms.ToTensor()(img).to(device).unsqueeze(0)

    if type(model) is SRCNN:
        tensor = nn.Upsample(scale_factor=2, mode='bicubic')(tensor)

    with torch.no_grad():
        print(f'Calculating {model._get_name()} output...')
        output: torch.Tensor = model(tensor)

    return ToPILImage()(output.squeeze(0).cpu())

def visualize_all_models(srcnn_src: str, srgan_src: str, swinir_src: str, device: torch.device, hr: Image, lr: Image):
    srcnn = SRCNN().to(device)
    model_from_file(srcnn, srcnn_src)

    srgan = SRGANgenerator().to(device)
    model_from_file(srgan, srgan_src)

    swinir = SwinIR().to(device)
    model_from_file(swinir, swinir_src)

    out_srcnn = image_output_from_model(srcnn, device, lr)
    out_srgan = image_output_from_model(srgan, device, lr)
    out_swinir = image_output_from_model(swinir, device, lr)

    plt.subplot(2, 2, 1)
    plt.title('High Resolution Image')
    plt.imshow(hr)
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title('SRCNN Output Image')
    plt.imshow(out_srcnn)
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.title('SRGAN Output Image')
    plt.imshow(out_srgan)
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title('SwinIR Output Image')
    plt.imshow(out_swinir)
    plt.axis('off')

    plt.show()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--srcnn', type=str, required=True)
    parser.add_argument('--srgan', type=str, required=True)
    parser.add_argument('--swinir', type=str, required=True)
    parser.add_argument('--lr', type=str, required=True)
    parser.add_argument('--hr', type=str, required=True)
    args = parser.parse_args()

    device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    hr_images = sorted(os.listdir(args.hr))
    lr_images = sorted(os.listdir(args.lr))

    index = np.random.randint(0, len(hr_images))

    hr_path = os.path.join(args.hr, hr_images[index])
    lr_path = os.path.join(args.lr, lr_images[index])

    hr_img = Image.open(hr_path).convert("RGB")
    lr_img = Image.open(lr_path).convert("RGB")

    visualize_all_models(args.srcnn, args.srgan, args.swinir, device, hr_img, lr_img)