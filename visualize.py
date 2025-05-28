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

def model_from_file(model: nn.Module, device: torch.device, path: str):
    try:
        model.load_state_dict(torch.load(path, weights_only=True, map_location=device))
        print(f'Loaded model from {path}')
    except FileNotFoundError:
        print(f'Couldn\'t load the model from {path}')

def image_output_from_model(model: nn.Module, device: torch.device, img: Image) -> Image:
    model = model.to(device)
    tensor: torch.Tensor = transforms.ToTensor()(img).to(device).unsqueeze(0)

    if type(model) is SRCNN:
        tensor = nn.Upsample(scale_factor=2, mode='bicubic')(tensor)

    with torch.no_grad():
        print(f'Calculating {model._get_name()} output...')
        output: torch.Tensor = model(tensor)

    return ToPILImage()(output.squeeze(0).cpu())

def subplot_image(
    grid: list[int],
    position: int,
    title: str,
    image: Image,
    share: plt.Axes = None
) -> plt.Axes:
    
    ax = plt.subplot(grid[0], grid[1], position,
        sharex = share,
        sharey = share
    )
    plt.title(title)
    plt.imshow(image)
    plt.axis('off')

    return ax

def visualize_all_models(hr: Image, lr: Image, device = torch.device('cpu'), srcnn_src: str = None, srgan_src: str = None, swinir_src: str = None):
    sources = [srcnn_src, srgan_src, swinir_src]
    num_of_models = len(sources) - sources.count(None) + 1

    plot_grid = {
        0: [1, 1],
        1: [1, 2],
        3: [1, 3],
        4: [2, 2]
    }[num_of_models]

    position = 1

    hr_axes = subplot_image(plot_grid, position, 'High Resolution Image', hr)

    for src in [srcnn_src, srgan_src, swinir_src]:
        if src is None:
            continue

        position += 1
        
        model: torch.Tensor = {
            srcnn_src: SRCNN(),
            srgan_src: SRGANgenerator(),
            swinir_src: SwinIR()
        }[src]

        model = model.to(device)

        model_from_file(model, device, src)
        output = image_output_from_model(model, device, lr)

        subplot_image(plot_grid, position, f'{model._get_name()} Output Image', output, hr_axes)
    
    print('Plotting images...')
    plt.show()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--srcnn', type=str, required=False)
    parser.add_argument('--srgan', type=str, required=False)
    parser.add_argument('--swinir', type=str, required=False)
    parser.add_argument('--lr', type=str, required=True)
    parser.add_argument('--hr', type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    hr_images = sorted(os.listdir(args.hr))
    lr_images = sorted(os.listdir(args.lr))

    index = np.random.randint(0, len(hr_images))

    hr_path = os.path.join(args.hr, hr_images[index])
    lr_path = os.path.join(args.lr, lr_images[index])

    hr_img = Image.open(hr_path).convert("RGB")
    lr_img = Image.open(lr_path).convert("RGB")

    visualize_all_models(hr_img, lr_img, device, args.srcnn, args.srgan, args.swinir)