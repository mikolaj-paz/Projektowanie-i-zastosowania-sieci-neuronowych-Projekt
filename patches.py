import os
from PIL import Image
import numpy as np
from patchify import patchify
from tqdm import tqdm
import torch
import torchvision.transforms as T

def make_patches(src_dir: str, dst_dir: str, patch_size: int, stride: int, scale: int = 1):
    images = sorted(os.listdir(src_dir))

    print(f'Creating patches from {src_dir} and saving to {dst_dir}...')
    for k in tqdm(range(len(images))):
        img_name = images[k]
        img_path = os.path.join(src_dir, img_name)
        img = np.array(Image.open(img_path).convert('RGB'))
        
        patches = patchify(img, (patch_size, patch_size, 3), stride)

        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                patch = patches[i, j, :, :][0]
                patch_img = Image.fromarray(patch)
                if scale != 1:
                    patch_img = patch_img.resize((patch_img.size[0] * scale, patch_img.size[1] * scale), Image.Resampling.BICUBIC)
                patch_img.save(dst_dir + '/{}_{}.png'.format(img_name.split('.')[0], i * patches.shape[1] + j))

def make_patches_ph(hr_dir: str, lr_dir: str, dst_dir: str, patch_size: int, stride: int, scale : int = 1, resize: bool = False):
    lr_images = sorted(os.listdir(lr_dir))
    hr_images = sorted(os.listdir(hr_dir))

    assert(len(lr_images) == len(hr_images))

    print(f'Creating patches from {hr_dir} and {lr_dir}...')

    all_hr, all_lr = [], []
    for k in tqdm(range(len(hr_images))):
        lr_path = os.path.join(lr_dir, lr_images[k])
        hr_path = os.path.join(hr_dir, hr_images[k])
        
        hr = np.array(Image.open(hr_path).convert('RGB'))
        lr = np.array(Image.open(lr_path).convert('RGB'))
        if resize:
            lr.resize(lr.shape[0] * scale, lr.shape[1] * scale, Image.Resampling.BICUBIC)

        hr_patches = patchify(hr, (patch_size, patch_size, 3), stride)
        lr_patches = patchify(lr, (
            patch_size // scale if not resize else patch_size,
            patch_size // scale if not resize else patch_size,
            3
        ), stride // scale if not resize else stride)

        for i in range(hr_patches.shape[0]):
            for j in range(hr_patches.shape[1]):
                hr_patch = hr_patches[i, j, :, :][0]
                lr_patch = lr_patches[i, j, :, :][0]
                all_hr.append(T.ToTensor()(Image.fromarray(hr_patch)))
                all_lr.append(T.ToTensor()(Image.fromarray(lr_patch)))
        
    data = {
        'inputs': torch.stack(all_lr),
        'targets': torch.stack(all_hr)
    }

    torch.save(data, f'{dst_dir}.pt')
    print(f'Patches saved in {dst_dir}.pt')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, required=True)
    parser.add_argument('--lr_src', type=str, required=False)
    parser.add_argument('--dst', type=str, required=True)
    parser.add_argument('--size', type=int, required=True)
    parser.add_argument('--stride', type=int, required=True)
    parser.add_argument('--ph', action=argparse.BooleanOptionalAction, type=bool, required=True)
    parser.add_argument('--scale', type=int, required=False)
    parser.add_argument('--resize', action=argparse.BooleanOptionalAction, type=bool, required=False)
    args = parser.parse_args()

    args.dst = args.dst + '_{}'.format(args.size * (args.scale if args.scale is not None else 1))

    if not os.path.exists(args.dst):
        os.makedirs(args.dst)

    if args.ph:
        assert(args.lr_src is not None)
        make_patches_ph(
            args.src, 
            args.lr_src,
            args.dst, 
            args.size, 
            args.stride, 
            args.scale if args.scale is not None else 1,
            args.resize if args.resize is not None else False
        )
    else:
        make_patches(
            args.src, 
            args.dst, 
            args.size, 
            args.stride, 
            args.scale if args.scale is not None else 1
        )
