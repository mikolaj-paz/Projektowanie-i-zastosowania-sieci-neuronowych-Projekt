import os
from PIL import Image
import numpy as np
from patchify import patchify
from tqdm import tqdm

def makePatches(src_dir : str, dst_dir : str, patch_size : int, stride : int, scale : int = 1):
    images = sorted(os.listdir(src_dir))

    print('Saving patches, created from', src_dir, 'to', dst_dir, '...')
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

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, required=True)
    parser.add_argument('--dst', type=str, required=True)
    parser.add_argument('--size', type=int, required=True)
    parser.add_argument('--stride', type=int, required=True)
    parser.add_argument('--scale', type=int, required=False)
    args = parser.parse_args()

    args.dst = args.dst + '_{}'.format(args.size * (args.scale if args.scale is not None else 1))

    if not os.path.exists(args.dst):
        os.makedirs(args.dst)

    if args.scale is not None:
        makePatches(args.src, args.dst, args.size, args.stride, args.scale)
    else:
        makePatches(args.src, args.dst, args.size, args.stride)
