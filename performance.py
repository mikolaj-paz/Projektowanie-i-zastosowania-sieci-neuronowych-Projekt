import torch
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.transforms import ToPILImage

from models.srcnn import SRCNN
from models.srgan import SRGANgenerator
from models.swinir import SwinIR

from visualize import model_from_file, compute_metrics

to_pillow = ToPILImage()

def evaluate_performance(
    model: torch.nn.Module,
    device: torch.device,
    lr: torch.Tensor,
    hr: torch.Tensor
) -> tuple[float, float, float]:
    
    lr = lr.to(device, non_blocking=True)
    hr = hr.to(device, non_blocking=True)
    
    if type(model) is SRCNN:
        lr = torch.nn.functional.interpolate(
            lr, scale_factor=2, mode='bicubic', align_corners=False
        )
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        output: torch.Tensor = model(lr)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    end = time.perf_counter()

    psnr, ssim = compute_metrics(
        to_pillow(output.squeeze(0).cpu()),
        to_pillow(hr.squeeze(0).cpu())
    )

    del output, lr, hr
    torch.cuda.empty_cache()

    return [
        end - start,
        float(psnr),
        float(ssim)
    ]

def calculate_average_performance_of_model(
    model: torch.nn.Module,
    device: torch.device,
    data_loader: DataLoader
) -> tuple[float, float, float]:
    
    n = 0
    t_sum = psnr_sum = ssim_sum = 0.

    for inputs, targets in tqdm(data_loader, desc='Processing', leave=False):
        t, p, s = evaluate_performance(model, device, inputs, targets)
        t_sum += t
        psnr_sum += p
        ssim_sum += s
        n += 1
    
    return [
        t_sum / n,
        psnr_sum / n,
        ssim_sum / n
    ]

if __name__ == '__main__':
    import argparse
    from dataset import DIV2KDataset, create_dataloader

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=str, required=True)
    parser.add_argument('--hr', type=str, required=True)
    parser.add_argument('--srcnn', type=str, required=True)
    parser.add_argument('--srgan', type=str, required=True)
    parser.add_argument('--swinir', type=str, required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print('Loading dataset...')
    dataset = DIV2KDataset(args.hr, args.lr)

    MODELS = [
        ('SRCNN', SRCNN, args.srcnn),
        ('SRGAN', SRGANgenerator, args.srgan),
        ('SwinIR', SwinIR, args.swinir)
    ]

    for name, cls, weight_path in MODELS:
        print(f'\n===> Evaluating {name} ...')
        model = cls()
        model_from_file(model, device, weight_path)
        model.to(device).eval()

        data_loader = create_dataloader(dataset, 1, shuffle=False, pin_memory=False)

        t_avg, psnr_avg, ssim_avg = calculate_average_performance_of_model(
            model, device, data_loader
        )

        print(f'- processing = {t_avg:.6f}'
              f'\n - PSNR = {psnr_avg:.6f} dB'
              f'\n - SSIM = {ssim_avg:.6f}')
        
        del model, data_loader
        torch.cuda.empty_cache()
        