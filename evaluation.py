import torch
import os
import pandas as pd
from torch.utils.data import DataLoader

from models.swinir import SwinIR
from training.train_swinir import SwinIRTraining

def evaluate_SwinIR_models(src: str, device: torch.device, data_loader: DataLoader, csv_dst: str) -> None:
    results = []
    models = sorted(os.listdir(src))

    for model_filename in models:
        model_path = os.path.join(src, model_filename)
        iterations = int(model_filename.split('_')[-1])

        model = SwinIR()
        model.load_state_dict(
            torch.load(model_path, weights_only=True, map_location=device)
        )
        model.to(device).eval()

        trainer = SwinIRTraining(model, device, None, data_loader)
        valid_loss, psnr_val, ssim_val = trainer.evaluate()

        results.append({
            'iterations': iterations,
            'valid_loss': float(valid_loss),
            'PSNR': psnr_val,
            'SSIM': ssim_val
        })

    df = pd.DataFrame(results)
    df.to_csv(csv_dst, sep=';', index=False, decimal=',')

if __name__ == '__main__':
    import argparse
    from dataset import DIV2KDataset, create_dataloader

    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, required=True)
    parser.add_argument('--valid_lr', type=str, required=True)
    parser.add_argument('--valid_hr', type=str, required=True)
    parser.add_argument('--batches', type=int, required=True)
    parser.add_argument('--dst', type=str, required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print('Loading dataset...')
    valid_dataset = DIV2KDataset(args.valid_hr, args.valid_lr)
    valid_loader = create_dataloader(valid_dataset, args.batches, shuffle=False)

    evaluate_SwinIR_models(args.src, device, valid_loader, args.dst)
