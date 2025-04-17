import torch
import torch.nn as nn
from torch.utils.data import DataLoader 
from tqdm import tqdm
import math

from train import NNTrainingBase

import sys, os
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_dir)
from models.swinir import SwinIR
from dataset import create_dataloader

class SwinIRTraining(NNTrainingBase):
    def __init__(self, model: SwinIR, device: torch.device, train_loader: DataLoader, valid_loader: DataLoader, use_writer=False):
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(.9, .99))

        super().__init__(
            model, 
            device, 
            train_loader, 
            valid_loader, 
            nn.L1Loss(),
            optimizer,
            scheduler=None,
            use_writer=use_writer
        )

    def update_MultiStepLR(self, milestones: list[int]):
        print(f'New MultiStepLR milestones: {milestones}')
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones,
            gamma=.5
        )

if __name__ == '__main__':
    import argparse
    from dataset import DIV2KDataset

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_lr', type=str, required=True)
    parser.add_argument('--train_hr', type=str, required=True)
    parser.add_argument('--valid_lr', type=str, required=True)
    parser.add_argument('--valid_hr', type=str, required=True)
    parser.add_argument('--dst', type=str, required=True)
    parser.add_argument('--iterations', type=int, required=True)
    parser.add_argument('--interval', type=int, required=True)
    parser.add_argument('--batches', type=int, required=True)
    parser.add_argument('--scale', type=int, required=False)
    parser.add_argument('--tensorboard', action=argparse.BooleanOptionalAction, type=bool, required=True)
    args = parser.parse_args()

    model = SwinIR(args.scale) if args.scale is not None else SwinIR()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print('Loading datasets...')
    train_dataset = DIV2KDataset(args.train_hr, args.train_lr)
    valid_dataset = DIV2KDataset(args.valid_hr, args.valid_lr)

    batch_size = args.batches

    train_loader = create_dataloader(train_dataset, batch_size)
    valid_loader = create_dataloader(valid_dataset, batch_size, shuffle=False)

    original_milestones = [
        250_000,
        400_000,
        450_000,
        475_000
    ]
    milestones = [int(m * args.iterations / 500_000) for m in original_milestones]

    trainer = SwinIRTraining(model, device, train_loader, valid_loader, use_writer=args.tensorboard)
    trainer.update_MultiStepLR(milestones)

    print('Training...')
    model = trainer.train_for_iterations(args.iterations, args.interval)

    torch.save(model.state_dict(), f'{args.dst}_{args.iterations}iter')