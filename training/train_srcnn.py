import torch
import torch.nn as nn
from torch.utils.data import DataLoader 

from train import NNTrainingBase

import sys, os
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_dir)
from models.srcnn import SRCNN
from dataset import create_dataloader

class SRCNNTraining(NNTrainingBase):
    def __init__(self, model: SRCNN, device: torch.device, train_loader: DataLoader, valid_loader: DataLoader, use_writer=False):
        optimizer = torch.optim.Adam([
            {'params': model.conv1.parameters(), 'lr': 1e-4},
            {'params': model.conv2.parameters(), 'lr': 1e-4},
            {'params': model.conv3.parameters(), 'lr': 1e-5},
        ])

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=.5)

        super().__init__(
            model, 
            device, 
            train_loader, 
            valid_loader, 
            nn.MSELoss(),
            optimizer,
            scheduler, 
            use_writer
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
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--batches', type=int, required=True)
    parser.add_argument('--scale', type=int, required=False)
    parser.add_argument('--tensorboard', action=argparse.BooleanOptionalAction, type=bool, required=True)
    args = parser.parse_args()

    model = SRCNN(args.scale) if args.scale is not None else SRCNN()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print('Loading datasets...')
    train_dataset = DIV2KDataset(args.train_hr, args.train_lr)
    valid_dataset = DIV2KDataset(args.valid_hr, args.valid_lr)

    batch_size = args.batches

    train_loader = create_dataloader(train_dataset, batch_size)
    valid_loader = create_dataloader(valid_dataset, batch_size, shuffle=False)

    print('Training...')
    trainer = SRCNNTraining(model, device, train_loader, valid_loader, use_writer=args.tensorboard)
    model = trainer.train_for_epochs(args.epochs)

    torch.save(model.state_dict(), f'{args.dst}_{args.epochs}')