import torch
import torch.nn as nn
from torch.utils.data import DataLoader 
from tqdm import tqdm

import sys, os
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_dir)
from models.srcnn import SRCNN

def train_srcnn(
    model: nn.Module,
    epochs: int,
    device: torch.device,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler
):
    model = model.to(device)
    for epoch in range(1, epochs + 1):
        print("-" * 30)
        print(f'Epoch {epoch}/{epochs}')

        model.train()
        train_loss = .0

        for inputs, targets in tqdm(train_loader, desc='Training', leave=False):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
        
        train_loss /= len(train_loader.dataset)

        model.eval()
        valid_loss = .0

        with torch.no_grad():
            for inputs, targets in tqdm(valid_loader, desc='Validation', leave=False):
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                outputs = model(inputs)

                loss = criterion(outputs, targets)

                valid_loss += loss.item() * inputs.size(0)

        valid_loss /= len(valid_loader.dataset)
        scheduler.step(valid_loss)

        print(f"Train Loss: {train_loss:.6f} | Valid Loss: {valid_loss:.6f}")

        return model


if __name__ == "__main__":
    import argparse
    from dataset import DIV2KDataset

    parser = argparse.ArgumentParser()
    parser.add_argument('--d', '--dst', type=str, required=True)
    parser.add_argument('--e', '--epochs', type=int, required=True)
    parser.add_argument('--b', '--batches', type=int, required=False)
    parser.add_argument('--s', '--scale', type=int, required=False)
    args = parser.parse_args()

    model = SRCNN(args.s) if args.s is not None else SRCNN()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f'Device: {device}')

    print('Loading datasets...')
    train_dataset = DIV2KDataset("patches/DIV2K_train_HR_480", "patches/DIV2K_train_LR_480")
    valid_dataset = DIV2KDataset("patches/DIV2K_valid_HR_480", "patches/DIV2K_valid_LR_480")

    batch_size = args.b if args.b is not None else 8

    train_loader = DataLoader(
        train_dataset,
        batch_size,
        shuffle=True, 
        num_workers=os.cpu_count() // 2, 
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size,
        shuffle=True, 
        num_workers=os.cpu_count() // 2, 
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=.001)

    print('Training...')
    # model = train_srcnn(
    #     model, 
    #     args.e, 
    #     device, 
    #     train_loader, 
    #     valid_loader,
    #     nn.MSELoss(),
    #     optimizer,
    #     torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=.5, threshold=.03, min_lr=1e-5)
    # )

    torch.save(model.state_dict(), args.d)