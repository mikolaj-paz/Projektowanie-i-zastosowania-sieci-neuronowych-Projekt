import torch
import torch.nn as nn
from torch.utils.data import DataLoader 
from tqdm import tqdm
import math
from skimage.metrics import structural_similarity as ssim
from torch.utils.tensorboard import SummaryWriter
from itertools import cycle

class NNTrainingBase:
    def __init__(
            self,
            model: nn.Module,
            device: torch.device,
            train_loader: DataLoader,
            valid_loader: DataLoader,
            criterion: nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler._LRScheduler = None,
            use_writer = False
        ):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.writer = SummaryWriter() if use_writer else None

    def train_one_batch(self, inputs: torch.Tensor, targets: torch.Tensor, train_loss = .0):
        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)

        self.optimizer.zero_grad()

        outputs: torch.Tensor = self.model(inputs)

        loss: torch.Tensor = self.criterion(outputs, targets)
        loss.backward()

        self.optimizer.step()

        train_loss += loss.item() * inputs.size(0)

        return train_loss

    def evaluate_one_batch(self, inputs: torch.Tensor, targets: torch.Tensor, valid_loss = .0, psnr_val = .0, ssim_val = .0):
        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)

        outputs: torch.Tensor = self.model(inputs)

        loss: torch.Tensor = self.criterion(outputs, targets)
        
        valid_loss += loss.item() * inputs.size(0)

        batch_ssim = .0
        for i in range(inputs.size(0)):
            output_img = outputs[i].squeeze(0).cpu().numpy()
            target_img = targets[i].squeeze(0).cpu().numpy()
            batch_ssim += ssim(target_img, output_img, data_range=1.0, channel_axis=0)
        ssim_val += batch_ssim / inputs.size(0)
            
        psnr_val += 10. * math.log10(1.0 / loss.item())

        return valid_loss, psnr_val, ssim_val

    def evaluate(self):
        self.model.eval()

        valid_loss = .0
        psnr_val = .0
        ssim_val = .0
        with torch.no_grad():
            for inputs, targets in tqdm(self.valid_loader, desc='Validation', leave=False):
                valid_loss, psnr_val, ssim_val = self.evaluate_one_batch(inputs, targets, valid_loss, psnr_val, ssim_val)
        loader_len = len(self.valid_loader)
        valid_loss /= loader_len
        psnr_val /= loader_len
        ssim_val /= loader_len

        return valid_loss, psnr_val, ssim_val

    def train_one_epoch(self, current_num: int, target_num: int):
        print('-' * 30)
        print(f'Epoch {current_num}/{target_num}')

        self.model.train()

        train_loss = .0
        for inputs, targets in tqdm(self.train_loader, desc='Training', leave=False):
            train_loss = self.train_one_batch(inputs, targets, train_loss)
        train_loss /= len(self.train_loader)

        valid_loss, psnr_val, ssim_val = self.evaluate()

        if self.scheduler is not None:
            self.scheduler.step()

        print(f'Train Loss: {train_loss:.6f} | Valid Loss: {valid_loss:.6f} | PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}')

        return train_loss, valid_loss, psnr_val, ssim_val

    def train_for_epochs(self, epochs: int):
        self.model = self.model.to(self.device)

        for epoch in range(1, epochs + 1):
            train_loss, valid_loss, psnr_val, ssim_val = self.train_one_epoch(epoch, epochs)

            if self.writer is not None:
                self.write(epoch, train_loss, valid_loss, psnr_val, ssim_val)

        return self.model
    
    def train_for_iterations(self, iterations: int, val_interval: int):
        self.model = self.model.to(self.device)
        train_iterator = cycle(self.train_loader)

        for step in range(iterations // val_interval):
            print('-' * 30)
            print(f'Iterations {step * val_interval + 1}-{(step + 1) * val_interval}')

            self.model.train()

            train_loss = .0
            for i in tqdm(range(val_interval), desc='Training', leave=False):
                inputs, targets = next(train_iterator)
                train_loss = self.train_one_batch(inputs, targets, train_loss)
                if self.scheduler is not None:
                    self.scheduler.step()
            train_loss /= val_interval

            valid_loss, psnr_val, ssim_val = self.evaluate()

            print(f'Train Loss: {train_loss:.6f} | Valid Loss: {valid_loss:.6f} | PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}')

            if self.writer is not None:
                self.write(step * val_interval + 1, train_loss, valid_loss, psnr_val, ssim_val)

        return self.model
                
    def write(self, arg: int, train_loss: float, valid_loss: float, psnr_val: float, ssim_val: float):
        assert(self.writer is not None)
        self.writer.add_scalars(
            'Training vs. Validation Loss',
            { 'Training': train_loss, 'Validation': valid_loss },
            arg
        )
        self.writer.add_scalar(
            'PSNR',
            psnr_val,
            arg
        )
        self.writer.add_scalar(
            'SSIM',
            ssim_val,
            arg
        )
        self.writer.flush()
