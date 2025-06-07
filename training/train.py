import torch
import torch.nn as nn
from torch.utils.data import DataLoader 
from tqdm import tqdm
import math
from skimage.metrics import structural_similarity as ssim
from torch.utils.tensorboard import SummaryWriter

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

    def train_one_batch(self, inputs: torch.Tensor, targets: torch.Tensor):
        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)

        self.optimizer.zero_grad()

        outputs: torch.Tensor = self.model(inputs)

        loss: torch.Tensor = self.criterion(outputs, targets)
        loss.backward()

        self.optimizer.step()

        return loss.item()

    def evaluate_one_batch(self, inputs: torch.Tensor, targets: torch.Tensor):
        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)

        outputs: torch.Tensor = self.model(inputs)
        loss: torch.Tensor = self.criterion(outputs, targets)

        psnr_sum = 0.
        ssim_sum = 0.
        batch_size = inputs.size(0)

        for i in range(batch_size):
            output_img = outputs[i].permute(1, 2, 0).cpu().numpy()
            target_img = targets[i].permute(1, 2, 0).cpu().numpy()
            ssim_sum += ssim(target_img, output_img, data_range=1.0, channel_axis=-1)

            mse_i = torch.mean((outputs[i] - targets[i]) ** 2).item()
            psnr_sum += 10.0 * math.log10(1.0 / mse_i)

        return (
            loss * inputs.size(0),
            psnr_sum,
            ssim_sum,
            batch_size
        )

    def evaluate(self):
        self.model.eval()

        valid_loss = psnr_val = ssim_val = total_img = .0
        with torch.no_grad():
            for inputs, targets in tqdm(self.valid_loader, desc='Validation', leave=False):
                vl, pv, sv, n = self.evaluate_one_batch(inputs, targets)
                valid_loss += vl
                psnr_val += pv
                ssim_val += sv
                total_img += n

        return (
            valid_loss / total_img,
            psnr_val / total_img,
            ssim_val / total_img
        )

    def train_one_epoch(self, current_num: int, target_num: int):
        print('-' * 30)
        print(f'Epoch {current_num}/{target_num}')

        self.model.train()

        train_loss = .0
        num_examples = 0
        for inputs, targets in tqdm(self.train_loader, desc='Training', leave=False):
            loss_val = self.train_one_batch(inputs, targets)
            train_loss += loss_val * inputs.size(0)
            num_examples += inputs.size(0)
        train_loss /= num_examples

        valid_loss, psnr_val, ssim_val = self.evaluate()

        if self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(valid_loss)
            else:
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
        bar = None

        self.model = self.model.to(self.device)
        steps = 0
        running_loss = 0.0
        n_img = 0

        while steps < iterations:
            for inputs, targets in self.train_loader:
                if steps >= iterations:
                    bar = None
                    break

                if steps % val_interval == 0:
                    print('-' * 30)
                    print(f'Iterations {steps + 1}-{steps + val_interval}')
                    bar = tqdm(total=val_interval, desc='Training', leave=False)

                self.model.train()
                batch_size = inputs.size(0)
                batch_loss = self.train_one_batch(inputs, targets)
                running_loss += batch_loss * batch_size
                n_img += batch_size

                steps += 1
                bar.update()

                if steps % 10_000 == 0:
                    torch.save(self.model.state_dict(), f'tmp/checkpoint_{steps}')

                if steps % val_interval == 0:
                    bar = None

                    train_loss /= running_loss / n_img
                    valid_loss, psnr_val, ssim_val = self.evaluate()
                    
                    print(f'Train Loss: {train_loss:.6f} | Valid Loss: {valid_loss:.6f} | PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}')

                    if self.writer is not None:
                        self.write(steps, train_loss, valid_loss, psnr_val, ssim_val)

                    running_loss = 0.0
                    n_img = 0
                
                if self.scheduler is not None:
                    self.scheduler.step()

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
