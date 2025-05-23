# https://www.digitalocean.com/community/tutorials/super-resolution-generative-adversarial-networks

import torch
import torch.nn as nn
from torch.utils.data import DataLoader 
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import math
from skimage.metrics import structural_similarity as ssim

import sys, os
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_dir)
from models.srgan import SRGANgenerator, SRGANdiscriminator, FeatureExtractor

class SRGANTraining():
    def __init__(self, generator: SRGANgenerator, discriminator: SRGANdiscriminator, device: torch.device, train_loader: DataLoader, valid_loader: DataLoader, use_writer=False):
        self.generator = generator
        self.discriminator = discriminator
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.writer = SummaryWriter() if use_writer else None

        self.adversarial_criterion = nn.BCELoss()
        self.content_criterion = nn.MSELoss()

        self.optimizer_generator = torch.optim.Adam(generator.parameters(), lr=1e-4)
        self.optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

        self.scheduler_generator = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_generator)

        self.feature_extractor = FeatureExtractor()

    def calculate_perceptual_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs_features = self.feature_extractor(outputs)
            targets_features = self.feature_extractor(targets)
        return self.content_criterion(outputs_features, targets_features)
    
    def calculate_adversarial_loss(self, outputs: torch.Tensor) -> torch.Tensor:
        return self.adversarial_criterion(outputs, torch.ones_like(outputs))
    
    def calculate_discriminator_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_loss = self.adversarial_criterion(targets, torch.ones_like(targets))
        outputs_loss = self.adversarial_criterion(outputs, torch.zeros_like(outputs))
        return (targets_loss + outputs_loss) / 2.

    def train_one_batch(self, inputs: torch.Tensor, targets: torch.Tensor, generator_train_loss = .0, discriminator_train_loss = .0):
        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)

        outputs: torch.Tensor = self.generator(inputs)

        self.optimizer_generator.zero_grad()
        self.optimizer_discriminator.zero_grad()

        adversarial_loss = self.calculate_adversarial_loss(self.discriminator(outputs))
        perceptual_loss = self.calculate_perceptual_loss(outputs, targets)
        content_loss: torch.Tensor = self.content_criterion(outputs, targets)
        generator_loss = .006 * perceptual_loss + .001 * adversarial_loss + content_loss

        generator_loss.backward()
        self.optimizer_generator.step()

        discriminator_loss: torch.Tensor = self.calculate_discriminator_loss(
            self.discriminator(outputs.detach()),
            self.discriminator(targets)
        )

        discriminator_loss.backward()
        self.optimizer_discriminator.step()

        generator_train_loss += generator_loss.item() * inputs.size(0)
        discriminator_train_loss += discriminator_loss.item() * inputs.size(0)

        return generator_train_loss, discriminator_train_loss

    def evaluate_one_batch(self, inputs: torch.Tensor, targets: torch.Tensor, valid_loss = .0, vgg_loss = .0, psnr_val = .0, ssim_val = .0):
        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)

        outputs: torch.Tensor = self.generator(inputs)

        loss: torch.Tensor = self.content_criterion(outputs, targets)

        valid_loss += loss.item() * inputs.size(0)

        batch_ssim = .0
        for i in range(inputs.size(0)):
            output_img = (outputs[i].squeeze(0).cpu().numpy())
            target_img = (targets[i].squeeze(0).cpu().numpy())
            batch_ssim += ssim(target_img, output_img, data_range=1.0, channel_axis=0)
        ssim_val += batch_ssim / inputs.size(0)

        psnr_val += 10. * math.log10(1.0 / loss.item())

        vgg_loss += self.calculate_perceptual_loss(outputs, targets).item() * inputs.size(0)

        return valid_loss, vgg_loss, psnr_val, ssim_val

    def evaluate(self):
        self.generator.eval()
        self.discriminator.eval()
        
        valid_loss = .0
        vgg_loss = .0
        psnr_val = .0
        ssim_val = .0
        with torch.no_grad():
            for inputs, targets in tqdm(self.valid_loader, desc='Validation', leave=False):
                valid_loss, vgg_loss, psnr_val, ssim_val = self.evaluate_one_batch(inputs, targets, valid_loss, vgg_loss, psnr_val, ssim_val)
        loader_len = len(self.valid_loader)
        valid_loss /= loader_len
        vgg_loss /= loader_len
        psnr_val /= loader_len
        ssim_val /= loader_len

        return valid_loss, vgg_loss, psnr_val, ssim_val
    
    def train_for_iterations(self, iterations: int, val_interval: int):
        bar = None

        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)
        self.feature_extractor = self.feature_extractor.to(self.device)

        steps = 0

        generator_train_loss = .0
        discriminator_train_loss = .0

        while steps < iterations:
            for inputs, targets in self.train_loader:
                if steps >= iterations:
                    bar = None
                    break

                if steps % val_interval == 0:
                    print('-' * 30)
                    print(f'Iterations {steps + 1}-{steps + val_interval}')
                    bar = tqdm(total=val_interval, desc='Training', leave=False)

                self.generator.train()
                self.discriminator.train()
                
                generator_train_loss, discriminator_train_loss = self.train_one_batch(inputs, targets, generator_train_loss, discriminator_train_loss)

                steps += 1
                bar.update()

                if steps % val_interval == 0:
                    bar = None

                    train_loss /= val_interval
                    valid_loss, vgg_loss, psnr_val, ssim_val = self.evaluate()
                    self.scheduler_generator.step(vgg_loss)

                    print(f'Generator Train Loss: {(generator_train_loss / val_interval):.6f} | Discriminator Train Loss: {(discriminator_train_loss / val_interval):.6f}')
                    print(f'Valid Loss: {valid_loss:.6f} | VGG Loss: {vgg_loss:.6f} | PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}')
                    
                    if self.writer is not None:
                        self.write(steps, generator_train_loss, valid_loss, psnr_val, ssim_val)

                    generator_train_loss = .0
                    discriminator_train_loss = .0

        return self.generator, self.discriminator
    
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

if __name__ == '__main__':
    import argparse
    from dataset import DIV2KDataset, create_dataloader

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_lr', type=str, required=True)
    parser.add_argument('--train_hr', type=str, required=True)
    parser.add_argument('--valid_lr', type=str, required=True)
    parser.add_argument('--valid_hr', type=str, required=True)
    parser.add_argument('--dst', type=str, required=True)
    parser.add_argument('--iterations', type=int, required=True)
    parser.add_argument('--interval', type=int, required=True)
    parser.add_argument('--batches', type=int, required=True)
    parser.add_argument('--val_batches', type=int, required=True)
    parser.add_argument('--scale', type=int, required=False)
    parser.add_argument('--tensorboard', action=argparse.BooleanOptionalAction, type=bool, required=True)
    args = parser.parse_args()

    generator = SRGANgenerator(upscale_factor=args.scale) if args.scale is not None else SRGANgenerator()
    discriminator = SRGANdiscriminator()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print('Loading datasets...')
    train_dataset = DIV2KDataset(args.train_hr, args.train_lr)
    valid_dataset = DIV2KDataset(args.valid_hr, args.valid_lr)

    batch_size = args.batches
    val_batch_size = args.val_batches

    train_loader = create_dataloader(train_dataset, batch_size)
    valid_loader = create_dataloader(valid_dataset, val_batch_size, shuffle=False)

    print('Training...')
    trainer = SRGANTraining(generator, discriminator, device, train_loader, valid_loader, use_writer=args.tensorboard)
    model, _ = trainer.train_for_iterations(args.iterations, args.interval)

    torch.save(model.state_dict(), f'{args.dst}_{args.iterations}iter')