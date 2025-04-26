import torch
import torch.nn as nn
import torchvision
from torchvision.models import VGG19_Weights
import torchvision.transforms as transforms
from torch.utils.data import DataLoader 
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import math
from skimage.metrics import structural_similarity as ssim
from itertools import cycle

import sys, os
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_dir)
from models.srgan import SRGANgenerator, SRGANdiscriminator

class SRGANTraining():
    def __init__(self, generator: SRGANgenerator, discriminator: SRGANdiscriminator, device: torch.device, train_loader: DataLoader, valid_loader: DataLoader, use_writer=False):
        self.generator = generator
        self.discriminator = discriminator
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.writer = SummaryWriter() if use_writer else None

        self.adversarial_criterion = torch.nn.BCEWithLogitsLoss()
        self.content_criterion = torch.nn.MSELoss()

        self.optimizer_generator = torch.optim.Adam(generator.parameters(), lr=1e-4)
        self.optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

        self.scheduler_generator = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_generator)

        self.hr_transform = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        ])

        self.vgg: torchvision.models.VGG = torchvision.models.vgg19(weights=VGG19_Weights.DEFAULT).features[:35].eval().to(device)
        for param in self.vgg.parameters():
            param.requires_grad = False

    def train_generator_one_batch(self, inputs: torch.Tensor, outputs: torch.Tensor, targets: torch.Tensor, real_labels, train_loss = .0):
        self.optimizer_generator.zero_grad()

        # Content loss
        with torch.no_grad():
            gen_features = self.vgg(outputs.detach())
            real_features = self.vgg(targets.detach())
        content_loss: torch.Tensor = .006 * self.content_criterion(gen_features, real_features)

        # Adversarial loss
        fake_outputs = self.discriminator(outputs)
        adversarial_loss: torch.Tensor = self.adversarial_criterion(fake_outputs, real_labels)

        # Total loss
        total_loss = content_loss + 1e-3 * adversarial_loss
        total_loss.backward()

        self.optimizer_generator.step()

        train_loss += total_loss.item() * inputs.size(0)
        
        return train_loss
    
    def train_discriminator_one_batch(self, inputs: torch.Tensor, outputs: torch.Tensor, targets: torch.Tensor, real_labels, fake_labels, train_loss = .0):
        self.optimizer_discriminator.zero_grad()

        pred_real = self.discriminator(targets)
        pred_fake = self.discriminator(outputs.detach())

        loss_real = self.adversarial_criterion(pred_real, real_labels)
        loss_fake = self.adversarial_criterion(pred_fake, fake_labels)

        total_loss = loss_real + loss_fake
        total_loss.backward()

        self.optimizer_discriminator.step()

        train_loss += total_loss.item() * inputs.size(0)

        return train_loss

    def train_one_batch(self, inputs: torch.Tensor, targets: torch.Tensor, generator_train_loss = .0, discriminator_train_loss = .0):
        inputs = inputs.to(self.device, non_blocking=True)
        targets = self.hr_transform(targets).to(self.device, non_blocking=True)

        real_labels = torch.ones((inputs.size(0), 1), dtype=torch.float32, device=self.device)
        fake_labels = torch.zeros((inputs.size(0), 1), dtype=torch.float32, device=self.device)

        self.optimizer_generator.zero_grad()
        outputs: torch.Tensor = self.generator(inputs)
        generator_train_loss = self.train_generator_one_batch(inputs, outputs, targets, real_labels, generator_train_loss)

        outputs = outputs.detach()
        discriminator_train_loss = self.train_discriminator_one_batch(inputs, outputs, targets, real_labels, fake_labels, discriminator_train_loss)

        return generator_train_loss, discriminator_train_loss

    def evaluate_one_batch(self, inputs: torch.Tensor, targets: torch.Tensor, valid_loss = .0, vgg_loss = .0, psnr_val = .0, ssim_val = .0):
        inputs = inputs.to(self.device, non_blocking=True)
        targets = self.hr_transform(targets).to(self.device, non_blocking=True)

        outputs: torch.Tensor = self.generator(inputs)

        loss: torch.Tensor = self.content_criterion(outputs, targets)

        valid_loss += loss.item() * inputs.size(0)

        batch_ssim = .0
        for i in range(inputs.size(0)):
            output_img = (outputs[i].squeeze(0).cpu().numpy() * .5) + .5
            target_img = (targets[i].squeeze(0).cpu().numpy() * .5) + .5
            batch_ssim += ssim(target_img, output_img, data_range=1.0, channel_axis=0)
        ssim_val += batch_ssim / inputs.size(0)

        psnr_val += 10. * math.log10(1.0 / loss.item())

        with torch.no_grad():
            outputs_features = self.vgg(outputs.detach())
            targets_features = self.vgg(targets.detach())
        content_loss: torch.Tensor = .006 * self.content_criterion(outputs_features, targets_features)
        vgg_loss += content_loss.item()

        return valid_loss, vgg_loss, psnr_val, ssim_val

    def evaluate(self):
        self.generator.eval()
        
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

    def train_one_epoch(self, current_num: int, target_num: int):
        print('-' * 30)
        print(f'Epoch {current_num}/{target_num}')

        self.generator.train()
        self.discriminator.train()
        
        generator_train_loss = .0
        discriminator_train_loss = .0
        for inputs, targets in tqdm(self.train_loader, desc='Training', leave=False):
            generator_train_loss, discriminator_train_loss = self.train_one_batch(inputs, targets, generator_train_loss, discriminator_train_loss)
        generator_train_loss /= len(self.train_loader)
        discriminator_train_loss /= len(self.train_loader)

        valid_loss, vgg_loss, psnr_val, ssim_val = self.evaluate()

        print(f'Generator Train Loss: {generator_train_loss:.6f} | Discriminator Train Loss: {discriminator_train_loss:.6f}')
        print(f'Valid Loss: {valid_loss:.6f} | PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}')

        self.scheduler_generator.step(vgg_loss)

        return generator_train_loss, valid_loss, psnr_val, ssim_val

    def train_for_epochs(self, epochs: int):
        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)

        for epoch in range(1, epochs + 1):
            train_loss, valid_loss, psnr_val, ssim_val = self.train_one_epoch(epoch, epochs)

            if self.writer is not None:
                self.write(epoch, train_loss, valid_loss, psnr_val, ssim_val)

        return self.generator, self.discriminator
    
    def train_for_iterations(self, iterations: int, val_interval: int):
        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)

        steps = 0

        generator_train_loss = .0
        discriminator_train_loss = .0

        while steps < iterations:
            for inputs, targets, in self.train_loader:
                if steps >= iterations:
                    break

                self.generator.train()
                self.discriminator.train()
                
                generator_train_loss, discriminator_train_loss = self.train_one_batch(inputs, targets, generator_train_loss, discriminator_train_loss)

                steps += 1

                if steps % val_interval == 0:
                    valid_loss, vgg_loss, psnr_val, ssim_val = self.evaluate()
                    self.scheduler_generator.step(vgg_loss)

                    generator_train_loss /= val_interval
                    discriminator_train_loss /= val_interval
                    print(f'Generator Train Loss: {generator_train_loss:.6f} | Discriminator Train Loss: {discriminator_train_loss:.6f}')
                    print(f'Valid Loss: {valid_loss:.6f} | VGG Loss: {vgg_loss:.6f} | PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}')
                    
                    if self.writer is not None:
                        self.write(steps, generator_train_loss, valid_loss, psnr_val, ssim_val)

                    generator_train_loss = .0
                    discriminator_train_loss = .0

        return self.generator, self.discriminator

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