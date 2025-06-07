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

def set_requires_grad(module: nn.Module, flag: bool):
    for param in module.parameters():
        param.requires_grad = flag

class SRGANTraining():
    def __init__(
        self,
        generator: SRGANgenerator,
        discriminator: SRGANdiscriminator,
        device: torch.device,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        use_writer = False
    ):
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

    def __perceptual_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        outputs_features = self.feature_extractor(outputs)
        with torch.no_grad():
            targets_features = self.feature_extractor(targets)
        return self.content_criterion(outputs_features, targets_features)
    
    def __adversarial_loss(self, outputs: torch.Tensor) -> torch.Tensor:
        return self.adversarial_criterion(outputs, torch.ones_like(outputs))
    
    def __discriminator_loss(self, outputs_fake: torch.Tensor, outputs_real: torch.Tensor) -> torch.Tensor:
        real_loss = self.adversarial_criterion(outputs_real, torch.ones_like(outputs_real))
        fake_loss = self.adversarial_criterion(outputs_fake, torch.zeros_like(outputs_fake))
        return 0.5 * (real_loss + fake_loss)

    def train_one_batch(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[float, float]:
        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)

        set_requires_grad(self.discriminator, False)
        self.optimizer_generator.zero_grad(set_to_none=True)

        fakes: torch.Tensor = self.generator(inputs)

        generator_adversarial_loss = self.__adversarial_loss(self.discriminator(fakes))
        generator_content_loss: torch.Tensor = self.content_criterion(fakes, targets)
        generator_perceptual_loss = self.__perceptual_loss(fakes, targets)

        generator_total_loss = generator_content_loss + .006 * generator_perceptual_loss + .001 * generator_adversarial_loss
        generator_total_loss.backward()
        self.optimizer_generator.step()

        set_requires_grad(self.discriminator, True)
        self.optimizer_discriminator.zero_grad(set_to_none=True)

        discriminator_fakes = self.discriminator(fakes.detach())
        discriminator_real = self.discriminator(targets)
        discriminator_total_loss = self.__discriminator_loss(discriminator_fakes, discriminator_real)
        discriminator_total_loss.backward()
        self.optimizer_discriminator.step()

        return generator_total_loss.item(), discriminator_total_loss.item()

    def evaluate_one_batch(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[float, float, float, float, int]:
        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)

        with torch.no_grad():
            outputs: torch.Tensor = self.generator(inputs)
            batch_size = inputs.size(0)

            content_loss = self.content_criterion(outputs, targets).item() * batch_size
            vgg_loss = self.__perceptual_loss(outputs, targets).item() * batch_size

            psnr_sum = ssim_sum = .0
            for i in range(batch_size):
                output_img = outputs[i].permute(1, 2, 0).cpu().numpy()
                target_img = targets[i].permute(1, 2, 0).cpu().numpy()

                ssim_sum += ssim(target_img, output_img, data_range=1.0, channel_axis=-1)

                mse_i = torch.mean((outputs[i] - targets[i]) ** 2).item()
                psnr_sum += 10.0 * math.log10(1.0 / mse_i)

        return content_loss, vgg_loss, psnr_sum, ssim_sum, batch_size

    def evaluate(self):
        self.generator.eval()
        self.discriminator.eval()
        
        content_loss = vgg_loss = psnr_val = ssim_val = total_img = 0.0
        with torch.no_grad():
            for inputs, targets in tqdm(self.valid_loader, desc='Validation', leave=False):
                cl, vl, pv, sv, n = self.evaluate_one_batch(inputs, targets)
                content_loss += cl
                vgg_loss += vl
                psnr_val += pv
                ssim_val += sv
                total_img += n

        return (
            content_loss / total_img,
            vgg_loss / total_img,
            psnr_val / total_img,
            ssim_val / total_img
        )
    
    def train_for_iterations(self, iterations: int, val_interval: int):
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        self.feature_extractor.to(self.device)

        bar = None
        steps = 0
        img_since_val = 0
        generator_loss_sum = .0
        discriminator_loss_sum = .0

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
                
                generator_loss, discriminator_loss = self.train_one_batch(inputs, targets)
                batch_size = inputs.size(0)
                generator_loss_sum += generator_loss * batch_size
                discriminator_loss_sum += discriminator_loss * batch_size
                img_since_val += batch_size

                steps += 1
                bar.update()

                if steps % 10_000 == 0:
                    torch.save(self.generator.state_dict(), f'tmp/checkpoint_{steps}')

                if steps % val_interval == 0:
                    bar = None

                    generator_loss_average = generator_loss_sum / img_since_val
                    discriminator_loss_average = discriminator_loss_sum / img_since_val

                    valid_loss, vgg_loss, psnr_val, ssim_val = self.evaluate()

                    self.scheduler_generator.step(vgg_loss)

                    print(
                        f'Generator Train Loss: {generator_loss_average:.6f} | Discriminator Train Loss: {discriminator_loss_average:.6f}'
                    )
                    print(
                        f'Valid Loss: {valid_loss:.6f} | VGG Loss: {vgg_loss:.6f} | PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}'
                    )
                    
                    if self.writer is not None:
                        self.write(steps, generator_loss_average, valid_loss, psnr_val, ssim_val)

                    generator_loss_sum = discriminator_loss_sum = 0.0
                    img_since_val = 0

        return self.generator, self.discriminator
    
    def write(self, step: int, train_loss: float, valid_loss: float, psnr_val: float, ssim_val: float):
        assert self.writer is not None
        self.writer.add_scalars(
            'Training vs. Validation Loss',
            { 'Training': train_loss, 'Validation': valid_loss },
            step
        )
        self.writer.add_scalar('PSNR', psnr_val, step)
        self.writer.add_scalar('SSIM', ssim_val, step)
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

    train_loader = create_dataloader(train_dataset, args.batches)
    valid_loader = create_dataloader(valid_dataset, args.val_batches, shuffle=False)

    print('Training...')
    trainer = SRGANTraining(
        generator, discriminator, device, train_loader, valid_loader, use_writer=args.tensorboard
    )
    model, _ = trainer.train_for_iterations(args.iterations, args.interval)

    torch.save(model.state_dict(), f'{args.dst}_{args.iterations}iter')