# https://arxiv.org/pdf/1609.04802

import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights

class SRGANresidual(nn.Module):
    def __init__(self, in_channels=64):
        super(SRGANresidual, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.PReLU()
    
    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn(x)
        return x + residual
    
class SRGANupsampling(nn.Module):
    def __init__(self):
        super(SRGANupsampling, self).__init__()
        self.conv = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
        self.shuffle = nn.PixelShuffle(2)
        self.relu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.shuffle(x)
        return self.relu(x)

class SRGANgenerator(nn.Module):
    def __init__(self, num_residual=16, upscale_factor=2):
        super(SRGANgenerator, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )
        self.residual = nn.Sequential (
            *[SRGANresidual() for _ in range(num_residual)]
        )
        self.post_residual = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        self.upsampling = nn.Sequential(
            *[SRGANupsampling() for _ in range(int(upscale_factor / 2))]
        )
        self.final = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        x = self.initial(x)
        residual = x
        x = self.residual(x)
        x = self.post_residual(x)
        x += residual
        x = self.upsampling(x)
        return self.final(x)

class SRGANdiscriminator(nn.Module):
    def __init__(self):
        super(SRGANdiscriminator, self).__init__()
        
        class ConvBlock(nn.Module):
            def __init__(self, channels_in : int, channels_out : int, kernel_size=3, stride=1):
                super(ConvBlock, self).__init__()
                self.model = nn.Sequential(
                    nn.Conv2d(channels_in, channels_out, kernel_size, stride, padding=kernel_size // 2),
                    nn.BatchNorm2d(channels_out),
                    nn.LeakyReLU(.2, inplace=True)
                )

            def forward(self, x):
                return self.model(x)
        
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(.2, inplace=True)
        )
        self.conv = nn.Sequential(
            ConvBlock(64, 64, stride=2),
            ConvBlock(64, 128, stride=1),
            ConvBlock(128, 128, stride=2),
            ConvBlock(128, 256, stride=1),
            ConvBlock(256, 256, stride=2),
            ConvBlock(256, 512, stride=1),
            ConvBlock(512, 512, stride=2)
        )
        self.final = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.initial(x)
        x = self.conv(x)
        return self.final(x)
    
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(vgg.features)[:36]).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        return self.feature_extractor(x)