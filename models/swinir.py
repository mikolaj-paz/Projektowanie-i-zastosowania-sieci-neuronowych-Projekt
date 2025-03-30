import torch.nn as nn
from torchvision.models.swin_transformer import SwinTransformerBlock

class RSTB(nn.Module):
    def __init__(self, dim=180, num_heads=6, window_size=8, num_blocks=6):
        super(RSTB, self).__init__()
        self.stls = nn.Sequential(
            *[
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=(window_size, window_size),
                    shift_size=[0, 0] if i % 2 == 0 else [window_size // 2, window_size // 2]
                ) 
                for i in range(num_blocks)
            ]
        )
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.stls(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv(x)
        return x + residual

class SwinIR(nn.Module):

    def __init__(self, upscale_factor=2, dim=180, num_blocks=6):
        super(SwinIR, self).__init__()
        self.shallow = nn.Conv2d(3, dim, kernel_size=3, padding=1)
        self.deep = nn.Sequential(*[RSTB(dim) for _ in range(num_blocks)])
        self.reconstruction = nn.Sequential(
            nn.Conv2d(dim, dim * (upscale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.Conv2d(dim, 3, kernel_size=3, padding=1)
        )
            
    def forward(self, x):
        x = self.shallow(x)
        residual = x
        x = self.deep(x)
        x += residual
        x = self.reconstruction(x)
        return x
        