# https://arxiv.org/abs/1501.00092
# https://debuggercafe.com/image-super-resolution-using-srcnn-and-pytorch/
# https://debuggercafe.com/image-super-resolution-using-deep-learning-and-pytorch/

import torch.nn as nn
import torch.nn.functional as F

class SRCNN(nn.Module):
    def __init__(self, upscale_factor=2):
        super(SRCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

        for layer in [self.conv1, self.conv2, self.conv3]:
            nn.init.normal_(self.conv3.weight, mean=.0, std=.001)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x