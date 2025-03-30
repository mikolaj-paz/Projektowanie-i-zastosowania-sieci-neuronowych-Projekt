import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.swin_transformer import swin_v2_b

class SwinIR(nn.Module):
    def __init__(self):
        super().__init__()
        