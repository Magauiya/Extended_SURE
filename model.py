import torch.nn.functional as F

from utils import *

"""
Link: https://github.com/milesial/Pytorch-UNet
"""

class UNet(nn.Module):
    def __init__(self, cfg):
        super(UNet, self).__init__()

        self.inc = DoubleConv(cfg.channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if cfg.bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, cfg.bilinear)
        self.up2 = Up(512, 256 // factor, cfg.bilinear)
        self.up3 = Up(256, 128 // factor, cfg.bilinear)
        self.up4 = Up(128, 64, cfg.bilinear)
        self.conv = nn.Sequential(nn.ReLU(), OutConv(64, cfg.channels))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        res = self.up1(x5, x4)
        res = self.up2(res, x3)
        res = self.up3(res, x2)
        res = self.up4(res, x1)
        res = self.conv(res) + x # long skip connection

        return res