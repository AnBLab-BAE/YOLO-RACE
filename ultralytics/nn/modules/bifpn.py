# models/bifpn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = 'BiFPN'

class BiFPN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BiFPN, self).__init__()
        self.conv6_up = nn.Conv2d(in_channels, out_channels, 1)
        self.conv5_up = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv4_up = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv3_up = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.conv4_down = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv5_down = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv6_down = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv7_down = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x):
        p3, p4, p5, p6, p7 = x

        p6_up = F.interpolate(self.conv6_up(p7), size=p6.shape[-2:], mode='nearest') + p6
        p5_up = F.interpolate(self.conv5_up(p6_up), size=p5.shape[-2:], mode='nearest') + p5
        p4_up = F.interpolate(self.conv4_up(p5_up), size=p4.shape[-2:], mode='nearest') + p4
        p3_out = F.interpolate(self.conv3_up(p4_up), size=p3.shape[-2:], mode='nearest') + p3

        p4_down = F.interpolate(self.conv4_down(p3_out), size=p4.shape[-2:], mode='nearest') + p4
        p5_down = F.interpolate(self.conv5_down(p4_down), size=p5.shape[-2:], mode='nearest') + p5
        p6_down = F.interpolate(self.conv6_down(p5_down), size=p6.shape[-2:], mode='nearest') + p6
        p7_out = F.interpolate(self.conv7_down(p6_down), size=p7.shape[-2:], mode='nearest') + p7

        return p3_out, p4_down, p5_down, p6_down, p7_out
