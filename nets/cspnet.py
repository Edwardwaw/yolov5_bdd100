import torch
from torch import nn
import math
from nets.commons import Focus,BottleneckCSP,Conv,SPP
from nets.utils import make_divisible,calc_block_num




class CSPNet(nn.Module):
    def __init__(self,in_channels,
                 depth_multiple=0.33,
                 width_multiple=0.50
                 ):
        super(CSPNet, self).__init__()
        channel_64 = make_divisible(64 * width_multiple, 8)
        channel_128 = make_divisible(128 * width_multiple, 8)
        channel_256 = make_divisible(256 * width_multiple, 8)
        channel_512 = make_divisible(512 * width_multiple, 8)
        channel_1024 = make_divisible(1024 * width_multiple, 8)

        self.stem = Focus(in_channels, channel_64, 3)
        self.layer1 = nn.Sequential(
            Conv(channel_64, channel_128, 3, 2),
            BottleneckCSP(channel_128, channel_128, calc_block_num(3, depth_multiple))
        )

        self.layer2 = nn.Sequential(
            Conv(channel_128, channel_256, 3, 2),
            BottleneckCSP(channel_256, channel_256, calc_block_num(9, depth_multiple))
        )

        self.layer3 = nn.Sequential(
            Conv(channel_256, channel_512, 3, 2),
            BottleneckCSP(channel_512, channel_512, calc_block_num(9, depth_multiple))
        )

        self.layer4 = nn.Sequential(
            Conv(channel_512, channel_1024, 3, 2),
            SPP(channel_1024, channel_1024),
            BottleneckCSP(channel_1024, channel_1024, calc_block_num(3, depth_multiple), shortcut=False)
        )

        self.channels = [channel_256, channel_512, channel_1024]  # out-channels of [c3,c4,c5]

    def forward(self,x):
        x = self.stem(x)
        x = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c3,c4,c5]

