import torch
from torch import nn
from nets.commons import Conv,BottleneckCSP
# from nets.utils import make_divisible,calc_block_num


class YOLOCSPNeck(nn.Module):
    def __init__(self, ch_3, ch_4, ch_5, csp_depth=1):
        super(YOLOCSPNeck, self).__init__()
        self.f5_latent = Conv(ch_5, ch_4, 1, 1)
        self.f4_conv = BottleneckCSP(ch_4 * 2, ch_4, csp_depth, shortcut=False)

        self.f4_latent = Conv(ch_4, ch_3, 1, 1)
        self.p3_out = BottleneckCSP(ch_3 * 2, ch_3, csp_depth, shortcut=False)

        self.p3_to_p4 = Conv(ch_3, ch_3, 3, 2)
        self.p4_out = BottleneckCSP(ch_3 * 2, ch_4, csp_depth, shortcut=False)

        self.p4_to_p5 = Conv(ch_4, ch_4, 3, 2)
        self.p5_out = BottleneckCSP(ch_4 * 2, ch_5, csp_depth, shortcut=False)

    def forward(self, xs):
        ch_3, ch_4, ch_5 = xs
        f5_latent = self.f5_latent(ch_5)
        f4 = self.f4_conv(torch.cat([nn.UpsamplingNearest2d(scale_factor=2)(f5_latent), ch_4], dim=1))
        f4_latent = self.f4_latent(f4)
        p3 = self.p3_out(torch.cat([nn.UpsamplingNearest2d(scale_factor=2)(f4_latent), ch_3], dim=1))
        p3_down_sample = self.p3_to_p4(p3)
        p4 = self.p4_out(torch.cat([p3_down_sample, f4_latent], dim=1))
        p4_down_sample = self.p4_to_p5(p4)
        p5 = self.p5_out(torch.cat([p4_down_sample, f5_latent], dim=1))
        return [p3, p4, p5]



    