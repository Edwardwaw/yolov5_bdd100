import torch
import math
from torch import nn
from nets.cspnet import CSPNet
from nets.utils import calc_block_num
from nets.necks import YOLOCSPNeck


default_anchors = [[10, 13, 16, 30, 33, 23],  # P3/8
                   [30, 61, 62, 45, 59, 119],  # P4/16
                   [116, 90, 156, 198, 373, 326]]  # P5/32
default_strides = [8., 16., 32.]


def model_scale(name="s"):
    name_dict = {
        "s": (0.33, 0.50),
        "m": (0.67, 0.75),
        "l": (1.00, 1.00),
        "x": (1.33, 1.25)
    }
    multiples = name_dict.get(name, None)
    if multiples is None:
        raise NotImplementedError("scale_name only support s,m,l,x")
    return multiples



class YOLOv5Head(nn.Module):
    def __init__(self, c3, c4, c5, num_cls, anchors=None, strides=None):
        super(YOLOv5Head, self).__init__()
        self.num_cls = num_cls # number of classes
        if anchors is None:
            anchors = default_anchors
        if strides is None:
            strides = default_strides
        self.strides = strides
        self.num_outputs = num_cls + 5  # number of outputs per anchor
        self.num_layers = len(anchors)  # number of detection layers, here is 3
        self.anchor_per_grid = len(anchors[0]) // 2   # number of anchors per location, here is 3
        self.grid = [torch.zeros(1)] * self.num_layers  # init grid, shape=[3]
        a = torch.tensor(anchors).float().view(self.num_layers, -1, 2)    # shape=[3,3,2],原图尺度
        self.anchor_grid = a.clone().view(self.num_layers, 1, -1, 1, 1, 2)   # shape(nl,1,na,1,1,2)/[3,1,3,1,1,2]
        self.m = nn.ModuleList(nn.Conv2d(x, self.num_outputs * self.anchor_per_grid, 1) for x in [c3, c4, c5])  # output conv

        # initialize detection layer weight
        for mi, s in zip(self.m, self.strides):  # from
            b = mi.bias.view(self.anchor_per_grid, -1)  # conv.bias(255) to (3,85)
            b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, 5:] += math.log(0.6 / (self.num_cls - 0.99))  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xs):
        '''
        return:
        if training, return [x (len=nl, x[i].shape=(bs,3,ny,nx,85)), normalized_anchor (shape=[3,3,2]) ]
        if inference, return (decode_pred)  decode_pred.shape=[bs,-1,no]  ,no=num_class+5
        '''
        assert self.num_layers == len(xs)
        z = []
        for i in range(self.num_layers):
            xs[i] = self.m[i](xs[i])   # conv
            bs, _, ny, nx = xs[i].shape
            # x(bs,255,20,20) to x(bs,3,20,20,85) 生成预测值
            xs[i] = xs[i].view(bs, self.anchor_per_grid, self.num_outputs, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if xs[i].device != self.anchor_grid.device:
                self.anchor_grid = self.anchor_grid.to(xs[i].device)

            # if inference, first generate a set of grid points in current featuremap.
            # and then, decode output
            if not self.training:
                if self.grid[i].shape[2:4] != xs[i].shape[2:4] or self.grid[i].device != xs[i].device:
                    # shape=(1, 1, ny, nx, 2) 每个grid的点代表当前featuremap上的坐标点
                    self.grid[i] = self.make_grid(nx, ny).to(xs[i].device)
                # decode output and generate predicted boxes in input size.
                '''
                   xy_pred=(2*sigmoid(xy_pred)-0.5+grid_xy)*stride
                   wh_pred=(2*sigmoid(wh_pred))**2*anchor_wh(原图尺度)
                   
                   note: 2*sigmoid(xy_pred)-0.5 在(-0.5,1.5)的范围内,每个anchor box可以预测相邻grid的所属目标，
                         这样会产生更多的正目标
                   
                   why xy_pred can be computed by above formaluation? 
                     Generally, the groundtruth bbox would be matched with three grids (contains two neighbor grids), 
                     so the regression arange should be -0.5~1.5. More details refer to build_targets()
                '''
                y = xs[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.strides[i]  #xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.num_outputs))
        if self.training:
            strides = torch.tensor(data=self.strides, device=self.anchor_grid.device).view(-1, 1, 1)
            normalized_anchor = self.anchor_grid.view(self.num_layers, -1, 2) / strides
            return xs, normalized_anchor
        else:
            return torch.cat(z, 1)

    @staticmethod
    def make_grid(nx, ny):
        # return meshgrid
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()




class YOLOv5(nn.Module):
    def __init__(self,
                 in_channels=3,
                 num_cls=80,
                 scale_name='s',
                 strides=None,
                 anchors=None
                 ):
        super(YOLOv5, self).__init__()
        depth_multiple, width_multiple = model_scale(scale_name)
        self.backbone=CSPNet(in_channels,depth_multiple,width_multiple)
        ch_3,ch_4,ch_5=self.backbone.channels
        self.neck=YOLOCSPNeck(ch_3,ch_4,ch_5,csp_depth=calc_block_num(3,depth_multiple))
        self.head=YOLOv5Head(ch_3,ch_4,ch_5,num_cls,anchors,strides)
    def forward(self,x):
        x=self.head(self.neck(self.backbone(x)))
        return x




if __name__ == '__main__':
    input_tesnor=torch.rand(size=(4,3,416,416))
    # net=CSPNet(3)
    # c3,c4,c5=net.channels
    # from nets.necks import YOLOCSPNeck
    # head=YOLOCSPNeck(c3,c4,c5,calc_block_num(3,0.33))
    # out=head(net(input_tesnor))
    # for item in out:
    #     print(item.shape)

    net=YOLOv5(3,80).eval()
    out=net(input_tesnor)
    print(out.shape)


