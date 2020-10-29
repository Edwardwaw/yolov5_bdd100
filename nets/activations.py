import torch
import torch.nn as nn
import torch.nn.functional as F


#################################################################################################

class Swish(nn.Module):
    @staticmethod
    def forward(self, x):
        return x*torch.sigmoid(x)


class HardSwish(nn.Module):
    @staticmethod
    def forward(self, x):
        # return x*F.hardsigmid(x)    # for torchscript and CoreML
        return x * F.hardtanh(x + 3, 0., 6.) / 6.    # for torchscript, CoreML and ONNX



class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx,i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx,grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    @staticmethod
    def forward(self, x):
        return SwishImplementation.apply(x)





###################################################################################################################
class Mish(nn.Module):
    @staticmethod
    def forward(x):
        return x * F.softplus(x).tanh()


class MishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):
        ctx.save_for_backward(x)
        return x.mul(torch.tanh(F.softplus(x)))

    @staticmethod
    def backward(ctx,grad_output):
        # x = ctx.saved_tensors[0]
        x = ctx.saved_variables[0]
        sx = torch.sigmoid(x)
        fx = F.softplus(x).tanh()
        return grad_output * (fx + x * sx * (1 - fx * fx))


class MemoryEfficientMish(nn.Module):
    @staticmethod
    def forward(self,x):
        return MemoryEfficientMish.apply(x)






# FReLU https://arxiv.org/abs/2007.11824 -------------------------------------------------------------------------------
class FReLU(nn.Module):
    def __init__(self, c1, k=3):  # ch_in, kernel
        super().__init__()
        self.conv = nn.Conv2d(c1, c1, k, 1, 1, groups=c1)
        self.bn = nn.BatchNorm2d(c1)

    def forward(self, x):
        return torch.max(x, self.bn(self.conv(x)))