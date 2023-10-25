import torch
import math
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


class Conv2D_initializer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias=False):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad)

    def forward(self, x):
       device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       l = self.conv.to(device)
       return l(x)

class adapter(nn.Module):
    def __init__(self, kernel=1, stride=1, pad=0):
        super().__init__()
        self.stride = stride
        self.pad = pad
        self.kernel = kernel
    
    def adapt(self, inp, out_channels):  
        net = Conv2D_initializer(inp.size(1), out_channels, self.kernel, self.stride, self.pad)
        out=net(inp)
        return out

    def convert4Dto2D(self, inp):
        _3Dinp = inp.shape[:-1]
        nb_voxels = np.prod(_3Dinp)
        return inp.reshape(nb_voxels, inp.shape[-1])
        
    def signspatternmatrix(self, a, out):
        inp = self.adapt(a, out)
        _inp =self.convert4Dto2D(inp)
        sgn = torch.empty(_inp.shape, dtype=torch.double)
        sgn = torch.where(_inp>0, 1, torch.where(_inp<0, -1, 0))
        inertia = [torch.count_nonzero(sgn==1), torch.count_nonzero(sgn==-1), torch.count_nonzero(sgn==0)]
        x = torch.tensor([float(i) for i in inertia])
        return sgn.float(), x 
                