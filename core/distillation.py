import torch
import torch.nn as nn
from torch.nn import functional as F
from adaptation import Conv2D_initializer, adapter

class DistillerMKD:
    def __init__(self, reduction="batchmean", temperature=10):
        self.red = reduction
        self.temp = temperature

    def KL_loss(self,inp, out):
        kl_loss = nn.KLDivLoss(reduction=self.red)
        out = kl_loss(F.log_softmax(inp/self.temp), F.softmax(out/self.temp)*(self.temp**2))
        return out
    
    def RE_loss(self, inp, out):
        mse = nn.MSELoss()
        out = mse(inp, out)
        return out



