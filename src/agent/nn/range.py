import torch.nn as nn

class Range(nn.Module):
    def __init__(self, l, r):
        super().__init__()
        self.l = l
        self.r = r

    def forward(self, x):
        assert(len(x.shape) == 2)
        return x[:,self.l:self.r]
