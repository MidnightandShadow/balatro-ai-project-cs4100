import torch.nn as nn

class Select(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        assert(len(x.shape) == 3)
        if self.dim == 0:
            return x[0,:,:]
        if self.dim == 1:
            return x[:,0,:]
        if self.dim == 2:
            return x[:,:,0]
