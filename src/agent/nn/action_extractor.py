import torch
import torch.nn as nn

class ActionExtractor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert(len(x.shape) == 2)
        assert(x.shape[-1] == 6)
        
        # x : (B, 6)
        torch.where()
