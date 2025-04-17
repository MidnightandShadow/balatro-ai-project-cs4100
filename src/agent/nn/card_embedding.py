import torch
import torch.nn as nn
from src.agent.device import device
from src.common import Card

class CardEmbedding(nn.Module):
    def __init__(self, card_range_min : int, card_range_max : int, in_dim : int, emb_dim=26):
        super().__init__()
        self.range_min = card_range_min
        self.range_max = card_range_max
        self.emb_dim = emb_dim
        self.in_dim = in_dim
        # (52, 18) first index is the card num, second is the embedding
        self.register_buffer(
            "card_buffer", 
            torch.tensor([[Card.int_to_emb(i,p) for i in range(52)] for p in range(8)]).to(device)
        )
        self.register_buffer(
            "pos", torch.tensor([0,1,2,3,4,5,6,7])
        )

    def forward(self, x):
        dim = len(x.shape)
        if dim == 2:
            x = x.unsqueeze(1)
        xs = x[:,0,self.range_min:self.range_max].type(torch.int)
        out = torch.ones((x.shape[0],self.in_dim,self.emb_dim)).to(device)
        # MAGIC NUMBER 26 comes from the card embedding dimension.
        out[:,self.range_min:self.range_max,:26] = self.card_buffer[self.pos, xs]
        out[:,:self.range_min,:] *= x[:,:,:self.range_min].swapaxes(1,2).to(device)
        out[:,self.range_max:self.in_dim,:] *= x[:,:,self.range_max:self.in_dim].swapaxes(1,2).to(device)
        return out
