import torch
import torch.nn as nn
from src.agent.device import device
from src.common import Card

class CardEmbedding(nn.Module):
    def __init__(self, card_range_min : int, card_range_max : int, in_dim : int, emb_dim=18):
        super().__init__()
        self.range_min = card_range_min
        self.range_max = card_range_max
        self.emb_dim = emb_dim
        self.in_dim = in_dim

    def forward(self, x):
        dim = len(x.shape)
        if dim == 2:
            x = x.unsqueeze(1)
        xs = x[:,0,self.range_min:self.range_max]
        out = torch.zeros((x.shape[0],self.in_dim,self.emb_dim)).to(device)
        for i, obs_hand in enumerate(xs):
            for j, card in enumerate(obs_hand):
                emb = Card.int_to_emb(int(card.item()))
                emb = emb + [1] * (self.emb_dim - len(emb)) # one extend emb
                out[i,self.range_min+j,:] = torch.tensor(emb).to(device)
        for i in range(x.shape[0]):
            for j in range(0, self.range_min):
                out[i,j,:] = torch.tensor([x[i,0,j].item()] * self.emb_dim).to(device)
        for i in range(x.shape[0]):
            for j in range(self.range_max, self.in_dim):
                out[i,j,:] = torch.tensor([x[i,0,j].item()] * self.emb_dim).to(device)
        return out
