from src.common import Card

import torch
import torch.nn as nn
from src.agent.nn.positional_encoding import PositionalEncoding
from src.agent.device import device

class BalatroTransformer(nn.Module):
    def __init__(self):
        """

        DQNv1 was a success! Although, it's troubling me that it discards so infrequently.
        I think the reward function is foolproof -- I think maybe the bigger issue here is that the 
        hand, discard, and deck information need to be passed into the beginning of the NN in order to make 
        an informed decision about which of the 436 actions to pick.

        """
        super(BalatroTransformer, self).__init__()

        self.NHEADS = 18
        self.EMBDIM = 450
        self.NLAYERS = 4

        self.pe =  PositionalEncoding(self.EMBDIM).to(device)
        self.mha = nn.ParameterList(
            nn.MultiheadAttention(self.EMBDIM, self.NHEADS, batch_first=True).to(device)
            for _ in range(self.NLAYERS)
        )
        self.key = nn.ParameterList(
            torch.zeros((1,self.NHEADS,self.EMBDIM)).to(device) for _ in range(self.NLAYERS)
        )
        self.val = nn.ParameterList(
            torch.zeros((1,self.NHEADS,self.EMBDIM)).to(device) for _ in range(self.NLAYERS)
        )
        self.lin = nn.ParameterList(
            nn.Linear(self.EMBDIM,self.EMBDIM).to(device) for _ in range(self.NLAYERS)
        )
        self.batch_norm = nn.ParameterList(
            nn.BatchNorm1d(num_features=63).to(device) for _ in range(self.NLAYERS)
        )
        self.dropout = nn.Dropout(p=0.1)
        self.relu = nn.ReLU().to(device)

        self.flatten = nn.Flatten().to(device)
        self.softmax = nn.Softmax(dim=1).to(device)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)

        x.to(device)
        batch_size = x.shape[0]

        # x        :: (B,1,63)
        # obs_hand :: (B,8)
        # decks    :: (B,52)
        # rest     :: (B,3)
        obs_hands = x[:,:,0:8].squeeze(1).to(device)
        decks     = x[:,:,8:60].squeeze(1).to(device)
        rest      = x[:,:,60:].squeeze(1).to(device)

        """ EMBEDDING """
        ob_hand_embeddings = torch.zeros((batch_size,8,self.EMBDIM)).to(device)
        deck_embeddings = torch.zeros((batch_size,52,self.EMBDIM)).to(device)
        rest_embeddings = torch.zeros((batch_size,3,self.EMBDIM)).to(device)
        for i, obs_hand in enumerate(obs_hands):
            for j, card in enumerate(obs_hand):
                emb = Card.int_to_emb(int(card.item()))
                emb = emb + [1] * (self.EMBDIM - len(emb)) # one extend emb
                ob_hand_embeddings[i,j] = torch.tensor(emb)
        for i, deck in enumerate(decks):
            for j, present_bit in enumerate(deck):
                if present_bit:
                    emb = Card.int_to_emb(j)
                    emb = emb + [1] * (self.EMBDIM - len(emb)) # one extend emb
                    deck_embeddings[i,j] = torch.tensor(emb)
        for i, r in enumerate(rest):
            for j, val in enumerate(r):
                emb = [val] * (self.EMBDIM)
                rest_embeddings[i,j] = torch.tensor(emb)


        """ POSITION ENCODE THE OBSERVABLE HAND """
        ob_hand_embeddings = self.pe(ob_hand_embeddings)

        """ CONCATENATE ALL EMBEDDINGS """
        x = torch.concatenate((ob_hand_embeddings, deck_embeddings, rest_embeddings), dim=1)

        """ MULTI-ATTENTION BLOCKS """
        K = torch.ones((batch_size, self.EMBDIM, self.NHEADS)).to(device)
        V = torch.ones((batch_size, self.EMBDIM, self.NHEADS)).to(device)

        for i in range(self.NLAYERS):
            x = self.mha[i].forward(
                x, 
                K@self.key[i],
                V@self.val[i],
                need_weights=False
            )[0]
            x = self.lin[i].forward(x)
            x = self.relu(x)
            x = self.batch_norm[i](x)
            x = self.dropout(x)
        x = x[:,0,:436]
        x = self.softmax(x)

        return x



