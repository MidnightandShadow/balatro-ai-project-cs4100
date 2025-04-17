import sys
sys.path.extend([".", "./src"])

import torch.nn as nn
import torch

from src.env import BalatroEnv
from src.agent.device import device

EMB_DIM = 104
BATCH_SIZE = 256
DENORM = 10_000
N = 436

MODEL_DIR_PATH = "models"
DECODER_PATH = f"{MODEL_DIR_PATH}/decoder.pth"

def main():
    """
    26-bit internal representation for a played hand 
    """
    print(f"Using {device} device")

    encoder = nn.Sequential(
        nn.Linear(436, 436),
        #nn.LayerNorm(436),
        nn.Linear(436, 436),
        #nn.LayerNorm(436),
        nn.Linear(436, EMB_DIM),
        #nn.LayerNorm(EMB_DIM),
    )
    decoder = nn.Sequential(
        nn.Linear(EMB_DIM,436),
        #nn.LayerNorm(436),
        nn.Linear(436,436),
        #nn.LayerNorm(436),
        nn.Linear(436,436),
    )


    def loss_fn(pred, y):
        """ old loss """
        return torch.mean((pred - y)**2) / DENORM**2

    def top_loss_fn(pred, y):
        """
        We want the top N to be close
        """
        v = (torch.topk(pred, N, dim=1))
        vals, inds = v.values, v.indices
        pred = pred.gather(1, inds)
        y = y.gather(1, inds)
        return torch.mean((pred - y)**2) / DENORM**2

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters())
    )
    
    for iteration in range(10_000):
        X = torch.randn(BATCH_SIZE,436) * DENORM
        y = X

        y_predictions = decoder(encoder(X))

        optimizer.zero_grad()
        loss = (top_loss_fn if N != 436 else loss_fn)(y_predictions, y)
        print(f"Iteration #{iteration+1}, loss: {loss}")
        loss.backward()
        optimizer.step()
    torch.save(decoder, DECODER_PATH)


if __name__ == "__main__":
    main()
