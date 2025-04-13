import sys
sys.path.extend([".", "./src"])

import torch.nn as nn
import torch

from src.env import BalatroEnv
from src.agent.device import device

EMB_DIM = 16

def main():
    """
    18-bit internal representation for a played hand 
    """
    print(f"Using {device} device")

    encoder = nn.Sequential(
        nn.Linear(436, EMB_DIM)
    )

    X = torch.zeros((436, 436))
    for i in range(436):
        X[i,i] = 1
    y = torch.zeros((436, EMB_DIM))
    for i in range(436):
        y[i] = torch.tensor(BalatroEnv.action_index_to_embedding(i))

    loss_fn = nn.SmoothL1Loss()
    
    for iteration in range(10_000):
        optimizer = torch.optim.AdamW(encoder.parameters())

        y_predictions = encoder(X)

        optimizer.zero_grad()
        loss = loss_fn(y_predictions, y)
        print(f"Iteration #{iteration+1}, loss: {loss}")
        loss.backward()
        optimizer.step()
    print(encoder(X))


if __name__ == "__main__":
    main()
