import sys
sys.path.extend([".", "./src"])

import torch.nn as nn
import torch

from src.env import BalatroEnv

def main():
    X = torch.zeros((436,9))
    for i in range(436):
        X[i] = torch.tensor(BalatroEnv.action_index_to_embedding(i))

    y = torch.zeros(436, 436)
    for i in range(436):
        y[i,i] = 1

    decoder = nn.Sequential(
        nn.Linear(9,436),
        nn.Linear(436,436),
        nn.Linear(436,436),
    )

    print(X)
    print(y)

    loss_fn = nn.MSELoss()
    for iteration in range(10_000):
        optimizer = torch.optim.AdamW(decoder.parameters())

        y_predictions = decoder(X)

        optimizer.zero_grad()
        loss = loss_fn(y_predictions, y)
        print(f"Iteration #{iteration+1}, loss: {loss}")
        loss.backward()
        optimizer.step()

    print(torch.argmax(decoder(X), dim=1))



if __name__ == "__main__":
    main()
