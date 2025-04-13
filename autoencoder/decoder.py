import sys
sys.path.extend([".", "./src"])

import torch.nn as nn
import torch

from src.env import BalatroEnv

MODEL_DIR_PATH = "models"
DECODER_PATH = f"{MODEL_DIR_PATH}/decoder.pth"

EMB_DIM = 16

def main():
    """
    9-bit representation is bad because the model needs to learn that a 8-bit combination 
    means different actions when the last ACTION/DISCARD bit is on vs off.
    """
    X = torch.zeros((436,EMB_DIM))
    for i in range(436):
        X[i] = torch.tensor(BalatroEnv.action_index_to_embedding(i))

    y = torch.zeros(436, 436)
    for i in range(436):
        y[i,i] = 1

    decoder = nn.Sequential(
        nn.Linear(EMB_DIM,436),
        nn.Linear(436,436),
        nn.Linear(436,436),
        nn.Linear(436,436),
    )

    loss_fn = nn.MSELoss()
    for iteration in range(1000):
        optimizer = torch.optim.AdamW(decoder.parameters())

        y_predictions = decoder(X)

        optimizer.zero_grad()
        loss = loss_fn(y_predictions, y)
        print(f"Iteration #{iteration+1}, loss: {loss}")
        loss.backward()
        optimizer.step()

    print(s := torch.argmax(decoder(X), dim=1))
    correct = sum(1 if i == v.item() else 0 for i, v in enumerate(s))
    print("Num correct:", correct, "/ 436")
    torch.save(decoder, DECODER_PATH)



if __name__ == "__main__":
    main()
