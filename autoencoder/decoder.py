import sys
sys.path.extend([".", "./src"])

import torch.nn as nn
import torch

from src.env import BalatroEnv

MODEL_DIR_PATH = "models"
DECODER_PATH = f"{MODEL_DIR_PATH}/decoder.pth"

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

    loss_fn = nn.MSELoss()
    for iteration in range(10_000):
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
