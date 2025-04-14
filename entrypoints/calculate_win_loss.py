# Not sure how to get rid of this... lmk if someone finds a workaround
import sys
sys.path.extend([".", "./src"])

import torch
import argparse

from src.env import BalatroEnv

ITERATIONS = 100_000

def main():
    parser = argparse.Parser()
    parser.add_argument("model_path")
    args = parser.parse_args()
    model_path = args.model_path
    model = load_model(model_path)
    evaluate_model(model)

def load_model(path:str):
    return torch.load(path, weights_only=False)

def evaluate_model(model:torch.nn.Module):
    env = BalatroEnv(INITIAL_GAME_STATE, manager)
    agent = None #TODO: fill in agent instantiation over here
    win_counter = 0
    for _ in tdqm(range(ITERATIONS)):
        cur_state, _ = env.reset()
        done = False
        while not done:
            action = agent.get_action(cur_state)
            try:
                state, _, done, _, _ = env.step(action)
            except IllegalActionException:
                break
        if env.game_state.did_player_win():
            win_counter += 1
    print(f"Win-loss: {win_counter/ITERATIONS:.2f}")


if __name__ == "__main__":
    main()
