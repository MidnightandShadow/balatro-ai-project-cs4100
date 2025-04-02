# Not sure how to get rid of this... lmk if someone finds a workaround
import sys

from torch import cross
sys.path.extend([".", "./src"])

from gymnasium.spaces.utils import flatten

from src.referee import *
from src.env import BalatroEnv
from src.agent.policy_nn import DQNAgent

NUM_GAMES = 100
if __name__ == "__main__":
    manager = ObserverManager()
    manager.add_observer(PlayerObserver())
    env = BalatroEnv(INITIAL_GAME_STATE, manager)
    agent = DQNAgent(env)

    for _ in range(NUM_GAMES):
        cur_state, _ = env.reset()
        cur_state = flatten(env.observation_space, cur_state)
        done = False
        while not done:
            action = agent.get_action(cur_state)
            nxt_state, reward, terminated, truncated, _ = env.step(action)
            nxt_state = flatten(env.observation_space, nxt_state)
            agent.update(cur_state, action, reward, terminated, nxt_state)
            cur_state = nxt_state
            done = terminated
