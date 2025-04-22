from src.agent.agent import Agent
from src.env import BalatroEnv
from src.player import Player
from src.strategy import Strategy


class HardcodedAgent(Agent):
    """
    Represents an agent that defers to some "human player"/hardcoded policies for taking actions.
    """
    def __init__(self, env: BalatroEnv, strategy: Strategy):
        super().__init__(env)
        self.player = Player(strategy)

    def get_action(self, obs) -> int:
        action = self.player.take_action(self.env.get_observable_state())
        return self.env.action_to_action_index(action)

    def update(
        self,
        obs,
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        """
        Currently left unimplemented, but there could be a hardcoded_agent that uses this information
        to pick when to use which hardcoded policies.
        """
        pass

