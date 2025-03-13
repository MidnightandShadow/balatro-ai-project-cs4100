from src.constants import (
    TEXT_HASH_SEPARATOR,
    SMALL_BLIND_CHIPS,
    HAND_ACTIONS,
    DISCARD_ACTIONS,
)
from src.game_state import INITIAL_GAME_STATE, GameState, generate_deck
from src.observer import PlayerObserver, Observer
from src.observer_manager import ObserverManager
from src.player import Player
from src.simulator import simulate_game
from src.strategy import *

# This file contains top-level scripts to run games with strategies (and observe as desired).


def play_a_single_default_game_with_a_single_strategy_and_observe_it(
    strategy: Strategy, observers: list[Observer]
):
    print("GAME START")
    print(f"Strategy chosen: {strategy}")
    print(TEXT_HASH_SEPARATOR)
    player = Player(strategy)

    observer_manager = ObserverManager()
    for observer in observers:
        observer_manager.add_observer(observer)

    simulate_game(INITIAL_GAME_STATE, player, observer_manager)

    print(TEXT_HASH_SEPARATOR)
    print("GAME END")


def play_games_with_a_single_strategy(
    strategy: Strategy, games_to_play=1, blind_chips=SMALL_BLIND_CHIPS
):
    print("GAME START")
    print(f"Chips to beat: {blind_chips}\nStrategy chosen: {strategy}")
    print(TEXT_HASH_SEPARATOR)
    player = Player(strategy)

    observer_manager = ObserverManager()

    times_won = 0
    times_lost = 0
    for _ in range(games_to_play):
        new_deck = generate_deck()
        new_game = GameState(
            blind_chips=blind_chips,
            scored_chips=0,
            hand_actions=HAND_ACTIONS,
            discard_actions=DISCARD_ACTIONS,
            deck=new_deck,
        )
        player_won = simulate_game(new_game, player, observer_manager)
        if player_won:
            times_won += 1
        else:
            times_lost += 1

    print(f"times won: {times_won}/{games_to_play}")
    print(f"times lost: {times_lost}/{games_to_play}")
    percent_won = (times_won / games_to_play) * 100
    print(f"% won: {percent_won}%")
    print(f"% lost: {100 - percent_won}%")
    print(TEXT_HASH_SEPARATOR)
    print("GAME END")
