from src.common import Action, ScoredHand
from src.observable_state import ObservableState
from src.observer import Observer


# An ObserverManager maintains a list of Observers to notify whenever the manager receives any game-related updates.
class ObserverManager:
    def __init__(self):
        self.observers: list[Observer] = []

    def add_observer(self, observer: Observer):
        self.observers.append(observer)

    def notify_observers_state(self, observable_state: ObservableState):
        for observer in self.observers:
            observer.notify_state(observable_state)

    def notify_observers_turn(self, observable_state: ObservableState, action: Action, scored_hand: ScoredHand):
        for observer in self.observers:
            observer.notify_turn(observable_state, action, scored_hand)

    def notify_observers_game_over(self, player_won: bool):
        for observer in self.observers:
            observer.notify_game_over(player_won)
