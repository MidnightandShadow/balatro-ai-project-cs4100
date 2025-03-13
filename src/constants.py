import os

try:
    width = os.get_terminal_size().columns
except OSError:
    width = 88

# Game constants
SMALL_BLIND_CHIPS = 300
BIG_BLIND_CHIPS = 450
BOSS_BLIND_CHIPS = 600
HAND_ACTIONS = 4
DISCARD_ACTIONS = 4
OBSERVABLE_HAND_SIZE = 8

# Formatting constants
TEXT_EQUALS_SEPARATOR = "=" * width

TEXT_DASH_SEPARATOR = "-" * width
TEXT_HASH_SEPARATOR = "#" * width
