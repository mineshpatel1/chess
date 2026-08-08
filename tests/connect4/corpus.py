"""
Positions to test against, generated once and written down.

A fixture that regenerates on every run is not a fixture: it moves when the code under it moves,
which is exactly what a fixture is for stopping. So the drawn game below is a literal, and the
comment records how it was found rather than the search being left in to run again.

`positions` is the exception, and is a corpus rather than a fixture - a reproducible spread of
positions for the search comparisons to range over, in the same shape as tests/chess/corpus.py.
"""

import random
from typing import List

from games.connect4.board import Connect4

# A full board that nobody won, found by playing seeded random games until one drew - the first
# was seed 1168, after 1,168 tries, which is about how rare a random draw is. Every column is
# full and no line of four exists, so this is the only way `outcome_without_moves` gets reached.
DRAWN_GAME: List[int] = [
    5, 5, 6, 3, 5, 0, 6, 5, 0, 0, 6, 5, 2, 5, 0, 3, 4, 6, 4, 0, 6,
    4, 6, 3, 3, 0, 2, 1, 2, 4, 3, 2, 3, 4, 2, 2, 4, 1, 1, 1, 1, 1,
]

# Games decided one way each, one per direction of line.
VERTICAL_WIN: List[int] = [3, 4, 3, 4, 3, 4, 3]
HORIZONTAL_WIN: List[int] = [0, 0, 1, 1, 2, 2, 3]
RISING_DIAGONAL_WIN: List[int] = [0, 1, 1, 2, 2, 3, 2, 3, 3, 6, 3]
FALLING_DIAGONAL_WIN: List[int] = [3, 2, 2, 1, 1, 0, 1, 0, 0, 6, 0]

# The second player winning, so that nothing is only ever tested from Yellow's side.
RED_WIN: List[int] = [6, 3, 6, 4, 6, 5, 0, 2]


def positions(count: int, seed: int = 0, plies: int = 8) -> List[List[int]]:
    """
    A reproducible spread of move sequences, each a position part-way through a game.

    Random rather than hand-picked, and reproducible rather than merely random: a comparison
    between two searches means nothing if the positions it ran over cannot be run again.
    Sequences that finish early are dropped, so every one of these is a live position with
    moves to choose between.
    """
    rng = random.Random(seed)
    found: List[List[int]] = []

    while len(found) < count:
        board = Connect4()
        for _ in range(plies):
            if board.is_game_over:
                break
            board.make_move(rng.choice(list(board.legal_moves)))
        else:
            if not board.is_game_over:
                found.append(board.columns_played)

    return found
