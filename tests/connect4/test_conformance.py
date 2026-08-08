"""
Connect 4 against the shared game contract.

This is the suite the contract was written for. Chess passing it is a weak signal, chess being
the game it was extracted from; Connect 4 is the first game to have to satisfy it without having
shaped it, and it exercises the two parts chess never reaches - `outcome`, for a game that can
be won while moves remain, and `outcome_without_moves`, for one that ends by filling up.

A whole game is 42 plies and a position is two integers, so the playout knobs are turned up well
past what chess can afford: 40 playouts of a full game each, against chess's 4 of ten plies.
"""

import unittest

from games.base import DRAW, win
from games.connect4.board import Connect4
from games.connect4.constants import COLS, RED, ROWS, YELLOW
from tests.conformance import GameConformanceTests
from tests.connect4.corpus import (
    DRAWN_GAME,
    FALLING_DIAGONAL_WIN,
    HORIZONTAL_WIN,
    RED_WIN,
    RISING_DIAGONAL_WIN,
    VERTICAL_WIN,
)


class TestConnect4Conformance(GameConformanceTests, unittest.TestCase):
    # Connect 4 branches seven ways and a position is two integers, so the whole game space is
    # cheap to walk. These run a full game per playout, which chess cannot do at any price.
    PLAYOUTS = 40
    PLAYOUT_PLIES = ROWS * COLS  # A whole game
    SEARCH_DEPTH = 3

    def new_game(self) -> Connect4:
        return Connect4()

    def decided_games(self):
        """One finish per direction of line, one for each player, and the full board."""
        return [
            (Connect4(VERTICAL_WIN), win(YELLOW)),
            (Connect4(HORIZONTAL_WIN), win(YELLOW)),
            (Connect4(RISING_DIAGONAL_WIN), win(YELLOW)),
            (Connect4(FALLING_DIAGONAL_WIN), win(YELLOW)),
            (Connect4(RED_WIN), win(RED)),
            (Connect4(DRAWN_GAME), DRAW),
        ]

    def forced_win_in_one(self):
        """Yellow has three along the bottom and both ends open."""
        return Connect4([1, 0, 2, 0, 3, 0]), YELLOW


def main():
    unittest.main()


if __name__ == '__main__':
    main()
