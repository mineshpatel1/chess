"""
Tic-tac-toe against the shared game contract.

The suite in tests/conformance.py has carried a sketch of this file in its docstring since the
contract was written, as an illustration of what adding a game would look like. This is it, and
the sketch turned out to be accurate: three hooks and nothing else.

It is also the game that finally exercises the contract at full strength. Chess shaped the
contract, so its passing proves little; Connect 4 could walk whole games but not all of them.
Here a position is two nine-bit integers and the entire game is 255,168 playouts, so the knobs go
up to the point where the "sampling" is no longer sampling anything - PLAYOUTS is high enough
that the seeded walks cover a large share of the openings, and SEARCH_DEPTH is the whole game,
which no other game in the project can afford.
"""

import unittest

from games.base import DRAW, win
from games.tictactoe.board import TicTacToe
from games.tictactoe.constants import CELLS, CROSS, NOUGHT
from tests.conformance import GameConformanceTests
from tests.tictactoe.corpus import (
    ANTI_DIAGONAL_WIN,
    COLUMN_WIN,
    DIAGONAL_WIN,
    DRAWN_GAME,
    NOUGHT_WIN,
    ROW_WIN,
)


class TestTicTacToeConformance(GameConformanceTests, unittest.TestCase):
    # Nine cells and a position that is two small integers, so everything here is cheap. A whole
    # game per playout, a hundred playouts, and a search that solves the game outright - where
    # Connect 4 runs 40 playouts at depth 3 and chess 4 playouts of ten plies at depth 2.
    PLAYOUTS = 100
    PLAYOUT_PLIES = CELLS  # A whole game
    SEARCH_DEPTH = TicTacToe.SOLVED_DEPTH

    def new_game(self) -> TicTacToe:
        return TicTacToe()

    def decided_games(self):
        """One finish per direction of line, one for each player, and the full board."""
        return [
            (TicTacToe(ROW_WIN), win(CROSS)),
            (TicTacToe(COLUMN_WIN), win(CROSS)),
            (TicTacToe(DIAGONAL_WIN), win(CROSS)),
            (TicTacToe(ANTI_DIAGONAL_WIN), win(CROSS)),
            (TicTacToe(NOUGHT_WIN), win(NOUGHT)),
            (TicTacToe(DRAWN_GAME), DRAW),
        ]

    def forced_win_in_one(self):
        """Crosses have two of the top row and Noughts have not noticed."""
        return TicTacToe([0, 3, 1, 4]), CROSS


def main():
    unittest.main()


if __name__ == '__main__':
    main()
