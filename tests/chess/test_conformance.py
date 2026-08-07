"""
Chess against the shared game contract.

Chess is the game the contract was extracted from, so it passing is a weak signal on its own.
It is here because the contract has to be exercised by something until a second game arrives,
and because it catches the regression where a change made for another game quietly breaks the
one that already worked.
"""

import unittest

from games.base import DRAW, win
from games.chess.board import ChessBoard
from games.chess.constants import WHITE, BLACK
from tests.conformance import GameConformanceTests


class TestChessConformance(GameConformanceTests, unittest.TestCase):
    # Chess branches far wider than the games this contract exists to admit, so the playouts
    # are kept shorter than a small game would want.
    PLAYOUTS = 4
    PLAYOUT_PLIES = 10
    SEARCH_DEPTH = 2

    def new_game(self) -> ChessBoard:
        return ChessBoard()

    def decided_games(self):
        return [
            # White mated by Qh4, so Black won
            (
                ChessBoard('rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 0 1'),
                win(BLACK),
            ),
            # Black to move, nowhere to go, not in check
            (ChessBoard('7k/5Q2/6K1/8/8/8/8/8 b - - 0 1'), DRAW),
            # Bishop and King cannot mate
            (ChessBoard('7k/8/8/8/8/8/8/6BK w - - 0 1'), DRAW),
            # The halfmove clock has run out
            (ChessBoard('4k3/8/8/8/8/8/8/R3K3 w - - 100 60'), DRAW),
        ]

    def forced_win_in_one(self):
        return ChessBoard('6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1'), WHITE  # Ra8 is mate


def main():
    unittest.main()


if __name__ == '__main__':
    main()
