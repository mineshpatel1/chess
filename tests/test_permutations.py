import unittest

from ai.benchmark import traverse_moves
from game.board import *


class TestPermutations(unittest.TestCase):
    """
    Perft move enumeration, verified against the reference counts published at
    https://www.chessprogramming.org/Perft_Results
    """

    def _assert_perft(self, fen: str, counts: dict):
        for depth in sorted(counts):
            board = Board(fen)
            self.assertEqual(
                traverse_moves(board, depth, False),
                counts[depth],
                f"perft({depth}) mismatch for {fen}",
            )

    def test_perft_starting_position(self):
        self._assert_perft(STARTING_STATE, {1: 20, 2: 400, 3: 8902})

        # Verified, but too slow for the suite: depth 4 = 197281, depth 5 = 4865609

    def test_perft_position_3(self):
        """
        An endgame with pawns abreast of both Kings on open ranks, so this position is only
        enumerated correctly if an en passant capture that exposes its own King is rejected.
        Both colours have such a capture available here.
        """
        self._assert_perft(
            '8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1',
            {1: 14, 2: 191, 3: 2812, 4: 43238},
        )

    def test_perft_position_5(self):
        """
        Black has a Knight on f2 bearing down on h1, so this position is only enumerated
        correctly if capturing a Rook on its original square removes the matching castling
        right. Otherwise White is offered a castle with a Rook that no longer exists.
        """
        self._assert_perft(
            'rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8',
            {1: 44, 2: 1486, 3: 62379},
        )


def main():
    unittest.main()


if __name__ == '__main__':
    main()
