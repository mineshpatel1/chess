import unittest

from ai.benchmark import traverse_moves
from game.board import *


class TestPermutations(unittest.TestCase):
    def test_perft(self):
        # Verifying possible moves from starting state, as per https://www.chessprogramming.org/Perft_Results
        b = Board(STARTING_STATE)
        self.assertEqual(traverse_moves(b, 1, False), 20)
        self.assertEqual(traverse_moves(b, 2, False), 400)
        self.assertEqual(traverse_moves(b, 3, False), 8902)

        # Following tests have been verified but are slow to test
        # self.assertEqual(traverse_moves(b, 4, False), 197281)  # 5 seconds
        # self.assertEqual(traverse_moves(b, 5, False), 4865609) # 2 minutes


def main():
    unittest.main()


if __name__ == '__main__':
    main()
