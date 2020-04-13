import unittest

from engine import position
from engine.exceptions import FiftyMoveDraw, ThreefoldRepetition
from engine.position import Position

from engine import game
from engine.game import Game
from engine.constants import WHITE, BLACK, STARTING_STATE
from engine.bitboard import *


class TestBoard(unittest.TestCase):
    def test_print(self):
        _board = bitboard_to_str(BB_A1)
        match = ("""
8 [ ][ ][ ][ ][ ][ ][ ][ ]
7 [ ][ ][ ][ ][ ][ ][ ][ ]
6 [ ][ ][ ][ ][ ][ ][ ][ ]
5 [ ][ ][ ][ ][ ][ ][ ][ ]
4 [ ][ ][ ][ ][ ][ ][ ][ ]
3 [ ][ ][ ][ ][ ][ ][ ][ ]
2 [ ][ ][ ][ ][ ][ ][ ][ ]
1 [•][ ][ ][ ][ ][ ][ ][ ]
   A  B  C  D  E  F  G  H """)
        self.assertEqual(_board, match)

        _board = bitboard_to_str(BB_BOARD)
        match = ("""
8 [•][•][•][•][•][•][•][•]
7 [•][•][•][•][•][•][•][•]
6 [•][•][•][•][•][•][•][•]
5 [•][•][•][•][•][•][•][•]
4 [•][•][•][•][•][•][•][•]
3 [•][•][•][•][•][•][•][•]
2 [•][•][•][•][•][•][•][•]
1 [•][•][•][•][•][•][•][•]
   A  B  C  D  E  F  G  H """)
        self.assertEqual(_board, match)

        _board = bitboard_to_str(BB_G5)
        match = ("""
8 [ ][ ][ ][ ][ ][ ][ ][ ]
7 [ ][ ][ ][ ][ ][ ][ ][ ]
6 [ ][ ][ ][ ][ ][ ][ ][ ]
5 [ ][ ][ ][ ][ ][ ][•][ ]
4 [ ][ ][ ][ ][ ][ ][ ][ ]
3 [ ][ ][ ][ ][ ][ ][ ][ ]
2 [ ][ ][ ][ ][ ][ ][ ][ ]
1 [ ][ ][ ][ ][ ][ ][ ][ ]
   A  B  C  D  E  F  G  H """)
        self.assertEqual(_board, match)

        _board = bitboard_to_str(BB_BLACK_SQUARES)
        match = ("""
8 [ ][•][ ][•][ ][•][ ][•]
7 [•][ ][•][ ][•][ ][•][ ]
6 [ ][•][ ][•][ ][•][ ][•]
5 [•][ ][•][ ][•][ ][•][ ]
4 [ ][•][ ][•][ ][•][ ][•]
3 [•][ ][•][ ][•][ ][•][ ]
2 [ ][•][ ][•][ ][•][ ][•]
1 [•][ ][•][ ][•][ ][•][ ]
   A  B  C  D  E  F  G  H """)
        self.assertEqual(_board, match)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
