import unittest

from engine.bitboard import *


class TestBoard(unittest.TestCase):
    def test_print(self):
        _board = BB_A1
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
        self.assertEqual(bitboard_to_str(_board), match)

        _board = BB_BOARD
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
        self.assertEqual(bitboard_to_str(_board), match)

        _board = BB_G5
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
        self.assertEqual(bitboard_to_str(_board), match)

        _board = BB_BLACK_SQUARES
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
        self.assertEqual(bitboard_to_str(_board), match)

    def test_print_board(self):
            _board = Board()
            match = ("""
8 [♜][♞][♝][♛][♚][♝][♞][♜]
7 [♟][♟][♟][♟][♟][♟][♟][♟]
6 [ ][ ][ ][ ][ ][ ][ ][ ]
5 [ ][ ][ ][ ][ ][ ][ ][ ]
4 [ ][ ][ ][ ][ ][ ][ ][ ]
3 [ ][ ][ ][ ][ ][ ][ ][ ]
2 [♙][♙][♙][♙][♙][♙][♙][♙]
1 [♖][♘][♗][♕][♔][♗][♘][♖]
   A  B  C  D  E  F  G  H """)
            self.assertEqual(str(_board), match)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
