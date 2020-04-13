import unittest

from engine.bitboard import *


class TestBoard(unittest.TestCase):
    def test_precalc_moves(self):
        self.assertEqual(BB_KNIGHT_MOVES[D5], 5666883501293568)
        self.assertEqual(BB_KING_MOVES[D5], 30872694685696)
        self.assertEqual(BB_PAWN_ATTACKS[WHITE][D5], 21990232555520)
        self.assertEqual(BB_PAWN_ATTACKS[BLACK][D5], 335544320)

        self.assertEqual(BB_RAYS[NORTH][D5], 578721348210130944)
        self.assertEqual(BB_RAYS[NORTHEAST][D5], 4620710809868173312)
        self.assertEqual(BB_RAYS[EAST][D5], 1030792151040)
        self.assertEqual(BB_RAYS[SOUTHEAST][D5], 270549120)
        self.assertEqual(BB_RAYS[SOUTH][D5], 134744072)
        self.assertEqual(BB_RAYS[SOUTHWEST][D5], 67240192)
        self.assertEqual(BB_RAYS[WEST][D5], 30064771072)
        self.assertEqual(BB_RAYS[NORTHWEST][D5], 72624942037860352)

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
