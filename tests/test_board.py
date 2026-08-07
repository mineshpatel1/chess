import unittest

from game.board import *


class TestBitboard(unittest.TestCase):
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

    def test_value(self):
        for fen, val in (
            (STARTING_STATE, 0),
            ('rn1qk3/p1p1p3/8/3Q4/8/8/PPPPPP1P/RNBQKBNR b - - 0 1', 2780),
            ('rnbqkbnr/pppp1ppp/8/8/3q4/8/P2P1PPP/4KBNR w - - 0 1', -3150),
        ):
            _board = Board(fen=fen)
            self.assertEqual(_board.value, val)

    def test_weighted_value(self):
        # Note these cannot detect which King table is in use: all three place the Kings on
        # e1 and e8, which are mirror squares, so the two King terms cancel exactly whichever
        # table is selected. See test_king_position_by_game_phase.
        for fen, val in (
            (STARTING_STATE, 0),
            ('rn1qk3/p1p1p3/8/3Q4/8/8/PPPPPP1P/RNBQKBNR b - - 0 1', 2730),
            ('rnbqkbnr/pppp1ppp/8/8/3q4/8/P2P1PPP/4KBNR w - - 0 1', -3120),
        ):
            _board = Board(fen=fen)
            self.assertEqual(_board.weighted_value, val)

    def test_is_endgame(self):
        for fen, result in (
            (STARTING_STATE, False),
            ('rn1qk3/p1p1p3/8/3Q4/8/8/PPPPPP1P/RNBQKBNR b - - 0 1', False),
            ('rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w - - 0 1', True),  # Queens off
            ('4k3/8/8/8/8/8/8/2BQK1N1 w - - 0 1', True),  # Queen on, but only four pieces
            ('8/8/8/4k3/8/8/8/R6K w - - 0 1', True),  # King and Rook against a King
        ):
            self.assertEqual(Board(fen=fen).is_endgame, result, fen)

    def test_king_position_by_game_phase(self):
        """
        The King wants opposite things in each phase, so the two piece-square tables must not
        be selected the wrong way round. Each pair below is identical in material and differs
        only in where the White King stands.

        Orderings are asserted rather than totals: the tables are tuning constants, and
        pinning their sums would break on any retune.
        """
        # Endgame: no queens, so the King should walk towards the centre
        centralised = Board(fen='8/8/8/4k3/4K3/8/8/R7 w - - 0 1')
        cornered = Board(fen='8/8/8/4k3/8/8/8/R6K w - - 0 1')
        self.assertTrue(centralised.is_endgame)
        self.assertGreater(centralised.weighted_value, cornered.weighted_value)

        # Opening: queens and a full board, so the King should stay tucked away instead
        tucked = Board(fen='rn2k3/8/8/8/8/8/8/RN1Q2KR w - - 0 1')
        exposed = Board(fen='rn2k3/8/8/8/3K4/8/8/RN1Q3R w - - 0 1')
        self.assertFalse(tucked.is_endgame)
        self.assertEqual(tucked.value, exposed.value)  # Same material, only the King moved
        self.assertGreater(tucked.weighted_value, exposed.weighted_value)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
