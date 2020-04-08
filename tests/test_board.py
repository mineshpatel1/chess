import unittest

import log
import board
import position
from board import Board


class TestBoard(unittest.TestCase):
    def _verify_pos(self, index, file, rank, coord):
        f, r = position.index_to_rankfile(index)
        self.assertEqual((f, r), (file, rank))

        i = position.rankfile_to_index(f, r)
        self.assertEqual(i, index)

        p = position.from_index(index)
        self.assertEqual((p.file, p.rank, str(p)), (file, rank, coord))

    def test_positions(self):
        for index, file, rank, coord in [
            (0, 0, 0, 'A1'),   # Bottom-Left
            (7, 7, 0, 'H1'),   # Bottom-Right
            (56, 0, 7, 'A8'),  # Top-Left
            (63, 7, 7, 'H8'),  # Top-Right

            (12, 4, 1, 'E2'),  # Misc positions
            (38, 6, 4, 'G5'),
            (53, 5, 6, 'F7'),
        ]:
            self._verify_pos(index, file, rank, coord)

    def test_start_layout(self):
        _board = Board(state=board.STARTING_STATE)
        _pieces = {
            0: 'R', 1: 'N', 2: 'B', 3: 'Q', 4: 'K', 5: 'B', 6: 'N', 7: 'R', 8: 'P', 9: 'P', 10: 'P', 11: 'P', 12: 'P',
            13: 'P', 14: 'P', 15: 'P', 48: 'p', 49: 'p', 50: 'p', 51: 'p', 52: 'p', 53: 'p', 54: 'p', 55: 'p', 56: 'r',
            57: 'n', 58: 'b', 59: 'q', 60: 'k', 61: 'b', 62: 'n', 63: 'r',
        }

        for piece in sorted(_board.pieces):
            self.assertEqual(_pieces[piece.pos.index], piece.code)

    def test_fen(self):
        for fen in [
            'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR',
            'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR',
            'rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR',
            'rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R',
        ]:
            _board = board.Board(fen)
            self.assertEqual(_board.fen, fen)

    def test_print(self):
        _board = Board(state=board.STARTING_STATE)
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
