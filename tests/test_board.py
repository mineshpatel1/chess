import unittest

import log
import board
import position

from board import Board
from position import Position
from constants import WHITE, BLACK


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

    def test_coord_to_position(self):
        for coord, pos in [
            ('A1', Position(0, 0)),
            ('E7', Position(4, 6)),
            ('D5', Position(3, 4)),
            ('H8', Position(7, 7)),
        ]:
            self.assertEqual(position.from_coord(coord), pos)

    def test_start_layout(self):
        _board = Board(state=board.STARTING_STATE)
        _pieces = {
            0: 'R', 1: 'N', 2: 'B', 3: 'Q', 4: 'K', 5: 'B', 6: 'N', 7: 'R', 8: 'P', 9: 'P', 10: 'P', 11: 'P', 12: 'P',
            13: 'P', 14: 'P', 15: 'P', 48: 'p', 49: 'p', 50: 'p', 51: 'p', 52: 'p', 53: 'p', 54: 'p', 55: 'p', 56: 'r',
            57: 'n', 58: 'b', 59: 'q', 60: 'k', 61: 'b', 62: 'n', 63: 'r',
        }

        for piece in sorted(_board.pieces):
            self.assertEqual(_pieces[piece.pos.index], piece.code)

    def test_check(self):
        for test in [
            ('rnbqkb2/ppppp1p1/5p2/2n1r2p/3KP3/3P4/PPP2PPP/RNBQ1BNR', WHITE, False),
            ('rnb1kbnr/pppp1ppp/4p3/8/7q/5P2/PPPPP1PP/RNBQKBNR', WHITE, True),
            ('rnb1kbnr/pppp1ppp/4p3/7q/8/BP3P2/P1PPP1PP/RN1QKBNR', BLACK, False),
            ('rnb2bnr/ppppkppp/4p3/7q/8/BP3P2/P1PPP1PP/RN1QKBNR', BLACK, True),
        ]:
            _board = Board(state=test[0])
            self.assertEqual(_board.is_in_check(test[1]), test[2])

    def test_checkmate(self):
        for test in [
            ('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR', WHITE, False),
            ('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR', BLACK, False),
            ('rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR', WHITE, True),
            ('rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR', BLACK, False),
            ('3q1bRk/5p2/5N1p/8/8/8/2r2PPP/6K1', WHITE, False),
            ('3q1bRk/5p2/5N1p/8/8/8/2r2PPP/6K1', BLACK, True),
        ]:
            _board = Board(state=test[0])
            self.assertEqual(_board.is_checkmate(test[1]), test[2])

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
