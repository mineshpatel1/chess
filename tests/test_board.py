import unittest

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
            0: 'r', 1: 'n', 2: 'b', 3: 'q', 4: 'k', 5: 'b', 6: 'n', 7: 'r', 8: 'p', 9: 'p', 10: 'p', 11: 'p', 12: 'p',
            13: 'p', 14: 'p', 15: 'p', 48: 'P', 49: 'P', 50: 'P', 51: 'P', 52: 'P', 53: 'P', 54: 'P', 55: 'P', 56: 'R',
            57: 'N', 58: 'B', 59: 'Q', 60: 'K', 61: 'B', 62: 'N', 63: 'R',
        }

        for piece in sorted(_board.pieces):
            self.assertEqual(_pieces[piece.pos.index], piece.code)

    def test_print(self):
        _board = Board(state=board.STARTING_STATE)
        match = ("""
8 [♜][♞][♝][♛][♚][♝][♞][♜]
7 [♟][♟][ ][♟][♟][♟][♟][♟]
6 [ ][ ][ ][ ][ ][ ][ ][ ]
5 [ ][ ][♟][ ][ ][ ][ ][ ]
4 [ ][ ][ ][ ][♙][ ][ ][ ]
3 [ ][ ][ ][ ][ ][♘][ ][ ]
2 [♙][♙][♙][♙][ ][♙][♙][♙]
1 [♖][♘][♗][♕][♔][♗][ ][♖]
   A  B  C  D  E  F  G  H """)
        self.assertEqual(str(_board), match)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
