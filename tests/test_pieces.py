import unittest

import log
import board

from position import Position
from board import Board
from exceptions import IllegalMove


class TestBoard(unittest.TestCase):
    def setUp(self):
        self.board = Board(state=board.STARTING_STATE)

    def tearDown(self):
        self.board = None

    def test_pawn_illegal_moves(self):
        with self.assertRaises(IllegalMove):
            self.board.move(Position(0, 1), Position(0, 4))

        with self.assertRaises(IllegalMove):
            self.board.move(Position(0, 1), Position(0, 1))

        with self.assertRaises(IllegalMove):
            self.board.move(Position(0, 1), Position(0, -1))

        with self.assertRaises(IllegalMove):
            self.board.move(Position(0, 1), Position(1, 2))

    def test_pawn_forward(self):
        self.board.move(Position(0, 1), Position(0, 3))

        with self.assertRaises(IllegalMove):
            self.board.move(Position(0, 3), Position(0, 5))

        self.board.move(Position(0, 3), Position(0, 4))
        self.board.move(Position(0, 4), Position(0, 5))

    def test_pawn_take(self):
        self.board = Board(state='rnbqkbnr/pppppppp/P7/8/8/8/1PPPPPPP/RNBQKBNR')
        pawn1 = self.board.squares[Position(0, 5).index].piece
        self.assertTrue(pawn1.is_white)
        self.assertEqual({Position(1, 6)}, pawn1.legal_moves(self.board))

        pawn2 = self.board.squares[Position(1, 6).index].piece
        self.assertFalse(pawn2.is_white)

        self.board.move(Position(0, 5), Position(1, 6))
        self.assertFalse(pawn2 in self.board.pieces)
        self.assertEqual(Position(1, 6), pawn1.pos)

    def test_rook(self):
        self.board = Board(state='rnbqkbnr/pppppp1p/8/8/P1R1p3/8/1PPP1PPP/1NBQKBNR')
        rook = self.board.squares[Position(2, 3).index].piece

        with self.assertRaises(IllegalMove):  # Cannot move beyond opposition piece
            self.board.move(Position(2, 3), Position(5, 3))

        with self.assertRaises(IllegalMove):  # Cannot move to occupied square (of same colour)
            self.board.move(Position(2, 3), Position(0, 3))

        # Take a piece of opposite colour
        self.board.move(Position(2, 3), Position(1, 3))
        piece = self.board.squares[Position(4, 3).index].piece
        self.assertEqual('p', piece.code)
        self.board.move(Position(1, 3), Position(4, 3))
        self.assertFalse(piece in self.board.pieces)
        self.assertEqual(Position(4, 3), rook.pos)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
