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

    def test_illegal_moves(self):
        with self.assertRaises(IllegalMove):
            self.board.move(Position(0, 1), Position(0, 4))

        with self.assertRaises(IllegalMove):
            self.board.move(Position(0, 1), Position(0, 1))

        with self.assertRaises(IllegalMove):
            self.board.move(Position(0, 1), Position(0, -1))

        with self.assertRaises(IllegalMove):
            self.board.move(Position(0, 1), Position(1, 2))

    def test_forward(self):
        self.board.move(Position(0, 1), Position(0, 3))

        with self.assertRaises(IllegalMove):
            self.board.move(Position(0, 3), Position(0, 5))

        self.board.move(Position(0, 3), Position(0, 4))
        self.board.move(Position(0, 4), Position(0, 5))

    def test_take(self):
        self.board = Board(state='rnbqkbnr/pppppppp/P7/8/8/8/1PPPPPPP/RNBQKBNR')
        pawn1 = self.board.squares[Position(0, 5).index].piece
        self.assertTrue(pawn1.is_white)
        self.assertEqual({Position(1, 6)}, pawn1.legal_moves(self.board))

        pawn2 = self.board.squares[Position(1, 6).index].piece
        self.assertFalse(pawn2.is_white)

        self.board.move(Position(0, 5), Position(1, 6))
        self.assertFalse(pawn2 in self.board.pieces)
        self.assertEqual(Position(1, 6), pawn1.pos)

def main():
    unittest.main()


if __name__ == '__main__':
    main()
