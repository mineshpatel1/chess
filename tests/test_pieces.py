import unittest

import log
import board
import position

from position import Position
from board import Board
from exceptions import IllegalMove


class TestBoard(unittest.TestCase):
    def tearDown(self):
        self.board = None

    def test_pawn_illegal_moves(self):
        self.board = Board(state=board.STARTING_STATE)

        with self.assertRaises(IllegalMove):
            self.board.move(Position(0, 1), Position(0, 4))

        with self.assertRaises(IllegalMove):
            self.board.move(Position(0, 1), Position(0, 1))

        with self.assertRaises(IllegalMove):
            self.board.move(Position(0, 1), Position(0, -1))

        with self.assertRaises(IllegalMove):
            self.board.move(Position(0, 1), Position(1, 2))

    def test_pawn_forward(self):
        self.board = Board(state=board.STARTING_STATE)
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

        self.assertEqual(
            rook.legal_moves(self.board),
            {position.from_coord(c) for c in {'B4', 'C7', 'D4', 'E4', 'C3', 'C6', 'C5'}}
        )

        with self.assertRaises(IllegalMove):  # Cannot move diagonally
            self.board.move(Position(2, 3), Position(3, 4))

        with self.assertRaises(IllegalMove):  # Cannot move beyond opposition piece
            self.board.move(Position(2, 3), Position(5, 3))

        with self.assertRaises(IllegalMove):  # Cannot move to occupied square (of same colour)
            self.board.move(Position(2, 3), Position(0, 3))

        self.board.move(Position(2, 3), Position(1, 3))  # Move cardinally

        # Take a piece of opposite colour
        piece = self.board.squares[Position(4, 3).index].piece
        self.assertEqual('p', piece.code)
        self.board.move(Position(1, 3), Position(4, 3))
        self.assertFalse(piece in self.board.pieces)
        self.assertEqual(Position(4, 3), rook.pos)

    def test_bishop(self):
        self.board = Board(state='rnbqkbnr/pppppppp/8/8/2B5/1P2P3/1PPP1PPP/RNBQK1NR')
        bishop = self.board.squares[Position(2, 3).index].piece

        self.assertEqual(
            bishop.legal_moves(self.board),
            {position.from_coord(c) for c in {'D3', 'E2', 'E6', 'F7', 'B5', 'A6', 'F1', 'D5'}}
        )

        with self.assertRaises(IllegalMove):  # Cannot move beyond opposition piece
            self.board.move(Position(2, 3), Position(6, 6))

        with self.assertRaises(IllegalMove):  # Cannot move to occupied square (of same colour)
            self.board.move(Position(2, 3), Position(0, 1))

        with self.assertRaises(IllegalMove):  # Cannot move cardinally
            self.board.move(Position(2, 3), Position(1, 3))

        self.board.move(Position(2, 3), Position(4, 5))  # Move diagonally

        # Take a piece of opposite colour
        piece = self.board.squares[Position(5, 6).index].piece
        self.assertEqual('p', piece.code)
        self.board.move(Position(4, 5), Position(5, 6))
        self.assertFalse(piece in self.board.pieces)
        self.assertEqual(Position(5, 6), bishop.pos)

    def test_queen(self):
        self.board = Board(state='rnbqkbnr/pppppp1p/B7/8/2Q3p1/1P2P3/1PPP1PPP/RNB1K1NR')
        queen = self.board.squares[Position(2, 3).index].piece

        self.assertEqual(
            queen.legal_moves(self.board),
            {
                position.from_coord(c) for c in
                {
                    'D3', 'B4', 'C7', 'D4', 'E2', 'F4',
                    'E6', 'F7', 'B5', 'G4', 'C6', 'E4',
                    'C3', 'F1', 'A4', 'D5', 'C5',
                }
            }
        )

        with self.assertRaises(IllegalMove):  # Cannot move beyond opposition piece
            self.board.move(Position(2, 3), Position(6, 7))

        with self.assertRaises(IllegalMove):  # Cannot move to occupied square (of same colour)
            self.board.move(Position(2, 3), Position(1, 2))

        with self.assertRaises(IllegalMove):  # Can only move cardinally and diagonally
            self.board.move(Position(2, 3), Position(6, 6))

        self.board.move(Position(2, 3), Position(3, 3))  # Move cardinally
        self.board.move(Position(3, 3), Position(6, 6))  # Move diagonally

        # Take a piece of opposite colour
        piece = self.board.squares[Position(7, 7).index].piece
        self.assertEqual('r', piece.code)
        self.board.move(Position(6, 6), Position(7, 7))
        self.assertFalse(piece in self.board.pieces)
        self.assertEqual(Position(7, 7), queen.pos)

def main():
    unittest.main()


if __name__ == '__main__':
    main()
