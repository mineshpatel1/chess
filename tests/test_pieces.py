import unittest

from engine import position
from engine.exceptions import IllegalMove
from engine.position import Position

from engine import board
from engine.board import Board
from engine.constants import WHITE, BLACK, ROOK, KING, QUEEN


class TestPieces(unittest.TestCase):
    def tearDown(self):
        self.board = None

    def test_pawn_illegal_moves(self):
        self.board = Board(state=board.STARTING_STATE)

        with self.assertRaises(IllegalMove):
            self.board._move(Position(0, 1), Position(0, 4))

        with self.assertRaises(IllegalMove):
            self.board._move(Position(0, 1), Position(0, 1))

        with self.assertRaises(IllegalMove):
            self.board._move(Position(0, 1), Position(0, -1))

        with self.assertRaises(IllegalMove):
            self.board._move(Position(0, 1), Position(1, 2))

    def test_pawn_forward(self):
        self.board = Board(state=board.STARTING_STATE)
        self.board._move(Position(0, 1), Position(0, 3))

        with self.assertRaises(IllegalMove):
            self.board._move(Position(0, 3), Position(0, 5))

        self.board._move(Position(0, 3), Position(0, 4))
        self.board._move(Position(0, 4), Position(0, 5))

    def test_pawn_take(self):
        self.board = Board(state='rnbqkbnr/pppppppp/P7/8/8/8/1PPPPPPP/RNBQKBNR')
        pawn1 = self.board.squares[Position(0, 5).index].piece
        self.assertEqual(pawn1.colour, WHITE)
        self.assertEqual({Position(1, 6)}, pawn1.legal_moves(self.board))

        pawn2 = self.board.squares[Position(1, 6).index].piece
        self.assertEqual(pawn2.colour, BLACK)

        self.board._move(Position(0, 5), Position(1, 6))
        self.assertFalse(pawn2 in self.board.pieces)
        self.assertEqual(Position(1, 6), pawn1.pos)

    def test_pawn_en_passant(self):
        # White En Passant
        self.board = Board(state='rnbqkbnr/pppppp1p/8/5Pp1/8/8/PPPPP1PP/RNBQKBNR w KQkq g6')
        self.assertEqual(self.board.en_passant, position.from_coord('G6'))
        pawn1 = self.board.squares[position.from_coord('F5').index].piece
        pawn2 = self.board.squares[position.from_coord('G5').index].piece
        self.assertEqual(
            pawn1.legal_moves(self.board),
            {position.from_coord(c) for c in {'F6', 'G6'}}
        )
        self.board._move(position.from_coord('F5'), position.from_coord('G6'))
        self.assertEqual(self.board.squares[position.from_coord('G6').index].piece, pawn1)
        self.assertFalse(pawn2 in self.board.pieces)
        self.assertEqual(self.board.en_passant, None)

        # Black En Passant
        self.board = Board(state='rnbqkbnr/pppppp1p/8/8/5Pp1/8/PPPPP1PP/RNBQKBNR b KQkq f3 0 1')
        self.assertEqual(self.board.en_passant, position.from_coord('F3'))
        pawn1 = self.board.squares[position.from_coord('G4').index].piece
        pawn2 = self.board.squares[position.from_coord('F4').index].piece
        self.assertEqual(
            pawn1.legal_moves(self.board),
            {position.from_coord(c) for c in {'G3', 'F3'}}
        )
        self.board._move(position.from_coord('G4'), position.from_coord('F3'))
        self.assertEqual(self.board.squares[position.from_coord('F3').index].piece, pawn1)
        self.assertFalse(pawn2 in self.board.pieces)
        self.assertEqual(self.board.en_passant, None)

    def test_pawn_promotion(self):
        fen = 'r1bqkbnr/p1pp3P/1pn1p3/5p2/8/4P3/PPPP2PP/RNBQKBNR w KQkq f6 0 1'
        self.board = Board(fen)
        h7 = position.from_coord('H7')
        g8 = position.from_coord('G8')
        pawn = self.board.squares[h7.index].piece
        self.board._move(h7, g8, simulate=True)  # Dry run
        self.assertEqual(self.board.fen, fen)

        self.board._move(h7, g8)  # Move pawn and promote piece to queen
        self.assertFalse(pawn in self.board.pieces)
        self.assertEqual(self.board.squares[g8.index].piece.type, QUEEN)  # Check the new piece is a queen

    def test_rook(self):
        self.board = Board(state='rnbqkbnr/pppppp1p/8/8/P1R1p3/8/1PPP1PPP/1NBQKBNR')
        rook = self.board.squares[Position(2, 3).index].piece

        self.assertEqual(
            rook.legal_moves(self.board),
            {position.from_coord(c) for c in {'B4', 'C7', 'D4', 'E4', 'C3', 'C6', 'C5'}}
        )

        with self.assertRaises(IllegalMove):  # Cannot move diagonally
            self.board._move(Position(2, 3), Position(3, 4))

        with self.assertRaises(IllegalMove):  # Cannot move beyond opposition piece
            self.board._move(Position(2, 3), Position(5, 3))

        with self.assertRaises(IllegalMove):  # Cannot move to occupied square (of same colour)
            self.board._move(Position(2, 3), Position(0, 3))

        self.board._move(Position(2, 3), Position(1, 3))  # Move cardinally

        # Take a piece of opposite colour
        piece = self.board.squares[Position(4, 3).index].piece
        self.assertEqual('p', piece.code)
        self.board._move(Position(1, 3), Position(4, 3))
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
            self.board._move(Position(2, 3), Position(6, 6))

        with self.assertRaises(IllegalMove):  # Cannot move to occupied square (of same colour)
            self.board._move(Position(2, 3), Position(0, 1))

        with self.assertRaises(IllegalMove):  # Cannot move cardinally
            self.board._move(Position(2, 3), Position(1, 3))

        self.board._move(Position(2, 3), Position(4, 5))  # Move diagonally

        # Take a piece of opposite colour
        piece = self.board.squares[Position(5, 6).index].piece
        self.assertEqual('p', piece.code)
        self.board._move(Position(4, 5), Position(5, 6))
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
            self.board._move(Position(2, 3), Position(6, 7))

        with self.assertRaises(IllegalMove):  # Cannot move to occupied square (of same colour)
            self.board._move(Position(2, 3), Position(1, 2))

        with self.assertRaises(IllegalMove):  # Can only move cardinally and diagonally
            self.board._move(Position(2, 3), Position(6, 6))

        self.board._move(Position(2, 3), Position(3, 3))  # Move cardinally
        self.board._move(Position(3, 3), Position(6, 6))  # Move diagonally

        # Take a piece of opposite colour
        piece = self.board.squares[Position(7, 7).index].piece
        self.assertEqual('r', piece.code)
        self.board._move(Position(6, 6), Position(7, 7))
        self.assertFalse(piece in self.board.pieces)
        self.assertEqual(Position(7, 7), queen.pos)

    def test_knight(self):
        self.board = Board(state='r1bqkbnr/pppp1p1p/B3p3/8/3Q1Np1/1Pn1P3/1PPP1PPP/RNB1K2R')
        knight = self.board.squares[Position(5, 3).index].piece

        self.assertEqual(
            knight.legal_moves(self.board),
            {position.from_coord(c) for c in {'E2', 'D3', 'H3', 'D5', 'H5', 'E6', 'G6'}}
        )

        with self.assertRaises(IllegalMove):  # Cannot move cardinally
            self.board._move(Position(5, 3), Position(5, 4))

        with self.assertRaises(IllegalMove):  # Cannot move diagonally
            self.board._move(Position(5, 3), Position(6, 4))

        with self.assertRaises(IllegalMove):  # Cannot move to occupied square (of same colour)
            self.board._move(Position(5, 3), Position(6, 1))

        self.board._move(Position(5, 3), Position(4, 1))  # Move L-shape

        # Take a piece of opposite colour
        piece = self.board.squares[Position(2, 2).index].piece
        self.assertEqual('n', piece.code)
        self.board._move(Position(4, 1), Position(2, 2))
        self.assertFalse(piece in self.board.pieces)
        self.assertEqual(Position(2, 2), knight.pos)

    def test_king(self):
        self.board = Board(state='rnb1kbnr/pppp1ppp/4p3/7q/8/BP3P2/P1PPP1PP/RN1QKBNR')
        king = self.board.squares[Position(4, 7).index].piece

        self.assertEqual(
            king.legal_moves(self.board),
            {position.from_coord(c) for c in {'D8', 'E7'}}
        )

        with self.assertRaises(IllegalMove):  # Cannot move if it puts player in check
            self.board._move(Position(4, 7), Position(4, 6))

        self.board._move(Position(4, 7), Position(3, 7))
        self.assertEqual(Position(3, 7), king.pos)

    def test_castle_flags(self):
        b = Board(state=board.STARTING_STATE)
        for piece in filter(lambda p: p.TYPE in (ROOK, KING), b.pieces):
            self.assertTrue(piece.can_castle)

        # Show castling is unavailable after moving pieces
        b._move(Position(0, 6), Position(0, 5))
        b._move(Position(0, 7), Position(0, 6))
        b._move(Position(4, 6), Position(4, 5))
        b._move(Position(4, 7), Position(4, 6))

        for piece in filter(lambda p: p.TYPE in (ROOK, KING), b.pieces):
            if str(piece.original_pos) in ('A8', 'E8'):
                self.assertFalse(piece.can_castle)
            else:
                self.assertTrue(piece.can_castle)

    def test_castling(self):
        # Black Queenside castling
        self.board = Board(state='r3kbnr/pp2pppp/5q2/2pp4/1n4b1/2K5/PPPPPPPP/RNBQ1BNR')
        king = self.board.squares[position.from_coord('E8').index].piece
        rook = self.board.squares[position.from_coord('A8').index].piece
        self.assertTrue(king.can_castle)
        self.assertEqual(
            king.legal_moves(self.board),
            {position.from_coord(c) for c in {'C8', 'D8', 'D7'}}
        )
        self.board._move(position.from_coord('E8'), position.from_coord('C8'))
        self.assertEqual(self.board.squares[position.from_coord('C8').index].piece, king)
        self.assertEqual(self.board.squares[position.from_coord('D8').index].piece, rook)

        self.board = Board(state='r3kbnr/pp2pppp/5q2/2pp4/1n6/5b1B/PPPPPP1P/RNBQK1NR')
        with self.assertRaises(IllegalMove):  # Cannot castle if it puts player in check
            self.board._move(position.from_coord('E8'), position.from_coord('C8'))

        self.board = Board(state='r3kbnr/pp2pppp/2N2q2/2pp4/1n6/5b1B/PPPPPP1P/R1BQK1NR')
        with self.assertRaises(IllegalMove):  # Cannot castle if intermediate square is attacked
            self.board._move(position.from_coord('E8'), position.from_coord('C8'))

        # White Queenside castling
        self.board = Board(state='rnbqkbnr/pppppppp/8/8/1P4Q1/B1N1P3/P1PP1PPP/R3KBNR')
        king = self.board.squares[position.from_coord('E1').index].piece
        rook = self.board.squares[position.from_coord('A1').index].piece
        self.board._move(position.from_coord('E1'), position.from_coord('C1'))
        self.assertEqual(self.board.squares[position.from_coord('C1').index].piece, king)
        self.assertEqual(self.board.squares[position.from_coord('D1').index].piece, rook)

        # Black Kingside castling
        self.board = Board(state='rnbqk2r/pppp1ppp/4p2n/2b5/8/8/PPPPPPPP/RNBQKBNR')
        king = self.board.squares[position.from_coord('E8').index].piece
        rook = self.board.squares[position.from_coord('H8').index].piece
        self.board._move(position.from_coord('E8'), position.from_coord('G8'))
        self.assertEqual(self.board.squares[position.from_coord('G8').index].piece, king)
        self.assertEqual(self.board.squares[position.from_coord('F8').index].piece, rook)

        # White Kingside castling
        self.board = Board(state='rnbqkbnr/pppppppp/8/8/8/3BPN2/PPPP1PPP/RNBQK2R')
        king = self.board.squares[position.from_coord('E1').index].piece
        rook = self.board.squares[position.from_coord('H1').index].piece
        self.board._move(position.from_coord('E1'), position.from_coord('G1'))
        self.assertEqual(self.board.squares[position.from_coord('G1').index].piece, king)
        self.assertEqual(self.board.squares[position.from_coord('F1').index].piece, rook)

def main():
    unittest.main()


if __name__ == '__main__':
    main()
