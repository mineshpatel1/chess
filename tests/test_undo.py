import unittest

from engine.constants import STARTING_STATE
from engine.board import Board, Move


class TestUndo(unittest.TestCase):
    def test_basic_undo(self):
        """Runs full game in reverse but there wasn't any castling, promotions or en passant takes."""
        moves = (
            ('F2F3', 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'),
            ('B7B6', 'rnbqkbnr/pppppppp/8/8/8/5P2/PPPPP1PP/RNBQKBNR b KQkq - 0 1'),
            ('C2C4', 'rnbqkbnr/p1pppppp/1p6/8/8/5P2/PPPPP1PP/RNBQKBNR w KQkq - 0 2'),
            ('B6B5', 'rnbqkbnr/p1pppppp/1p6/8/2P5/5P2/PP1PP1PP/RNBQKBNR b KQkq c3 0 2'),
            ('C4B5', 'rnbqkbnr/p1pppppp/8/1p6/2P5/5P2/PP1PP1PP/RNBQKBNR w KQkq - 0 3'),
            ('A7A6', 'rnbqkbnr/p1pppppp/8/1P6/8/5P2/PP1PP1PP/RNBQKBNR b KQkq - 0 3'),
            ('A2A4', 'rnbqkbnr/2pppppp/p7/1P6/8/5P2/PP1PP1PP/RNBQKBNR w KQkq - 0 4'),
            ('A6B5', 'rnbqkbnr/2pppppp/p7/1P6/P7/5P2/1P1PP1PP/RNBQKBNR b KQkq a3 0 4'),
            ('B2B3', 'rnbqkbnr/2pppppp/8/1p6/P7/5P2/1P1PP1PP/RNBQKBNR w KQkq - 0 5'),
            ('A8A4', 'rnbqkbnr/2pppppp/8/1p6/P7/1P3P2/3PP1PP/RNBQKBNR b KQkq - 0 5'),
            ('B3A4', '1nbqkbnr/2pppppp/8/1p6/r7/1P3P2/3PP1PP/RNBQKBNR w KQk - 0 6'),
            ('B5A4', '1nbqkbnr/2pppppp/8/1p6/P7/5P2/3PP1PP/RNBQKBNR b KQk - 0 6'),
            ('A1A4', '1nbqkbnr/2pppppp/8/8/p7/5P2/3PP1PP/RNBQKBNR w KQk - 0 7'),
            ('G8H6', '1nbqkbnr/2pppppp/8/8/R7/5P2/3PP1PP/1NBQKBNR b Kk - 0 7'),
            ('G2G4', '1nbqkb1r/2pppppp/7n/8/R7/5P2/3PP1PP/1NBQKBNR w Kk - 1 8'),
            ('H6G4', '1nbqkb1r/2pppppp/7n/8/R5P1/5P2/3PP2P/1NBQKBNR b Kk g3 0 8'),
            ('F3G4', '1nbqkb1r/2pppppp/8/8/R5n1/5P2/3PP2P/1NBQKBNR w Kk - 0 9'),
            ('D7D5', '1nbqkb1r/2pppppp/8/8/R5P1/8/3PP2P/1NBQKBNR b Kk - 0 9'),
            ('D2D3', '1nbqkb1r/2p1pppp/8/3p4/R5P1/8/3PP2P/1NBQKBNR w Kk d6 0 10'),
            ('C8G4', '1nbqkb1r/2p1pppp/8/3p4/R5P1/3P4/4P2P/1NBQKBNR b Kk - 0 10'),
            ('D1B3', '1n1qkb1r/2p1pppp/8/3p4/R5b1/3P4/4P2P/1NBQKBNR w Kk - 0 11'),
            ('G4E2', '1n1qkb1r/2p1pppp/8/3p4/R5b1/1Q1P4/4P2P/1NB1KBNR b Kk - 1 11'),
            ('G1E2', '1n1qkb1r/2p1pppp/8/3p4/R7/1Q1P4/4b2P/1NB1KBNR w Kk - 0 12'),
            ('D5D4', '1n1qkb1r/2p1pppp/8/3p4/R7/1Q1P4/4N2P/1NB1KB1R b Kk - 0 12'),
            ('E2D4', '1n1qkb1r/2p1pppp/8/8/R2p4/1Q1P4/4N2P/1NB1KB1R w Kk - 0 13'),
            ('D8D4', '1n1qkb1r/2p1pppp/8/8/R2N4/1Q1P4/7P/1NB1KB1R b Kk - 0 13'),
            ('A4D4', '1n2kb1r/2p1pppp/8/8/R2q4/1Q1P4/7P/1NB1KB1R w Kk - 0 14'),
            ('B8C6', '1n2kb1r/2p1pppp/8/8/3R4/1Q1P4/7P/1NB1KB1R b Kk - 0 14'),
            ('D4D5', '4kb1r/2p1pppp/2n5/8/3R4/1Q1P4/7P/1NB1KB1R w Kk - 1 15'),
            ('C6B4', '4kb1r/2p1pppp/2n5/3R4/8/1Q1P4/7P/1NB1KB1R b Kk - 2 15'),
            ('B3B4', '4kb1r/2p1pppp/8/3R4/1n6/1Q1P4/7P/1NB1KB1R w Kk - 3 16'),
            ('C7C6', '4kb1r/2p1pppp/8/3R4/1Q6/3P4/7P/1NB1KB1R b Kk - 0 16'),
            ('D5C5', '4kb1r/4pppp/2p5/3R4/1Q6/3P4/7P/1NB1KB1R w Kk - 0 17'),
            ('F7F5', '4kb1r/4pppp/2p5/2R5/1Q6/3P4/7P/1NB1KB1R b Kk - 1 17'),
            ('C5F5', '4kb1r/4p1pp/2p5/2R2p2/1Q6/3P4/7P/1NB1KB1R w Kk f6 0 18'),
            ('C6C5', '4kb1r/4p1pp/2p5/5R2/1Q6/3P4/7P/1NB1KB1R b Kk - 0 18'),
            ('B4C5', '4kb1r/4p1pp/8/2p2R2/1Q6/3P4/7P/1NB1KB1R w Kk - 0 19'),
            ('H8G8', '4kb1r/4p1pp/8/2Q2R2/8/3P4/7P/1NB1KB1R b Kk - 0 19'),
            ('D3D4', '4kbr1/4p1pp/8/2Q2R2/8/3P4/7P/1NB1KB1R w K - 1 20'),
            ('G8H8', '4kbr1/4p1pp/8/2Q2R2/3P4/8/7P/1NB1KB1R b K - 0 20'),
            ('D4D5', '4kb1r/4p1pp/8/2Q2R2/3P4/8/7P/1NB1KB1R w K - 1 21'),
            ('H8G8', '4kb1r/4p1pp/8/2QP1R2/8/8/7P/1NB1KB1R b K - 0 21'),
            ('C1B2', '4kbr1/4p1pp/8/2QP1R2/8/8/7P/1NB1KB1R w K - 1 22'),
            ('G8H8', '4kbr1/4p1pp/8/2QP1R2/8/8/1B5P/1N2KB1R b K - 2 22'),
            ('F1D3', '4kb1r/4p1pp/8/2QP1R2/8/8/1B5P/1N2KB1R w K - 3 23'),
            ('H8G8', '4kb1r/4p1pp/8/2QP1R2/8/3B4/1B5P/1N2K2R b K - 4 23'),
            ('F5H5', '4kbr1/4p1pp/8/2QP1R2/8/3B4/1B5P/1N2K2R w K - 5 24'),
            ('G8H8', '4kbr1/4p1pp/8/2QP3R/8/3B4/1B5P/1N2K2R b K - 6 24'),
            ('H5H4', '4kb1r/4p1pp/8/2QP3R/8/3B4/1B5P/1N2K2R w K - 7 25'),
            ('H8G8', '4kb1r/4p1pp/8/2QP4/7R/3B4/1B5P/1N2K2R b K - 8 25'),
            ('H4H7', '4kbr1/4p1pp/8/2QP4/7R/3B4/1B5P/1N2K2R w K - 9 26'),
            ('G8H8', '4kbr1/4p1pR/8/2QP4/8/3B4/1B5P/1N2K2R b K - 0 26'),
            ('H7H8', '4kb1r/4p1pR/8/2QP4/8/3B4/1B5P/1N2K2R w K - 1 27'),
            ('E7E6', '4kb1R/4p1p1/8/2QP4/8/3B4/1B5P/1N2K2R b K - 0 27'),
            ('D5E6', '4kb1R/6p1/4p3/2QP4/8/3B4/1B5P/1N2K2R w K - 0 28'),
            ('E8D8', '4kb1R/6p1/4P3/2Q5/8/3B4/1B5P/1N2K2R b K - 0 28'),
            ('H2H4', '3k1b1R/6p1/4P3/2Q5/8/3B4/1B5P/1N2K2R w K - 1 29'),
            ('D8E8', '3k1b1R/6p1/4P3/2Q5/7P/3B4/1B6/1N2K2R b K h3 0 29'),
            ('H4H5', '4kb1R/6p1/4P3/2Q5/7P/3B4/1B6/1N2K2R w K - 1 30'),
            ('E8D8', '4kb1R/6p1/4P3/2Q4P/8/3B4/1B6/1N2K2R b K - 0 30'),
            ('H8F8', '3k1b1R/6p1/4P3/2Q4P/8/3B4/1B6/1N2K2R w K - 1 31'),
        )

        # Play through the game as it happened
        board = Board()

        for uci, _fen in moves:
            m = Move.from_uci(uci)
            self.assertTrue(m in board.legal_moves)
            self.assertEqual(board.fen, _fen)
            board.make_move(m)

        # Validate the end state of the game (checkmate)
        self.assertEqual(board.fen, '3k1R2/6p1/4P3/2Q4P/8/3B4/1B6/1N2K2R b K - 0 31')
        self.assertTrue(board.is_checkmate)

        # Undo the moves and validate the board states match as the game played out
        i = 0
        fens = [m[1] for m in moves]
        while len(board._history) > 0:
            board.unmake_move()
            self.assertEqual(board.fen, fens[-1 * (i + 1)])
            i += 1
        self.assertEqual(board.fen, STARTING_STATE)  # Check we have the initial game board

    def test_complex(self):
        moves = (
            ('G1F3', 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'),
            ('G7G5', 'rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1'),
            ('F3D4', 'rnbqkbnr/pppppp1p/8/6p1/8/5N2/PPPPPPPP/RNBQKB1R w KQkq g6 0 2'),
            ('G5G4', 'rnbqkbnr/pppppp1p/8/6p1/3N4/8/PPPPPPPP/RNBQKB1R b KQkq - 1 2'),
            ('F2F4', 'rnbqkbnr/pppppp1p/8/8/3N2p1/8/PPPPPPPP/RNBQKB1R w KQkq - 0 3'),
            ('G4F3', 'rnbqkbnr/pppppp1p/8/8/3N1Pp1/8/PPPPP1PP/RNBQKB1R b KQkq f3 0 3'),
            ('E2F3', 'rnbqkbnr/pppppp1p/8/8/3N4/5p2/PPPPP1PP/RNBQKB1R w KQkq - 0 4'),
            ('G8F6', 'rnbqkbnr/pppppp1p/8/8/3N4/5P2/PPPP2PP/RNBQKB1R b KQkq - 0 4'),
            ('F1C4', 'rnbqkb1r/pppppp1p/5n2/8/3N4/5P2/PPPP2PP/RNBQKB1R w KQkq - 1 5'),
            ('F8H6', 'rnbqkb1r/pppppp1p/5n2/8/2BN4/5P2/PPPP2PP/RNBQK2R b KQkq - 2 5'),
            ('E1G1', 'rnbqk2r/pppppp1p/5n1b/8/2BN4/5P2/PPPP2PP/RNBQK2R w KQkq - 3 6'),
            ('B8C6', 'rnbqk2r/pppppp1p/5n1b/8/2BN4/5P2/PPPP2PP/RNBQ1RK1 b kq - 4 6'),
            ('D1E2', 'r1bqk2r/pppppp1p/2n2n1b/8/2BN4/5P2/PPPP2PP/RNBQ1RK1 w kq - 5 7'),
            ('F6E4', 'r1bqk2r/pppppp1p/2n2n1b/8/2BN4/5P2/PPPPQ1PP/RNB2RK1 b kq - 6 7'),
            ('F3E4', 'r1bqk2r/pppppp1p/2n4b/8/2BNn3/5P2/PPPPQ1PP/RNB2RK1 w kq - 7 8'),
            ('H6D2', 'r1bqk2r/pppppp1p/2n4b/8/2BNP3/8/PPPPQ1PP/RNB2RK1 b kq - 0 8'),
            ('C1D2', 'r1bqk2r/pppppp1p/2n5/8/2BNP3/8/PPPbQ1PP/RNB2RK1 w kq - 0 9'),
            ('A7A5', 'r1bqk2r/pppppp1p/2n5/8/2BNP3/8/PPPBQ1PP/RN3RK1 b kq - 0 9'),
            ('G2G4', 'r1bqk2r/1ppppp1p/2n5/p7/2BNP3/8/PPPBQ1PP/RN3RK1 w kq a6 0 10'),
            ('H7H6', 'r1bqk2r/1ppppp1p/2n5/p7/2BNP1P1/8/PPPBQ2P/RN3RK1 b kq g3 0 10'),
            ('G4G5', 'r1bqk2r/1ppppp2/2n4p/p7/2BNP1P1/8/PPPBQ2P/RN3RK1 w kq - 0 11'),
            ('H8H7', 'r1bqk2r/1ppppp2/2n4p/p5P1/2BNP3/8/PPPBQ2P/RN3RK1 b kq - 0 11'),
            ('G5H6', 'r1bqk3/1ppppp1r/2n4p/p5P1/2BNP3/8/PPPBQ2P/RN3RK1 w q - 1 12'),
            ('H7G7', 'r1bqk3/1ppppp1r/2n4P/p7/2BNP3/8/PPPBQ2P/RN3RK1 b q - 0 12'),
            ('G1F2', 'r1bqk3/1pppppr1/2n4P/p7/2BNP3/8/PPPBQ2P/RN3RK1 w q - 1 13'),
            ('G7G4', 'r1bqk3/1pppppr1/2n4P/p7/2BNP3/8/PPPBQK1P/RN3R2 b q - 2 13'),
            ('H6H7', 'r1bqk3/1ppppp2/2n4P/p7/2BNP1r1/8/PPPBQK1P/RN3R2 w q - 3 14'),
            ('E7E6', 'r1bqk3/1ppppp1P/2n5/p7/2BNP1r1/8/PPPBQK1P/RN3R2 b q - 0 14'),
            ('E2G4', 'r1bqk3/1ppp1p1P/2n1p3/p7/2BNP1r1/8/PPPBQK1P/RN3R2 w q - 0 15'),
            ('E8E7', 'r1bqk3/1ppp1p1P/2n1p3/p7/2BNP1Q1/8/PPPB1K1P/RN3R2 b q - 0 15'),
            ('H7H8q', 'r1bq4/1pppkp1P/2n1p3/p7/2BNP1Q1/8/PPPB1K1P/RN3R2 w - - 1 16'),
            ('A5A4', 'r1bq3Q/1pppkp2/2n1p3/p7/2BNP1Q1/8/PPPB1K1P/RN3R2 b - - 0 16'),
            ('H8D8', 'r1bq3Q/1pppkp2/2n1p3/8/p1BNP1Q1/8/PPPB1K1P/RN3R2 w - - 0 17'),
            ('E7D6', 'r1bQ4/1pppkp2/2n1p3/8/p1BNP1Q1/8/PPPB1K1P/RN3R2 b - - 0 17'),
            ('D8C8', 'r1bQ4/1ppp1p2/2nkp3/8/p1BNP1Q1/8/PPPB1K1P/RN3R2 w - - 1 18'),
            ('A8C8', 'r1Q5/1ppp1p2/2nkp3/8/p1BNP1Q1/8/PPPB1K1P/RN3R2 b - - 0 18'),
            ('D4C6', '2r5/1ppp1p2/2nkp3/8/p1BNP1Q1/8/PPPB1K1P/RN3R2 w - - 0 19'),
            ('D6C6', '2r5/1ppp1p2/2Nkp3/8/p1B1P1Q1/8/PPPB1K1P/RN3R2 b - - 0 19'),
            ('C4E6', '2r5/1ppp1p2/2k1p3/8/p1B1P1Q1/8/PPPB1K1P/RN3R2 w - - 0 20'),
            ('F7E6', '2r5/1ppp1p2/2k1B3/8/p3P1Q1/8/PPPB1K1P/RN3R2 b - - 0 20'),
            ('G4G7', '2r5/1ppp4/2k1p3/8/p3P1Q1/8/PPPB1K1P/RN3R2 w - - 0 21'),
            ('A4A3', '2r5/1ppp2Q1/2k1p3/8/p3P3/8/PPPB1K1P/RN3R2 b - - 1 21'),
            ('B2A3', '2r5/1ppp2Q1/2k1p3/8/4P3/p7/PPPB1K1P/RN3R2 w - - 0 22'),
            ('C8A8', '2r5/1ppp2Q1/2k1p3/8/4P3/P7/P1PB1K1P/RN3R2 b - - 0 22'),
            ('B1C3', 'r7/1ppp2Q1/2k1p3/8/4P3/P7/P1PB1K1P/RN3R2 w - - 1 23'),
            ('E6E5', 'r7/1ppp2Q1/2k1p3/8/4P3/P1N5/P1PB1K1P/R4R2 b - - 2 23'),
            ('F1G1', 'r7/1ppp2Q1/2k5/4p3/4P3/P1N5/P1PB1K1P/R4R2 w - - 0 24'),
            ('D7D6', 'r7/1ppp2Q1/2k5/4p3/4P3/P1N5/P1PB1K1P/R5R1 b - - 1 24'),
            ('G1G6', 'r7/1pp3Q1/2kp4/4p3/4P3/P1N5/P1PB1K1P/R5R1 w - - 0 25'),
            ('A8A3', 'r7/1pp3Q1/2kp2R1/4p3/4P3/P1N5/P1PB1K1P/R7 b - - 1 25'),
            ('H2H4', '8/1pp3Q1/2kp2R1/4p3/4P3/r1N5/P1PB1K1P/R7 w - - 0 26'),
            ('B7B5', '8/1pp3Q1/2kp2R1/4p3/4P2P/r1N5/P1PB1K2/R7 b - h3 0 26'),
            ('D2C1', '8/2p3Q1/2kp2R1/1p2p3/4P2P/r1N5/P1PB1K2/R7 w - b6 0 27'),
            ('A3A8', '8/2p3Q1/2kp2R1/1p2p3/4P2P/r1N5/P1P2K2/R1B5 b - - 1 27'),
            ('A1B1', 'r7/2p3Q1/2kp2R1/1p2p3/4P2P/2N5/P1P2K2/R1B5 w - - 2 28'),
            ('A8A2', 'r7/2p3Q1/2kp2R1/1p2p3/4P2P/2N5/P1P2K2/1RB5 b - - 3 28'),
            ('C3B5', '8/2p3Q1/2kp2R1/1p2p3/4P2P/2N5/r1P2K2/1RB5 w - - 0 29'),
            ('A2A1', '8/2p3Q1/2kp2R1/1N2p3/4P2P/8/r1P2K2/1RB5 b - - 0 29'),
            ('B1A1', '8/2p3Q1/2kp2R1/1N2p3/4P2P/8/2P2K2/rRB5 w - - 1 30'),
            ('C6B6', '8/2p3Q1/2kp2R1/1N2p3/4P2P/8/2P2K2/R1B5 b - - 0 30'),
            ('G7C7', '8/2p3Q1/1k1p2R1/1N2p3/4P2P/8/2P2K2/R1B5 w - - 1 31'),
            ('B6B5', '8/2Q5/1k1p2R1/1N2p3/4P2P/8/2P2K2/R1B5 b - - 0 31'),
            ('G6D6', '8/2Q5/3p2R1/1k2p3/4P2P/8/2P2K2/R1B5 w - - 0 32'),
            ('B5B4', '8/2Q5/3R4/1k2p3/4P2P/8/2P2K2/R1B5 b - - 0 32'),
            ('D6B6', '8/2Q5/3R4/4p3/1k2P2P/8/2P2K2/R1B5 w - - 1 33'),
        )

        # Play through the game as it happened
        board = Board()

        for uci, _fen in moves:
            m = Move.from_uci(uci)
            self.assertTrue(m in board.legal_moves)
            self.assertEqual(board.fen, _fen)
            board.make_move(m)

        # Validate the end state of the game (checkmate)
        self.assertEqual(board.fen, '8/2Q5/1R6/4p3/1k2P2P/8/2P2K2/R1B5 b - - 2 33')
        self.assertTrue(board.is_checkmate)

        # Undo the moves and validate the board states match as the game played out
        i = 0
        fens = [m[1] for m in moves]
        while len(board._history) > 0:
            board.unmake_move()
            self.assertEqual(board.fen, fens[-1 * (i + 1)])
            i += 1
        self.assertEqual(board.fen, STARTING_STATE)  # Check we have the initial game board

def main():
    unittest.main()


if __name__ == '__main__':
    main()
