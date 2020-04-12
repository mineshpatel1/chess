import unittest

from engine.game import Game
from engine import position
from engine.constants import STARTING_STATE
from engine.exceptions import Checkmate

class TestUndo(unittest.TestCase):
    def test_basic_undo(self):
        """Runs full game in reverse but there wasn't any castling, promotions or en passant takes."""
        moves = (
            ('F2', 'F3', 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'),
            ('B7', 'B6', 'rnbqkbnr/pppppppp/8/8/8/5P2/PPPPP1PP/RNBQKBNR b KQkq - 0 1'),
            ('C2', 'C4', 'rnbqkbnr/p1pppppp/1p6/8/8/5P2/PPPPP1PP/RNBQKBNR w KQkq - 0 2'),
            ('B6', 'B5', 'rnbqkbnr/p1pppppp/1p6/8/2P5/5P2/PP1PP1PP/RNBQKBNR b KQkq c3 0 2'),
            ('C4', 'B5', 'rnbqkbnr/p1pppppp/8/1p6/2P5/5P2/PP1PP1PP/RNBQKBNR w KQkq - 0 3'),
            ('A7', 'A6', 'rnbqkbnr/p1pppppp/8/1P6/8/5P2/PP1PP1PP/RNBQKBNR b KQkq - 0 3'),
            ('A2', 'A4', 'rnbqkbnr/2pppppp/p7/1P6/8/5P2/PP1PP1PP/RNBQKBNR w KQkq - 0 4'),
            ('A6', 'B5', 'rnbqkbnr/2pppppp/p7/1P6/P7/5P2/1P1PP1PP/RNBQKBNR b KQkq a3 0 4'),
            ('B2', 'B3', 'rnbqkbnr/2pppppp/8/1p6/P7/5P2/1P1PP1PP/RNBQKBNR w KQkq - 0 5'),
            ('A8', 'A4', 'rnbqkbnr/2pppppp/8/1p6/P7/1P3P2/3PP1PP/RNBQKBNR b KQkq - 0 5'),
            ('B3', 'A4', '1nbqkbnr/2pppppp/8/1p6/r7/1P3P2/3PP1PP/RNBQKBNR w KQk - 0 6'),
            ('B5', 'A4', '1nbqkbnr/2pppppp/8/1p6/P7/5P2/3PP1PP/RNBQKBNR b KQk - 0 6'),
            ('A1', 'A4', '1nbqkbnr/2pppppp/8/8/p7/5P2/3PP1PP/RNBQKBNR w KQk - 0 7'),
            ('G8', 'H6', '1nbqkbnr/2pppppp/8/8/R7/5P2/3PP1PP/1NBQKBNR b Kk - 0 7'),
            ('G2', 'G4', '1nbqkb1r/2pppppp/7n/8/R7/5P2/3PP1PP/1NBQKBNR w Kk - 1 8'),
            ('H6', 'G4', '1nbqkb1r/2pppppp/7n/8/R5P1/5P2/3PP2P/1NBQKBNR b Kk g3 0 8'),
            ('F3', 'G4', '1nbqkb1r/2pppppp/8/8/R5n1/5P2/3PP2P/1NBQKBNR w Kk - 0 9'),
            ('D7', 'D5', '1nbqkb1r/2pppppp/8/8/R5P1/8/3PP2P/1NBQKBNR b Kk - 0 9'),
            ('D2', 'D3', '1nbqkb1r/2p1pppp/8/3p4/R5P1/8/3PP2P/1NBQKBNR w Kk d6 0 10'),
            ('C8', 'G4', '1nbqkb1r/2p1pppp/8/3p4/R5P1/3P4/4P2P/1NBQKBNR b Kk - 0 10'),
            ('D1', 'B3', '1n1qkb1r/2p1pppp/8/3p4/R5b1/3P4/4P2P/1NBQKBNR w Kk - 0 11'),
            ('G4', 'E2', '1n1qkb1r/2p1pppp/8/3p4/R5b1/1Q1P4/4P2P/1NB1KBNR b Kk - 1 11'),
            ('G1', 'E2', '1n1qkb1r/2p1pppp/8/3p4/R7/1Q1P4/4b2P/1NB1KBNR w Kk - 0 12'),
            ('D5', 'D4', '1n1qkb1r/2p1pppp/8/3p4/R7/1Q1P4/4N2P/1NB1KB1R b Kk - 0 12'),
            ('E2', 'D4', '1n1qkb1r/2p1pppp/8/8/R2p4/1Q1P4/4N2P/1NB1KB1R w Kk - 0 13'),
            ('D8', 'D4', '1n1qkb1r/2p1pppp/8/8/R2N4/1Q1P4/7P/1NB1KB1R b Kk - 0 13'),
            ('A4', 'D4', '1n2kb1r/2p1pppp/8/8/R2q4/1Q1P4/7P/1NB1KB1R w Kk - 0 14'),
            ('B8', 'C6', '1n2kb1r/2p1pppp/8/8/3R4/1Q1P4/7P/1NB1KB1R b Kk - 0 14'),
            ('D4', 'D5', '4kb1r/2p1pppp/2n5/8/3R4/1Q1P4/7P/1NB1KB1R w Kk - 1 15'),
            ('C6', 'B4', '4kb1r/2p1pppp/2n5/3R4/8/1Q1P4/7P/1NB1KB1R b Kk - 2 15'),
            ('B3', 'B4', '4kb1r/2p1pppp/8/3R4/1n6/1Q1P4/7P/1NB1KB1R w Kk - 3 16'),
            ('C7', 'C6', '4kb1r/2p1pppp/8/3R4/1Q6/3P4/7P/1NB1KB1R b Kk - 0 16'),
            ('D5', 'C5', '4kb1r/4pppp/2p5/3R4/1Q6/3P4/7P/1NB1KB1R w Kk - 0 17'),
            ('F7', 'F5', '4kb1r/4pppp/2p5/2R5/1Q6/3P4/7P/1NB1KB1R b Kk - 1 17'),
            ('C5', 'F5', '4kb1r/4p1pp/2p5/2R2p2/1Q6/3P4/7P/1NB1KB1R w Kk f6 0 18'),
            ('C6', 'C5', '4kb1r/4p1pp/2p5/5R2/1Q6/3P4/7P/1NB1KB1R b Kk - 0 18'),
            ('B4', 'C5', '4kb1r/4p1pp/8/2p2R2/1Q6/3P4/7P/1NB1KB1R w Kk - 0 19'),
            ('H8', 'G8', '4kb1r/4p1pp/8/2Q2R2/8/3P4/7P/1NB1KB1R b Kk - 0 19'),
            ('D3', 'D4', '4kbr1/4p1pp/8/2Q2R2/8/3P4/7P/1NB1KB1R w K - 1 20'),
            ('G8', 'H8', '4kbr1/4p1pp/8/2Q2R2/3P4/8/7P/1NB1KB1R b K - 0 20'),
            ('D4', 'D5', '4kb1r/4p1pp/8/2Q2R2/3P4/8/7P/1NB1KB1R w K - 1 21'),
            ('H8', 'G8', '4kb1r/4p1pp/8/2QP1R2/8/8/7P/1NB1KB1R b K - 0 21'),
            ('C1', 'B2', '4kbr1/4p1pp/8/2QP1R2/8/8/7P/1NB1KB1R w K - 1 22'),
            ('G8', 'H8', '4kbr1/4p1pp/8/2QP1R2/8/8/1B5P/1N2KB1R b K - 2 22'),
            ('F1', 'D3', '4kb1r/4p1pp/8/2QP1R2/8/8/1B5P/1N2KB1R w K - 3 23'),
            ('H8', 'G8', '4kb1r/4p1pp/8/2QP1R2/8/3B4/1B5P/1N2K2R b K - 4 23'),
            ('F5', 'H5', '4kbr1/4p1pp/8/2QP1R2/8/3B4/1B5P/1N2K2R w K - 5 24'),
            ('G8', 'H8', '4kbr1/4p1pp/8/2QP3R/8/3B4/1B5P/1N2K2R b K - 6 24'),
            ('H5', 'H4', '4kb1r/4p1pp/8/2QP3R/8/3B4/1B5P/1N2K2R w K - 7 25'),
            ('H8', 'G8', '4kb1r/4p1pp/8/2QP4/7R/3B4/1B5P/1N2K2R b K - 8 25'),
            ('H4', 'H7', '4kbr1/4p1pp/8/2QP4/7R/3B4/1B5P/1N2K2R w K - 9 26'),
            ('G8', 'H8', '4kbr1/4p1pR/8/2QP4/8/3B4/1B5P/1N2K2R b K - 0 26'),
            ('H7', 'H8', '4kb1r/4p1pR/8/2QP4/8/3B4/1B5P/1N2K2R w K - 1 27'),
            ('E7', 'E6', '4kb1R/4p1p1/8/2QP4/8/3B4/1B5P/1N2K2R b K - 0 27'),
            ('D5', 'E6', '4kb1R/6p1/4p3/2QP4/8/3B4/1B5P/1N2K2R w K - 0 28'),
            ('E8', 'D8', '4kb1R/6p1/4P3/2Q5/8/3B4/1B5P/1N2K2R b K - 0 28'),
            ('H2', 'H4', '3k1b1R/6p1/4P3/2Q5/8/3B4/1B5P/1N2K2R w K - 1 29'),
            ('D8', 'E8', '3k1b1R/6p1/4P3/2Q5/7P/3B4/1B6/1N2K2R b K h3 0 29'),
            ('H4', 'H5', '4kb1R/6p1/4P3/2Q5/7P/3B4/1B6/1N2K2R w K - 1 30'),
            ('E8', 'D8', '4kb1R/6p1/4P3/2Q4P/8/3B4/1B6/1N2K2R b K - 0 30'),
            ('H8', 'F8', '3k1b1R/6p1/4P3/2Q4P/8/3B4/1B6/1N2K2R w K - 1 31'),
        )

        # Play through the game as it happened
        board = Game()

        for start, end, _fen in moves:
            board.player_move(position.from_coord(start), position.from_coord(end))

        # Validate the end state of the game (checkmate)
        self.assertEqual(board.fen, '3k1R2/6p1/4P3/2Q4P/8/3B4/1B6/1N2K2R b K - 0 31')
        with self.assertRaises(Checkmate):
            board.raise_if_game_over()

        # Undo the moves and validate the board states match as the game played out
        i = 0
        fens = [m[2] for m in moves]
        while len(board.move_history) > 0:
            board.undo_move()
            self.assertEqual(board.fen, fens[-1 * (i + 1)])
            i += 1
        self.assertEqual(board.fen, STARTING_STATE)  # Check we have the initial game board

def main():
    unittest.main()


if __name__ == '__main__':
    main()
