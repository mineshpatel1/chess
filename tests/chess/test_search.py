import io
import unittest
import contextlib

from games.chess.board import Board
from games.chess.constants import WHITE, BLACK
from ai.search import alpha_beta, terminal_score, MATE
from uci.engine import UciEngine


class TestTerminalValue(unittest.TestCase):
    """
    Direct cover for the scoring of positions with no legal moves.

    Negamax has every node speak for whoever is to move in it, so a finished position is
    scored from the point of view of the player facing it. Being mated is the worst thing that
    can happen to the side it happens to, and scores negatively no matter who is searching -
    which is exactly what lets one function serve every node of the tree.
    """

    # White has been mated by Qh4, and it is White to move
    MATED = 'rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 0 1'

    # Black to move with nowhere to go, but not in check
    STALEMATED = '7k/5Q2/6K1/8/8/8/8/8 b - - 0 1'

    def _score(self, board: Board, depth: int) -> int:
        return terminal_score(board.outcome_without_moves, board.turn, depth)

    def test_checkmate_is_scored_against_the_side_to_move(self):
        board = Board(fen=self.MATED)
        self.assertTrue(board.is_checkmate)
        self.assertEqual(board.turn, WHITE)

        # White is the side being mated, and it is White who has to look at this position
        self.assertEqual(self._score(board, 3), -(MATE + 3))

        # The winner is Black, whichever way round the search happens to be
        self.assertEqual(board.outcome_without_moves.winner, BLACK)

    def test_closer_mates_score_higher(self):
        board = Board(fen=self.MATED)

        # Remaining depth is larger nearer the root, so a shallower mate scores further from
        # zero. Being mated is negative, so nearer the root is the *smaller* number
        self.assertLess(self._score(board, 5), self._score(board, 3))

    def test_stalemate_scores_as_a_draw(self):
        board = Board(fen=self.STALEMATED)
        self.assertTrue(board.is_stalemate)

        # A draw is a draw whoever is searching, and at any depth
        self.assertEqual(self._score(board, 3), 0)
        self.assertEqual(self._score(board, 5), 0)
        self.assertIsNone(board.outcome_without_moves.winner)


class TestSearch(unittest.TestCase):
    """Tests of what the AI decides, rather than of what the rules permit."""

    def test_finds_mate_in_one(self):
        fen = '6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1'  # Ra8 is mate
        for depth in (2, 3):
            board = Board(fen=fen)
            move = alpha_beta(board, depth=depth)
            self.assertEqual(move.uci, 'a1a8', f'at depth {depth}')

            board.make_move(move)
            self.assertTrue(board.is_checkmate)

    def test_avoids_forced_mate(self):
        """
        Black threatens Ra1 mate and the King on h1 is boxed in by its own pawns. Only a move
        that makes luft survives it. Scoring mate against `player` rather than against the
        side to move made this branch invisible, and the engine walked into it.
        """
        fen = 'r5k1/5ppp/8/8/8/8/5PPP/7K w - - 0 1'
        for depth in (3, 4):
            move = alpha_beta(Board(fen=fen), depth=depth)
            self.assertIn(move.uci, {'g2g3', 'g2g4', 'h2h3', 'h2h4'}, f'at depth {depth}')

            # And prove the rejected moves really do lose
            board = Board(fen=fen)
            board.make_move(move)
            board.make_move('a8a1')
            self.assertFalse(board.is_checkmate, f'at depth {depth}')

    def test_prefers_the_faster_mate(self):
        """Rc8 is mate at once. Several Queen moves mate a move later and are generated first."""
        move = alpha_beta(Board(fen='7k/Q7/8/8/8/8/8/2R4K w - - 0 1'), depth=4)
        self.assertEqual(move.uci, 'c1c8')

    def test_prefers_a_draw_to_a_losing_position(self):
        """
        White is a Knight and four pawns down. Every Black pawn is blocked and the Knight on
        b8 is pinned to the King by the Rook on h8, so a5a6 takes away the King's last flight
        square and stalemates Black. A draw beats every other move on the board.
        """
        fen = 'kn5R/3p1p2/1P1p1p2/P2p1p2/3p1P2/3p4/3P4/7K w - - 0 1'
        self.assertLess(Board(fen=fen).value, 0)  # White really is losing

        move = alpha_beta(Board(fen=fen), depth=4)
        self.assertEqual(move.uci, 'a5a6')

        board = Board(fen=fen)
        board.make_move(move)
        self.assertTrue(board.is_stalemate)


class TestUciSearchOutput(unittest.TestCase):
    def _go(self, fen: str) -> str:
        engine = UciEngine()
        engine.set_fen(fen)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            engine.get_best_move()
        return output.getvalue().strip()

    def test_go_reports_a_move(self):
        self.assertTrue(self._go('6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1').startswith('bestmove '))

    def test_go_with_no_legal_moves(self):
        """
        Searching a finished game returns no move, which used to reach `move.uci` on None and
        kill the engine. UCI expects `bestmove (none)`.
        """
        self.assertEqual(
            self._go('rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 0 1'),
            'bestmove (none)',
        )
        self.assertEqual(self._go('7k/5Q2/6K1/8/8/8/8/8 b - - 0 1'), 'bestmove (none)')

    def test_unplayable_move_does_not_kill_the_engine(self):
        """
        UCI is the only interface, so a malformed move from the GUI is reported on stderr and
        the engine keeps answering rather than unwinding out of run().
        """
        engine = UciEngine()
        engine.set_fen('rnbqr3/pppp2P1/3k1n1p/2p1p3/3b4/8/PPPPPP1P/RNBQKBNR w KQ - 0 1')

        errors = io.StringIO()
        with contextlib.redirect_stderr(errors):
            engine.play_moves(['g7g8'])  # Promotion with no piece named
        self.assertIn('g7g8', errors.getvalue())

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            engine.get_best_move()
        self.assertTrue(output.getvalue().strip().startswith('bestmove '))

    def test_promotion_through_uci(self):
        """Each promotion piece named over UCI reaches a different position."""
        placements = set()
        for piece in 'qrbn':
            engine = UciEngine()
            engine.set_fen('rnbqr3/pppp2P1/3k1n1p/2p1p3/3b4/8/PPPPPP1P/RNBQKBNR w KQ - 0 1')
            engine.play_moves([f'g7g8{piece}'])
            placements.add(engine.board.fen)
        self.assertEqual(len(placements), 4)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
