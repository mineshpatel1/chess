"""
Cover for the game loop and for `result`, which is what tells the loop to stop.

The search deliberately never asks a state whether the game is over - it asks the two cheap
halves of the question separately, at every node. So `result` is exercised by nothing else,
and it is what every game loop and every new game will lean on.
"""

import unittest

from ai.search import alpha_beta, random_move
from ai.simulate import simulate_game
from games.base import DRAW
from games.chess.board import Board
from games.chess.constants import WHITE, BLACK


class TestResult(unittest.TestCase):
    """`result` must agree with the chess-specific properties it is derived from."""

    def test_an_unfinished_game_has_no_result(self):
        self.assertIsNone(Board().result)
        self.assertFalse(Board().is_game_over)

    def test_checkmate_names_the_winner(self):
        board = Board('rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 0 1')
        self.assertTrue(board.is_checkmate)
        self.assertEqual(board.result.winner, BLACK)  # White is mated, so Black won
        self.assertTrue(board.is_game_over)

    def test_stalemate_is_a_draw(self):
        board = Board('7k/5Q2/6K1/8/8/8/8/8 b - - 0 1')
        self.assertTrue(board.is_stalemate)
        self.assertEqual(board.result, DRAW)

    def test_insufficient_material_is_a_draw(self):
        """A draw chess knows about that the move list cannot show: both sides can still move."""
        board = Board('7k/8/8/8/8/8/8/6BK w - - 0 1')
        self.assertTrue(any(board.legal_moves))
        self.assertTrue(board.has_insufficient_material)
        self.assertEqual(board.result, DRAW)

    def test_fifty_move_rule_is_a_draw(self):
        board = Board('4k3/8/8/8/8/8/8/R3K3 w - - 100 60')
        self.assertTrue(any(board.legal_moves))
        self.assertEqual(board.result, DRAW)

    def test_is_game_over_cannot_disagree_with_result(self):
        for fen in (
            Board().fen,
            'rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 0 1',
            '7k/5Q2/6K1/8/8/8/8/8 b - - 0 1',
            '7k/8/8/8/8/8/8/6BK w - - 0 1',
            '4k3/8/8/8/8/8/8/R3K3 w - - 100 60',
        ):
            board = Board(fen)
            self.assertEqual(board.is_game_over, board.result is not None, fen)


class TestSimulateGame(unittest.TestCase):
    def test_plays_a_won_position_out_to_the_win(self):
        """Ra8 is mate in one, so a searching White finishes it on the first move."""
        board = Board('6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1')
        outcome = simulate_game(
            board, lambda s: alpha_beta(s, 2), random_move, print_summary=False
        )

        self.assertEqual(outcome.winner, WHITE)
        self.assertTrue(board.is_checkmate)
        self.assertEqual(board.pgn_uci.strip(), '1. a1a8')

    def test_stops_on_a_drawn_position(self):
        """The loop must end on a draw it can see, not play on until someone runs out of moves."""
        board = Board('7k/8/8/8/8/8/8/6BK w - - 0 1')
        outcome = simulate_game(board, random_move, random_move, print_summary=False)
        self.assertEqual(outcome, DRAW)

    def test_the_second_player_moves_second(self):
        """A move chooser must only ever be handed positions belonging to its own player."""
        seen = {'first': [], 'second': []}

        def watcher(label):
            def choose(state):
                seen[label].append(state.turn)
                return random_move(state)
            return choose

        board = Board('7k/5Q2/6K1/8/8/8/8/8 w - - 0 1')
        simulate_game(board, watcher('first'), watcher('second'), print_summary=False)

        self.assertTrue(all(turn is WHITE for turn in seen['first']), seen['first'])
        self.assertTrue(all(turn is BLACK for turn in seen['second']), seen['second'])


def main():
    unittest.main()


if __name__ == '__main__':
    main()
