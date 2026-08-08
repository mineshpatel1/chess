"""
The training targets: what a self-play game says a position was worth.

Every sign here was wrong in a previous attempt at a learned player in this project, and both
mistakes were invisible - they produce a network that trains happily and plays badly, which is
indistinguishable from one that has not trained enough.

Needs no PyTorch. `ai.zero.selfplay` takes its evaluator as an argument, so the data pipeline can
be checked with a stub in place of a network.
"""

import random
import unittest

from ai.zero.selfplay import (
    Example,
    OPENING_PLIES,
    _opening,
    _value_to,
    augment,
    play_game,
)
from games.base import DRAW
from games.tictactoe.board import TicTacToe
from games.tictactoe.constants import CELLS, CROSS, NOUGHT
from games.tictactoe.encoding import TicTacToeEncoder


def ignorant(state):
    return [1.0] * TicTacToeEncoder.POLICY_SIZE, 0.0


class TestValueTargets(unittest.TestCase):
    """
    The one number a self-play game actually teaches, and the one the 2021 code inverted.
    """

    def test_the_winner_sees_a_win_from_its_own_positions(self):
        self.assertEqual(1.0, _value_to(mover=CROSS, winner=CROSS))
        self.assertEqual(1.0, _value_to(mover=NOUGHT, winner=NOUGHT))

    def test_the_loser_sees_a_loss(self):
        self.assertEqual(-1.0, _value_to(mover=CROSS, winner=NOUGHT))
        self.assertEqual(-1.0, _value_to(mover=NOUGHT, winner=CROSS))

    def test_a_draw_is_zero_for_everyone(self):
        """
        Not a sign. Tic-tac-toe is drawn under any decent play, so most self-play games end level
        and most examples must say so. The 2021 code had no zero case and labelled drawn games
        +1/-1 by turn, which teaches that a dead-drawn position is won for whoever is on move.
        """
        self.assertEqual(0.0, _value_to(mover=CROSS, winner=None))
        self.assertEqual(0.0, _value_to(mover=NOUGHT, winner=None))


class TestPlayGame(unittest.TestCase):
    def test_it_produces_one_example_per_position_played(self):
        examples, finished = play_game(ignorant, TicTacToeEncoder, TicTacToe, simulations=10,
                                       rng=random.Random(0))
        self.assertEqual(len(finished.move_stack), len(examples))
        self.assertTrue(finished.is_game_over)

    def test_values_agree_with_how_the_game_actually_ended(self):
        """
        End to end over many games: every example's value has to be what the finished game was
        worth to the player who was on move in that position. Alternating positions therefore
        alternate in sign for a decisive game, and are all zero for a drawn one.
        """
        rng = random.Random(1)
        for _ in range(25):
            examples, finished = play_game(ignorant, TicTacToeEncoder, TicTacToe,
                                           simulations=10, rng=rng)
            winner = finished.result.winner

            state = TicTacToe()
            for example, move in zip(examples, finished.move_stack):
                expected = 0.0 if winner is None else (1.0 if winner == state.turn else -1.0)
                self.assertEqual(expected, example.value, str(finished))
                state.make_move(move)

    def test_a_drawn_game_labels_everything_zero(self):
        rng = random.Random(0)
        for _ in range(40):
            examples, finished = play_game(ignorant, TicTacToeEncoder, TicTacToe,
                                           simulations=10, rng=rng)
            if finished.result == DRAW:
                self.assertEqual([0.0] * len(examples), [e.value for e in examples])
                return
        self.skipTest('no drawn game came up in this sample')

    def test_policies_are_distributions(self):
        examples, _ = play_game(ignorant, TicTacToeEncoder, TicTacToe, simulations=10,
                                rng=random.Random(2))
        for example in examples:
            self.assertAlmostEqual(1.0, sum(example.policy), places=6)


class TestRandomOpenings(unittest.TestCase):
    """
    Coverage, for a caller that wants it.

    Off by default now: raising `c_puct` reached the same coverage through PUCT and root noise,
    which is where an AlphaZero is supposed to get it. These tests name their own ply count rather
    than reading OPENING_PLIES, so they go on testing the mechanism after the default changed -
    a test that follows a constant is a test that stops checking anything when the constant is
    turned off.
    """

    PLIES = 4

    def test_an_opening_leaves_a_playable_position(self):
        rng = random.Random(0)
        for _ in range(50):
            state = _opening(TicTacToe, self.PLIES, rng)
            self.assertFalse(state.is_game_over, str(state))
            self.assertTrue(list(state.legal_moves))

    def test_zero_plies_is_a_new_game(self):
        self.assertEqual(TicTacToe().signature, _opening(TicTacToe, 0, random.Random()).signature)

    def test_it_is_off_by_default(self):
        """Self-play starts from a new game unless a caller asks otherwise."""
        self.assertEqual(0, OPENING_PLIES)

    def test_openings_land_at_a_range_of_depths(self):
        """Drawn per game rather than fixed, or every opening would pile up at one depth."""
        rng = random.Random(0)
        depths = {len(_opening(TicTacToe, self.PLIES, rng).move_stack) for _ in range(60)}
        self.assertGreater(len(depths), 1, f'openings all landed at the same depth: {depths}')

    def test_nothing_from_the_opening_is_recorded(self):
        """
        The opening moves random, so they carry no search behind them and must not become
        training targets. Only the plies after it are examples.
        """
        rng = random.Random(3)
        examples, finished = play_game(ignorant, TicTacToeEncoder, TicTacToe, simulations=10,
                                       opening_plies=4, rng=rng)
        self.assertLess(len(examples), len(finished.move_stack) + 1)
        self.assertGreater(len(examples), 0)


class TestAugmentation(unittest.TestCase):
    def test_every_example_becomes_eight(self):
        examples = [Example(planes=TicTacToeEncoder.planes(TicTacToe([4, 0])),
                            policy=[0.0] * CELLS, value=0.5)]
        self.assertEqual(8, len(augment(examples, TicTacToeEncoder)))

    def test_the_value_is_carried_through_untouched(self):
        """Rotating a board does not change who is winning."""
        examples = [Example(planes=TicTacToeEncoder.planes(TicTacToe([4, 0])),
                            policy=[0.0] * CELLS, value=-1.0)]
        for grown in augment(examples, TicTacToeEncoder):
            self.assertEqual(-1.0, grown.value)

    def test_policies_stay_distributions(self):
        policy = [1.0 if i == 3 else 0.0 for i in range(CELLS)]
        examples = [Example(planes=TicTacToeEncoder.planes(TicTacToe([4])), policy=policy,
                            value=0.0)]
        for grown in augment(examples, TicTacToeEncoder):
            self.assertAlmostEqual(1.0, sum(grown.policy), places=6)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
