"""
The match harness, which is the only thing that can say whether an evaluation is any good.

Everything here is a smoke test. The matches that actually justify a weight take minutes and
hundreds of games, and belong at a prompt rather than in a suite - what belongs here is that the
harness counts correctly, pairs the openings, and reports an error bar that means something,
because a tally that is quietly wrong would send the tuning off in the wrong direction and look
like a result while it did it.
"""

import logging
import unittest

import log
from ai.match import MatchResult, play_match
from ai.search import alpha_beta, random_move
from games.connect4.board import Connect4


def always(column: int):
    """A player that drops into one column while it can. Predictable, and beatable."""
    return lambda state: column if column in state.legal_moves else next(iter(state.legal_moves))


class TestMatchResult(unittest.TestCase):
    def test_the_score_counts_a_draw_as_half(self):
        self.assertEqual(0.5, MatchResult(0, 10, 0).score)
        self.assertEqual(1.0, MatchResult(10, 0, 0).score)
        self.assertEqual(0.0, MatchResult(0, 0, 10).score)
        self.assertEqual(0.75, MatchResult(5, 5, 0).score)

    def test_an_all_draw_match_has_no_error(self):
        """Every game the same result is no spread, whatever the sample size."""
        self.assertEqual(0.0, MatchResult(0, 100, 0).error)

    def test_a_level_result_is_never_significant(self):
        self.assertFalse(MatchResult(25, 50, 25).is_significant)

    def test_a_thrashing_is_significant(self):
        self.assertTrue(MatchResult(90, 5, 5).is_significant)

    def test_a_narrow_lead_over_few_games_is_not_significant(self):
        """
        The trap the error bar exists for. 27-23 looks like a result and is not one; the same
        margin repeated over enough games is.
        """
        self.assertFalse(MatchResult(27, 0, 23).is_significant)
        self.assertTrue(MatchResult(2700, 0, 2300).is_significant)

    def test_an_empty_match_does_not_divide_by_zero(self):
        self.assertEqual(0.0, MatchResult(0, 0, 0).score)
        self.assertEqual(0.0, MatchResult(0, 0, 0).error)


class TestPlayMatch(unittest.TestCase):
    def setUp(self):
        log.setLevel(logging.ERROR)

    def tearDown(self):
        log.setLevel(logging.INFO)

    def test_every_game_is_accounted_for(self):
        result = play_match(Connect4, random_move, random_move, games=10, seed=0)
        self.assertEqual(10, result.games)

    def test_an_odd_number_of_games_is_rounded_down_to_a_pair(self):
        """Openings are played twice with the sides swapped, so games come in twos."""
        self.assertEqual(10, play_match(Connect4, random_move, random_move, games=11).games)

    def test_a_better_player_wins(self):
        """
        The end-to-end check on the whole apparatus. If a depth-2 search cannot beat a player
        that drops everything down one column, the tally is being counted from the wrong side.
        """
        result = play_match(
            Connect4, lambda state: alpha_beta(state, depth=2), always(0), games=20, seed=1
        )
        self.assertGreater(result.score, 0.5, str(result))
        self.assertTrue(result.is_significant, str(result))

    def test_a_worse_player_loses_by_the_same_margin(self):
        """
        The same match with the arguments swapped has to be the mirror image of itself. A
        harness that favours whoever it was handed first would make every challenger look good.
        """
        strong = lambda state: alpha_beta(state, depth=2)
        forward = play_match(Connect4, strong, always(0), games=20, seed=2)
        backward = play_match(Connect4, always(0), strong, games=20, seed=2)

        self.assertEqual(forward.wins, backward.losses)
        self.assertEqual(forward.losses, backward.wins)
        self.assertEqual(forward.draws, backward.draws)

    def test_a_player_against_itself_is_level(self):
        """
        Two identical deterministic searches, so every pair is one game and its mirror. Any
        imbalance would be the harness giving one seat an advantage - which for a game the
        first player wins outright is exactly what pairing the openings is there to remove.
        """
        strong = lambda state: alpha_beta(state, depth=2)
        result = play_match(Connect4, strong, strong, games=20, seed=3)

        self.assertEqual(result.wins, result.losses, str(result))
        self.assertEqual(0.5, result.score)

    def test_the_openings_differ_from_each_other(self):
        """Without random openings two deterministic searches play one game N times."""
        first = play_match(Connect4, random_move, random_move, games=40, seed=4)
        self.assertGreater(first.wins + first.losses, 0, 'no game was decided at all')

    def test_the_result_is_reproducible(self):
        """
        Same seed, same openings, same result - given players that are themselves
        deterministic. `random_move` is not: it draws on the global random module rather than
        on the match's generator, so a match between two of those repeats its openings and not
        its games. The seed is a promise about the openings only.
        """
        strong = lambda state: alpha_beta(state, depth=1)
        self.assertEqual(
            play_match(Connect4, strong, always(3), games=20, seed=5),
            play_match(Connect4, strong, always(3), games=20, seed=5),
        )

    def test_a_different_seed_gives_different_openings(self):
        strong = lambda state: alpha_beta(state, depth=1)
        self.assertNotEqual(
            play_match(Connect4, strong, always(3), games=20, seed=6),
            play_match(Connect4, strong, always(3), games=20, seed=7),
        )


def main():
    unittest.main()


if __name__ == '__main__':
    main()
