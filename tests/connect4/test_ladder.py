"""
The match ladder, and the one arithmetic invariant that pins it down.

A ladder has no oracle - that is why it exists, for the game where the exact answer cannot be had
in bulk - so it cannot be checked by comparing it with a right answer. What it can be checked
against is an *identity* and a calibration.

**The identity: a deterministic player against itself must score exactly 0.5.** Not approximately,
and not on average. Both games of a pair start from the same opening and both are played by the
same function, so they run identically move for move; whoever moves first wins both, or neither
does. Every pair is therefore one win and one loss to the challenger, or two draws. Any other
number means the pairing is broken, or the colours are assigned wrongly, or something is leaking
between games - and none of those would be obvious from a plausible-looking score.

**The calibration**: random must lose to a real search, and a deeper search must beat a shallower
one. Numbers that fail those are measuring the harness rather than the players.

The ladders used here are cut down - three cheap rungs and twenty games - because the suite should
not spend a minute playing depth-6 Connect 4 to learn something twenty games at depth 2 already
say. The default ladder's own shape is asserted separately, without playing anything.
"""

import random
import unittest

from ai.ladder import (
    GAMES,
    LADDERS,
    Ladder,
    Rung,
    Standing,
    balanced_openings,
    climb,
    for_game,
    make,
)
from ai.match import MatchResult, play_match
from ai.oracle import openings_at
from ai.players import player
from games.connect4.board import Connect4
from games.tictactoe.board import TicTacToe

# Cheap enough to run in the suite, and still enough rungs to be a ladder rather than a match.
CHEAP = Ladder(rungs=('random', 'minimax:1', 'minimax:2'), opening_plies=4, balanced=False)
FEW = 20


def beaten(score: float = 0.9) -> MatchResult:
    """A lopsided result, so that it is significant as well as ahead."""
    wins = int(100 * score)
    return MatchResult(wins=wins, draws=0, losses=100 - wins)


def level() -> MatchResult:
    return MatchResult(wins=50, draws=0, losses=50)


class TestADeterministicPlayerAgainstItself(unittest.TestCase):
    """
    The identity the whole harness rests on, stated for both games.

    Exactly 0.5, with no tolerance, because there is no randomness left in it once the openings are
    fixed: this is arithmetic, not a measurement.
    """

    def test_it_scores_exactly_level_on_connect_four(self):
        me = player('minimax:2')
        result = play_match(
            Connect4, me, me, games=FEW, print_summary=False,
            openings=openings_at(Connect4, 4)[:FEW // 2],
        )
        self.assertEqual(0.5, result.score, str(result))
        self.assertEqual(result.wins, result.losses)

    def test_it_scores_exactly_level_on_tic_tac_toe(self):
        me = player('minimax:9')
        result = play_match(
            TicTacToe, me, me, games=FEW, print_summary=False,
            openings=openings_at(TicTacToe, 2)[:FEW // 2],
        )
        self.assertEqual(0.5, result.score, str(result))

    def test_every_rung_of_a_self_ladder_that_is_the_same_player_is_level(self):
        """The same statement through `climb`, which is where the openings are actually chosen."""
        me = player('minimax:2')
        standing = climb(Connect4, me, Ladder(('minimax:2',), 4, balanced=False),
                         games=FEW, print_progress=False)
        self.assertEqual(0.5, standing.rungs[0].result.score)
        self.assertIsNone(standing.highest_beaten)


class TestOpeningsAreNeverReused(unittest.TestCase):
    """
    The reason `play_match` grew an `openings` argument at all.

    Two deterministic players replaying an opening play the identical game, so a repeat adds a
    result to the tally without adding anything to what the tally knows. The default path draws
    with replacement and does repeat; the ladder must not.
    """

    def test_a_match_starts_each_pair_from_a_different_position(self):
        plies = 4
        openings = openings_at(Connect4, plies)[:FEW // 2]
        seen = []

        def recorder(state):
            if len(state.columns_played) == plies:
                seen.append(state.signature)
            return player('minimax:1')(state)

        play_match(Connect4, recorder, player('minimax:2'), games=FEW,
                   print_summary=False, openings=openings)

        self.assertEqual(len(seen), len(set(seen)), 'an opening was played twice')
        self.assertTrue(set(seen).issubset({state.signature for state in openings}))

    def test_a_match_refuses_to_repeat_when_given_too_few(self):
        with self.assertRaises(ValueError):
            play_match(Connect4, player('random'), player('random'), games=FEW,
                       print_summary=False, openings=openings_at(Connect4, 1))

    def test_a_climb_refuses_when_the_game_has_too_few_openings(self):
        """
        Two plies of Connect 4 offer 49 positions, so 200 games cannot be honoured. Better to say
        so than to quietly play 49 openings four times each and report 200 games.
        """
        with self.assertRaises(ValueError):
            climb(Connect4, player('random'), Ladder(('random',), 2, balanced=False),
                  games=200, print_progress=False)

    def test_the_default_ladders_can_honour_the_default_game_count(self):
        """
        Counted the way the ladder counts it, which is after balancing. Tic-tac-toe starts three
        plies in rather than two for exactly this reason: two plies leave 72 openings but only 24
        drawn ones, and a 100-game ladder needs 50.
        """
        for name, ladder in LADDERS.items():
            game = {'Connect4': Connect4, 'TicTacToe': TicTacToe}[name]
            available = openings_at(game, ladder.opening_plies)
            if ladder.balanced:
                available = balanced_openings(game, available)
            self.assertGreaterEqual(
                len(available), GAMES // 2,
                f'{name} starts {ladder.opening_plies} plies in, where only {len(available)} '
                f'usable openings exist, but the default is {GAMES} games',
            )


class TestBalancedOpenings(unittest.TestCase):
    """
    Starting level, which is the difference between a ladder that discriminates and one that does
    not. A pair played from a decided opening is forced to 0.5 as soon as both players convert it,
    so those pairs go quiet exactly when the players get good enough to be worth telling apart.
    """

    def test_a_solvable_game_keeps_only_its_drawn_openings(self):
        from ai.oracle import solve
        drawn = balanced_openings(TicTacToe, openings_at(TicTacToe, 3))
        self.assertTrue(drawn)
        for state in drawn:
            self.assertEqual(0, solve(state), str(state))

    def test_it_keeps_the_drawn_ones_and_no_others(self):
        from ai.oracle import solve
        every = openings_at(TicTacToe, 3)
        self.assertEqual(
            sum(1 for state in every if solve(state) == 0),
            len(balanced_openings(TicTacToe, every)),
        )

    def test_a_game_with_a_corpus_uses_it(self):
        """
        Connect 4 four plies in is far beyond what our own solver reaches, so this can only work by
        looking the positions up in `ai/corpora/connect4.txt` - which is exactly why the corpus
        enumerates the opening in full.
        """
        drawn = balanced_openings(Connect4, openings_at(Connect4, 4))
        self.assertEqual(200, len(drawn), 'the corpus says 200 of the 1,120 are drawn')

    def test_a_game_with_neither_keeps_every_opening(self):
        """Chess can be neither solved nor looked up, and must still get a usable ladder."""
        from games.chess.board import ChessBoard
        every = openings_at(ChessBoard, 1)
        with self.assertLogs('chess', level='WARNING'):  # log.py's logger does not propagate
            self.assertEqual(len(every), len(balanced_openings(ChessBoard, every)))

    def test_a_balanced_climb_still_scores_a_player_against_itself_level(self):
        me = player('minimax:2')
        standing = climb(TicTacToe, me, Ladder(('minimax:2',), 3), games=FEW,
                         print_progress=False)
        self.assertEqual(0.5, standing.rungs[0].result.score)


class TestCalibration(unittest.TestCase):
    """
    Players whose relative strength is not in doubt, so a surprise here is the harness.

    Two climbs for the whole class - the weakest player and a strong one - since playing the
    ladder is the cost and the shape of a standing is the same whoever climbed it.
    """

    @classmethod
    def setUpClass(cls):
        random.seed(0)
        cls.weak = climb(Connect4, player('random'), CHEAP, games=FEW, print_progress=False)
        cls.strong = climb(Connect4, player('minimax:4'), CHEAP, games=FEW, print_progress=False)

    def test_random_loses_to_a_real_search(self):
        """
        Only the rungs that are searches are asserted. The bottom rung is random against random,
        which is level in expectation and genuinely noisy over twenty games - asserting that a
        random player beats nothing would be asserting that a coin never comes up heads twice.
        """
        by_spec = {rung.spec: rung for rung in self.weak.rungs}

        self.assertTrue(by_spec['minimax:1'].lost, str(self.weak))
        self.assertTrue(by_spec['minimax:2'].lost, str(self.weak))

    def test_a_search_beats_random_and_a_shallower_search(self):
        by_spec = {rung.spec: rung for rung in self.strong.rungs}

        self.assertTrue(by_spec['random'].beaten, str(self.strong))
        self.assertTrue(by_spec['minimax:1'].beaten, str(self.strong))

    def test_every_rung_is_played_even_after_a_hopeless_loss(self):
        """Fixed cost and comparable runs; the ladder does not stop where the player falls off."""
        self.assertEqual(len(CHEAP.rungs), len(self.weak.rungs))
        self.assertEqual(list(CHEAP.rungs), [rung.spec for rung in self.weak.rungs])

    def test_every_game_is_accounted_for_on_every_rung(self):
        for rung in self.weak.rungs:
            self.assertEqual(FEW, rung.result.games, rung.spec)


class TestReproducibility(unittest.TestCase):
    """A number that moves between runs is a number you cannot quote."""

    def test_the_same_seed_gives_the_same_standing(self):
        scores = [
            [rung.result.score for rung in
             climb(Connect4, player('minimax:2'), CHEAP, games=FEW, seed=7,
                   print_progress=False).rungs]
            for _ in range(2)
        ]
        self.assertEqual(scores[0], scores[1])

    def test_a_different_seed_chooses_different_openings(self):
        def first_opening(seed):
            standing_openings = openings_at(Connect4, CHEAP.opening_plies)
            import random as _random
            shuffled = list(standing_openings)
            _random.Random(seed).shuffle(shuffled)
            return shuffled[0].signature

        self.assertNotEqual(first_opening(0), first_opening(1))


class TestTheSummary(unittest.TestCase):
    """
    Read off synthetic results, so this needs no games and cannot be slow or flaky.

    The interesting case is the non-monotonic one. Connect 4 has genuine odd/even depth effects, so
    a player really can beat a deeper opponent while failing to beat a shallower one, and a summary
    that reported only the highest number would make that invisible.
    """

    @staticmethod
    def standing(*pairs) -> Standing:
        return Standing([Rung(spec, result) for spec, result in pairs], 100, 4)

    def test_the_highest_rung_beaten_is_the_last_one_beaten(self):
        report = self.standing(
            ('random', beaten()), ('minimax:1', beaten()), ('minimax:2', level()),
        )
        self.assertEqual('minimax:1', report.highest_beaten)
        self.assertEqual([], report.skipped)

    def test_beating_nothing_is_reported_as_nothing(self):
        report = self.standing(('random', level()), ('minimax:1', beaten(0.1)))
        self.assertIsNone(report.highest_beaten)
        self.assertEqual([], report.skipped)
        self.assertIn('none', str(report))

    def test_a_rung_missed_below_the_highest_one_beaten_is_named(self):
        report = self.standing(
            ('random', beaten()), ('minimax:1', level()),
            ('minimax:2', beaten()), ('minimax:3', level()),
        )
        self.assertEqual('minimax:2', report.highest_beaten)
        self.assertEqual(['minimax:1'], report.skipped)
        self.assertIn('not a clean ladder', str(report))

    def test_being_ahead_is_not_the_same_as_beating(self):
        """0.55 over 100 games is inside the noise, and must not count as a rung cleared."""
        narrow = MatchResult(wins=55, draws=0, losses=45)
        self.assertFalse(narrow.is_significant)
        self.assertIsNone(self.standing(('random', narrow)).highest_beaten)

    def test_the_verdicts_read_the_way_the_scores_do(self):
        report = self.standing(
            ('a', beaten()), ('b', level()), ('c', beaten(0.05)),
        )
        self.assertEqual(['beats', 'level', 'loses'], [rung.verdict for rung in report.rungs])


class TestLadderConfiguration(unittest.TestCase):
    def test_both_games_with_a_ladder_have_one(self):
        self.assertEqual(for_game(Connect4), LADDERS['Connect4'])
        self.assertEqual(for_game(TicTacToe), LADDERS['TicTacToe'])

    def test_a_game_without_a_ladder_says_so(self):
        from games.chess.board import ChessBoard
        with self.assertRaises(SystemExit):
            for_game(ChessBoard)

    def test_the_rungs_get_harder(self):
        """
        Not an assertion about play, which would need games - only that the sequence is ordered as
        written. A ladder whose rungs were shuffled would still run and would still report.
        """
        for ladder in LADDERS.values():
            depths = [int(spec.split(':')[1]) for spec in ladder.rungs if spec.startswith('mini')]
            self.assertEqual(sorted(depths), depths)
            self.assertEqual('random', ladder.rungs[0], 'the bottom rung should be the weakest')

    def test_a_ladder_can_be_overridden(self):
        custom = make(['random', 'minimax:3'], opening_plies=2)
        self.assertEqual(('random', 'minimax:3'), custom.rungs)
        self.assertEqual(2, custom.opening_plies)


if __name__ == '__main__':
    unittest.main()
