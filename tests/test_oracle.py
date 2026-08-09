"""
The measuring instrument, measured.

`ai.oracle` is what every learned player in this project is graded by, so it has to be checked
against something other than itself. Two things do that here: the solver is compared with the
independent one in tests/tictactoe/test_perfect_play.py, which shares nothing with it, and the
benchmark is calibrated on players whose score is known in advance.

That calibration is the important half. A benchmark that cannot give a known-perfect player 100%
is not measuring what it claims to, and a benchmark scoring everything highly would hide exactly
the failures it exists to find.
"""

import random
import unittest

from ai.oracle import (
    Grade,
    benchmark,
    enumerate_positions,
    move_values,
    openings_at,
    optimal_moves,
    play_every_line,
    solve,
)
from ai.search import alpha_beta, random_move
from games.tictactoe.board import TicTacToe
from games.tictactoe.constants import CELLS
from tests.tictactoe.corpus import DRAWN_GAME, ROW_WIN
from tests.tictactoe.test_perfect_play import solve as independent_solve
from games.tictactoe.constants import CROSS, NOUGHT

# Every position in tic-tac-toe, terminal ones included. The published figure.
POSITIONS = 5478

# Those of them where somebody still has a move to make, and so can be graded.
DECISIONS = 4520


class TestSolver(unittest.TestCase):
    def test_the_empty_board_is_drawn(self):
        self.assertEqual(0, solve(TicTacToe()))

    def test_it_agrees_with_the_independent_solver_everywhere(self):
        """
        The check that matters. tests/tictactoe/test_perfect_play.py has its own memoised minimax
        written over bare integers, sharing nothing with this one but the rules of the game - and
        two solvers that agree in all 5,478 positions are not both wrong in the same way.
        """
        for state in enumerate_positions(TicTacToe):
            state, _ = state
            mine = solve(state)
            theirs = independent_solve(state.marks[CROSS], state.marks[NOUGHT], state.turn)
            if not state.turn:
                theirs = -theirs  # Theirs speaks for Crosses; mine speaks for the player to move
            self.assertEqual(theirs, mine, str(state))

    def test_a_won_position_is_a_loss_for_the_player_to_move(self):
        """Crosses have the top row, so it is Noughts to move and Noughts have lost."""
        self.assertEqual(-1, solve(TicTacToe(ROW_WIN)))

    def test_a_full_board_is_drawn(self):
        self.assertEqual(0, solve(TicTacToe(DRAWN_GAME)))

    def test_every_opening_move_holds_the_draw(self):
        """
        Tic-tac-toe is drawn and no first move loses it, so all nine are optimal. Worth pinning:
        it is why a benchmark's ply-0 column reads 100% even for a random player.
        """
        self.assertEqual(CELLS, len(optimal_moves(TicTacToe())))

    def test_a_winning_move_is_the_only_optimal_one(self):
        state = TicTacToe([0, 3, 1, 4])  # Crosses have two of the top row, cell 2 finishes it
        self.assertEqual([2], optimal_moves(state))

    def test_move_values_covers_every_legal_move(self):
        state = TicTacToe([4, 0])
        self.assertEqual(sorted(state.legal_moves), sorted(move_values(state)))


class TestEnumeration(unittest.TestCase):
    def test_it_finds_every_position_once(self):
        positions = list(enumerate_positions(TicTacToe))
        self.assertEqual(POSITIONS, len(positions))

    def test_positions_are_deduplicated(self):
        """
        A position reachable four ways is one position, not four. Without this the benchmark
        would weight positions by how many move orders reach them, which is a property of the
        game tree rather than of the player being graded.
        """
        seen = [state.signature for state, _ in enumerate_positions(TicTacToe)]
        self.assertEqual(len(seen), len(set(seen)))

    def test_plies_run_from_zero_to_a_full_board(self):
        plies = {ply for _, ply in enumerate_positions(TicTacToe)}
        self.assertEqual(set(range(CELLS + 1)), plies)


class TestOpeningsAtAPly(unittest.TestCase):
    """
    Depth-limited enumeration, which `enumerate_positions` cannot stand in for: that one walks the
    whole tree, and Connect 4's is 4.5e12 positions.

    The counts below are properties of the games rather than of this code, which is what makes
    them worth asserting. They are also load-bearing in two places - the corpus's enumerated tier
    is built from them, and `ai.ladder` needs enough distinct positions to give every pair of a
    match its own opening.
    """

    # Distinct still-running positions at each ply, derived independently by playing every move
    # order out in a scratch script and deduplicating on the board.
    TIC_TAC_TOE = [1, 9, 72, 252, 756, 1140, 1372]
    CONNECT_FOUR = [1, 7, 49, 238, 1120, 4263, 16422]

    def test_the_counts_are_what_the_games_say_they_are(self):
        for ply, expected in enumerate(self.TIC_TAC_TOE):
            self.assertEqual(expected, len(openings_at(TicTacToe, ply)), f'ply {ply}')

    def test_it_works_for_a_game_far_too_big_to_enumerate_whole(self):
        from games.connect4.board import Connect4
        for ply, expected in enumerate(self.CONNECT_FOUR):
            self.assertEqual(expected, len(openings_at(Connect4, ply)), f'ply {ply}')

    def test_every_position_is_at_the_ply_asked_for(self):
        for state in openings_at(TicTacToe, 4):
            self.assertEqual(4, sum(bin(marks).count('1') for marks in state.marks.values()))

    def test_no_position_is_repeated(self):
        keys = [state.solver_key for state in openings_at(TicTacToe, 5)]
        self.assertEqual(len(keys), len(set(keys)))

    def test_finished_games_are_left_out(self):
        """
        An opening is somewhere a game can continue from. Tic-tac-toe can be won by ply 5, so this
        is not hypothetical - and a finished position handed to a match would score without a move
        being played.
        """
        for ply in range(len(self.TIC_TAC_TOE)):
            for state in openings_at(TicTacToe, ply):
                self.assertFalse(state.is_game_over, f'{state} at ply {ply}')

    def test_the_states_are_independent_of_each_other(self):
        """Copies, not one state rewound - a caller keeps them all at once and plays from each."""
        first, second = openings_at(TicTacToe, 1)[:2]
        before = second.signature
        first.make_move(next(iter(first.legal_moves)))
        self.assertEqual(before, second.signature)

    def test_the_empty_board_is_the_only_opening_at_ply_zero(self):
        self.assertEqual(1, len(openings_at(TicTacToe, 0)))
        self.assertEqual(TicTacToe().signature, openings_at(TicTacToe, 0)[0].signature)

    def test_a_negative_ply_is_refused(self):
        with self.assertRaises(ValueError):
            openings_at(TicTacToe, -1)


class TestBenchmarkCalibration(unittest.TestCase):
    """
    A benchmark is only worth what its calibration is worth. These pin both ends of the scale.
    """

    def test_a_perfect_player_scores_a_hundred_percent(self):
        """
        `alpha_beta` at SOLVED_DEPTH is proved perfect by tests/tictactoe/test_perfect_play.py,
        so anything short of 100% here is the benchmark being wrong, not the player.
        """
        report = benchmark(
            lambda s: alpha_beta(s, depth=TicTacToe.SOLVED_DEPTH),
            enumerate_positions(TicTacToe),
        )

        self.assertEqual(1.0, report.overall.rate)
        self.assertEqual(0, report.overall.blunders)
        self.assertEqual(DECISIONS, report.overall.positions)
        self.assertEqual([], report.worst)

    def test_a_perfect_player_scores_a_hundred_percent_from_both_seats(self):
        """
        The split that exists because an average hides it. A player can be flawless as the first
        player and hopeless as the second - which is what a previous learned player in this
        project actually was - and only the per-seat rates say so.
        """
        report = benchmark(
            lambda s: alpha_beta(s, depth=TicTacToe.SOLVED_DEPTH),
            enumerate_positions(TicTacToe),
        )

        self.assertEqual(1.0, report.by_seat[True].rate)
        self.assertEqual(1.0, report.by_seat[False].rate)
        self.assertEqual(DECISIONS, sum(g.positions for g in report.by_seat.values()))

    def test_a_shallow_search_scores_well_but_not_perfectly(self):
        """Depth 2 cannot see a fork coming, so it must land between random and perfect."""
        report = benchmark(lambda s: alpha_beta(s, depth=2), enumerate_positions(TicTacToe))
        self.assertLess(report.overall.rate, 1.0)
        self.assertGreater(report.overall.rate, 0.8)
        self.assertGreater(report.overall.blunders, 0)

    def test_a_random_player_scores_poorly(self):
        random.seed(0)
        report = benchmark(random_move, enumerate_positions(TicTacToe))

        self.assertLess(report.overall.rate, 0.8)
        self.assertGreater(report.overall.blunders, 100)
        self.assertTrue(report.worst, 'a blundering player should have examples recorded')

    def test_every_ply_is_graded(self):
        report = benchmark(lambda s: alpha_beta(s, depth=2), enumerate_positions(TicTacToe))
        self.assertEqual(set(range(CELLS)), set(report.by_ply))

    def test_an_illegal_move_is_refused_rather_than_scored(self):
        """A player returning nonsense should fail loudly, not quietly score zero for it."""
        with self.assertRaises(ValueError):
            benchmark(lambda s: 'not a move', enumerate_positions(TicTacToe))

    def test_a_value_function_is_graded_separately(self):
        """The true value scored against itself must have no error at all."""
        report = benchmark(
            lambda s: alpha_beta(s, depth=TicTacToe.SOLVED_DEPTH),
            enumerate_positions(TicTacToe),
            value_fn=lambda state: float(solve(state)),
        )
        self.assertAlmostEqual(0.0, report.value_error)


class TestPlayEveryLine(unittest.TestCase):
    """
    The other question about a player, and the one that decides whether it is worth playing.

    `benchmark` asks whether a player knows the whole game. This asks whether the game can be won
    against it, which is not the same question and can give a very different answer: a player can
    be wrong in a hundred positions and still be unbeatable, because it never walks into them.
    """

    def test_a_perfect_player_cannot_be_beaten_from_either_seat(self):
        perfect = lambda state: alpha_beta(state, depth=TicTacToe.SOLVED_DEPTH)

        for seat in (CROSS, NOUGHT):
            record = play_every_line(perfect, TicTacToe, seat)
            self.assertTrue(record.unbeaten, f'{record} from seat {seat}')
            self.assertEqual(0, record.losses)

    def test_a_perfect_player_only_draws_against_perfect_play(self):
        """Tic-tac-toe is drawn, so best against best is a draw down every line."""
        perfect = lambda state: alpha_beta(state, depth=TicTacToe.SOLVED_DEPTH)

        for seat in (CROSS, NOUGHT):
            record = play_every_line(perfect, TicTacToe, seat, opponent=optimal_moves)
            self.assertEqual(0, record.wins)
            self.assertEqual(0, record.losses)
            self.assertEqual(record.lines, record.draws)

    def test_a_weak_player_is_beaten(self):
        """A one-ply search cannot see a fork, so lines exist that beat it."""
        record = play_every_line(lambda s: alpha_beta(s, depth=1), TicTacToe, NOUGHT)
        self.assertFalse(record.unbeaten)
        self.assertGreater(record.losses, 0)

    def test_it_beats_a_weak_player_that_a_benchmark_would_still_score_highly(self):
        """
        The two measures coming apart, which is the reason both exist. Depth 1 scores respectably
        against the solver and is still losable to - and a player can equally be unbeatable while
        scoring poorly, by never reaching the positions it would get wrong.
        """
        graded = benchmark(lambda s: alpha_beta(s, depth=1), enumerate_positions(TicTacToe))
        played = play_every_line(lambda s: alpha_beta(s, depth=1), TicTacToe, NOUGHT)

        self.assertGreater(graded.overall.rate, 0.5)
        self.assertGreater(played.losses, 0)

    def test_every_line_is_accounted_for(self):
        record = play_every_line(lambda s: alpha_beta(s, depth=9), TicTacToe, CROSS)
        self.assertEqual(record.lines, record.wins + record.draws + record.losses)
        self.assertGreater(record.lines, 0)


class TestGrade(unittest.TestCase):
    def test_rates_are_reported_as_fractions(self):
        grade = Grade(positions=10, optimal=7, value_lost=5, blunders=3)
        self.assertAlmostEqual(0.7, grade.rate)
        self.assertAlmostEqual(0.5, grade.mean_value_lost)

    def test_an_empty_grade_does_not_divide_by_zero(self):
        grade = Grade(positions=0, optimal=0, value_lost=0, blunders=0)
        self.assertEqual(0.0, grade.rate)
        self.assertEqual(0.0, grade.mean_value_lost)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
