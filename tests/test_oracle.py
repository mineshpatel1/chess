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

from ai.oracle import Grade, benchmark, enumerate_positions, move_values, optimal_moves, solve
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


class TestBenchmarkCalibration(unittest.TestCase):
    """
    A benchmark is only worth what its calibration is worth. These pin both ends of the scale.
    """

    def test_a_perfect_player_scores_a_hundred_percent(self):
        """
        `alpha_beta` at SOLVED_DEPTH is proved perfect by tests/tictactoe/test_perfect_play.py,
        so anything short of 100% here is the benchmark being wrong, not the player.
        """
        report = benchmark(lambda s: alpha_beta(s, depth=TicTacToe.SOLVED_DEPTH), TicTacToe)

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
        report = benchmark(lambda s: alpha_beta(s, depth=TicTacToe.SOLVED_DEPTH), TicTacToe)

        self.assertEqual(1.0, report.by_seat[True].rate)
        self.assertEqual(1.0, report.by_seat[False].rate)
        self.assertEqual(DECISIONS, sum(g.positions for g in report.by_seat.values()))

    def test_a_shallow_search_scores_well_but_not_perfectly(self):
        """Depth 2 cannot see a fork coming, so it must land between random and perfect."""
        report = benchmark(lambda s: alpha_beta(s, depth=2), TicTacToe)
        self.assertLess(report.overall.rate, 1.0)
        self.assertGreater(report.overall.rate, 0.8)
        self.assertGreater(report.overall.blunders, 0)

    def test_a_random_player_scores_poorly(self):
        random.seed(0)
        report = benchmark(random_move, TicTacToe)

        self.assertLess(report.overall.rate, 0.8)
        self.assertGreater(report.overall.blunders, 100)
        self.assertTrue(report.worst, 'a blundering player should have examples recorded')

    def test_every_ply_is_graded(self):
        report = benchmark(lambda s: alpha_beta(s, depth=2), TicTacToe)
        self.assertEqual(set(range(CELLS)), set(report.by_ply))

    def test_an_illegal_move_is_refused_rather_than_scored(self):
        """A player returning nonsense should fail loudly, not quietly score zero for it."""
        with self.assertRaises(ValueError):
            benchmark(lambda s: 'not a move', TicTacToe)

    def test_a_value_function_is_graded_separately(self):
        """The true value scored against itself must have no error at all."""
        report = benchmark(
            lambda s: alpha_beta(s, depth=TicTacToe.SOLVED_DEPTH),
            TicTacToe,
            value_fn=lambda state: float(solve(state)),
        )
        self.assertAlmostEqual(0.0, report.value_error)


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
