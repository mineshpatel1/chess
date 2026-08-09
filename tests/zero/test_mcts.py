"""
The search, graded on its own.

This is the file whose absence let a previous attempt at a learned player in this project stay
broken for months. The search there was tested only *through* the network, so a search that could
not converge looked like a network that had not learned yet, and every fix was aimed at the wrong
component.

The trick is to take learning out of the picture entirely: hand the search a **perfect evaluator**
built from `ai.oracle` and require perfect play, then hand it an evaluator that knows *nothing*
and require it to improve with more simulations. The first says the search uses knowledge
correctly; the second says the search generates knowledge on its own. Together they pin the
search without a single trained weight.

None of this needs PyTorch. `ai.zero.mcts` takes its evaluator as an argument, so the component
that failed in 2021 is testable with nothing but the standard library.
"""

import random
import unittest

from ai.oracle import Table, benchmark, enumerate_positions, move_values, optimal_moves, solve
from ai.zero.mcts import MCTS, Node, drive, terminal_value
from games.tictactoe.board import TicTacToe
from games.tictactoe.encoding import TicTacToeEncoder
from tests.tictactoe.corpus import DRAWN_GAME, NOUGHT_WIN, ROW_WIN

TABLE = Table()


def perfect_evaluator(state):
    """The oracle wearing a network's interface: exact priors, exact value."""
    values = move_values(state, TABLE)
    best = max(values.values()) if values else 0
    winners = [move for move, value in values.items() if value == best]

    priors = [0.0] * TicTacToeEncoder.POLICY_SIZE
    for move in winners:
        priors[TicTacToeEncoder.action_index(move)] = 1.0 / len(winners)
    return priors, float(solve(state, TABLE))


def ignorant_evaluator(state):
    """Flat priors and no opinion at all: whatever this search does, the tree did it."""
    return [1.0] * TicTacToeEncoder.POLICY_SIZE, 0.0


class TestTerminalValue(unittest.TestCase):
    def test_a_lost_position_is_worst_for_the_player_to_move(self):
        """Crosses took the top row, so Noughts are to move and Noughts have lost."""
        self.assertEqual(-1.0, terminal_value(TicTacToe(ROW_WIN)))

    def test_it_speaks_for_whoever_is_on_move(self):
        """The same sign for the other player, which is what makes one convention enough."""
        self.assertEqual(-1.0, terminal_value(TicTacToe(NOUGHT_WIN)))

    def test_a_draw_is_zero(self):
        self.assertEqual(0.0, terminal_value(TicTacToe(DRAWN_GAME)))

    def test_an_unfinished_position_has_no_terminal_value(self):
        with self.assertRaises(ValueError):
            terminal_value(TicTacToe([4]))


class TestSearchWithPerfectKnowledge(unittest.TestCase):
    """
    Given the answer, the search must not lose it. Any failure here is the search mishandling
    knowledge it was handed - a sign error, a bad backup, a broken selection rule.
    """

    def test_it_plays_perfectly_at_every_simulation_count(self):
        for simulations in (1, 2, 10, 50):
            search = MCTS(perfect_evaluator, TicTacToeEncoder, simulations=simulations)
            report = benchmark(lambda s: search.search(s).move, enumerate_positions(TicTacToe))

            self.assertEqual(1.0, report.overall.rate, f'{simulations} simulations')

    def test_it_plays_perfectly_from_both_seats(self):
        """The split the 2021 player failed. An average over seats would have hidden it."""
        search = MCTS(perfect_evaluator, TicTacToeEncoder, simulations=25)
        report = benchmark(lambda s: search.search(s).move, enumerate_positions(TicTacToe))

        self.assertEqual(1.0, report.by_seat[True].rate)
        self.assertEqual(1.0, report.by_seat[False].rate)

    def test_a_single_simulation_still_follows_the_prior(self):
        """
        A regression. Expanding the root did not count as a visit, so `sqrt(N_parent)` was zero
        at the first selection, the exploration term vanished for every child, and the search
        took the first move generated no matter what the evaluator said. A one-simulation search
        was blind, and a network could not have fixed it.
        """
        search = MCTS(perfect_evaluator, TicTacToeEncoder, simulations=1)
        for state in (TicTacToe([0, 3, 1, 4]), TicTacToe([4, 0, 8])):
            self.assertIn(search.search(state).move, optimal_moves(state), str(state))

    def test_the_root_value_reports_the_position(self):
        """A drawn game reads as drawn, a lost one as lost."""
        self.assertAlmostEqual(
            0.0, MCTS(perfect_evaluator, TicTacToeEncoder, simulations=30).search(TicTacToe()).value,
            places=6,
        )


class TestSearchWithNoKnowledge(unittest.TestCase):
    """
    Given nothing, the search must find it out. This is the half that the 2021 implementation
    failed outright - it got *worse* with more simulations, because its tree corrupted itself.
    """

    def test_more_simulations_play_better(self):
        rates = []
        for simulations in (10, 50, 200):
            search = MCTS(ignorant_evaluator, TicTacToeEncoder, simulations=simulations,
                          rng=random.Random(0))
            rates.append(
                benchmark(lambda s: search.search(s).move,
                          enumerate_positions(TicTacToe)).overall.rate
            )

        self.assertEqual(rates, sorted(rates), f'search got worse with more thinking: {rates}')
        self.assertGreater(rates[-1], 0.98, 'pure search should nearly solve tic-tac-toe')

    def test_it_finds_a_win_in_one(self):
        state = TicTacToe([0, 3, 1, 4])  # Crosses complete the top row with cell 2
        search = MCTS(ignorant_evaluator, TicTacToeEncoder, simulations=100)
        self.assertEqual(2, search.search(state).move)

    def test_it_blocks_a_loss_in_one(self):
        state = TicTacToe([0, 3, 1])  # Noughts must take cell 2 or lose immediately
        search = MCTS(ignorant_evaluator, TicTacToeEncoder, simulations=100)
        self.assertEqual(2, search.search(state).move)


class TestTheTreeIsPaths(unittest.TestCase):
    """
    The structural fault of 2021, tested directly rather than through its symptoms.

    That version kept one dict keyed by position, so two routes to the same board shared a node -
    and re-expanding it reset its statistics and re-pointed its parent. MCTS needs a tree of
    paths; a tree of positions is a different data structure with the same picture.
    """

    def _tree(self, simulations=200):
        state = TicTacToe()
        search = MCTS(ignorant_evaluator, TicTacToeEncoder, simulations=simulations,
                      rng=random.Random(1))
        # Hand-rolled rather than via `search()`, because these tests inspect the tree and a
        # Result deliberately does not carry one. `drive` runs any of the search's generators to
        # completion, which is all `_expand` and `_simulate` now need to be usable one at a time.
        root = Node(prior=1.0)
        root.value_sum = drive(search._expand(state, root), ignorant_evaluator)
        root.visits = 1
        for _ in range(simulations):
            drive(search._simulate(state, root), ignorant_evaluator)
        return state, root

    def _walk(self, state, node, seen):
        """Maps each position reached to the set of distinct nodes representing it."""
        seen.setdefault(state.signature, set()).add(id(node))
        for move, child in node.children.items():
            state.make_move(move)
            self._walk(state, child, seen)
            state.unmake_move()

    def test_one_position_reached_two_ways_is_two_nodes(self):
        state, root = self._tree()
        seen = {}
        self._walk(state, root, seen)

        shared = {sig: nodes for sig, nodes in seen.items() if len(nodes) > 1}
        self.assertTrue(
            shared,
            'no position was reached by two paths, so this test proved nothing - '
            'tic-tac-toe transposes constantly and the tree should show it',
        )

    def test_every_visit_is_accounted_for_exactly_once(self):
        """
        The invariant a path-tree has and a position-keyed one cannot: a node is visited once when
        it is expanded, and every visit after that descends into exactly one child. So

            node.visits == 1 + sum(child.visits)

        at every expanded node in the tree. A node shared by two paths breaks this immediately -
        it collects visits from parents that are not its own, and the arithmetic stops adding up.
        This is the 2021 fault stated as a number rather than as a symptom.
        """
        state, root = self._tree()
        self._check_visits(root)

    def _check_visits(self, node):
        if not node.expanded:
            return
        children = sum(child.visits for child in node.children.values())
        self.assertEqual(
            node.visits, children + 1,
            f'node has {node.visits} visits but its children account for {children}',
        )
        for child in node.children.values():
            self._check_visits(child)

    def test_the_root_visit_count_matches_the_simulations_run(self):
        """One simulation is one visit at the root, plus the expansion that seeded it."""
        simulations = 50
        _, root = self._tree(simulations)
        self.assertEqual(simulations + 1, root.visits)


class TestPolicyOutput(unittest.TestCase):
    def test_the_policy_is_a_distribution_over_legal_moves_only(self):
        state = TicTacToe([4, 0])
        search = MCTS(ignorant_evaluator, TicTacToeEncoder, simulations=50)
        policy = search.search(state).policy

        self.assertAlmostEqual(1.0, sum(policy), places=6)
        legal = {TicTacToeEncoder.action_index(m) for m in state.legal_moves}
        for action, weight in enumerate(policy):
            if action not in legal:
                self.assertEqual(0.0, weight, f'action {action} is not legal here')

    def test_sampling_at_zero_temperature_is_the_best_move(self):
        search = MCTS(ignorant_evaluator, TicTacToeEncoder, simulations=50)
        result = search.search(TicTacToe())
        self.assertEqual(result.move, search.sample(result.visits, temperature=0.0))

    def test_sampling_at_temperature_stays_legal(self):
        state = TicTacToe([4, 0])
        search = MCTS(ignorant_evaluator, TicTacToeEncoder, simulations=50,
                      rng=random.Random(3))
        result = search.search(state)
        for _ in range(20):
            self.assertIn(search.sample(result.visits, temperature=1.0), list(state.legal_moves))


def main():
    unittest.main()


if __name__ == '__main__':
    main()
