"""
The proof that alpha-beta pruning did not change the answer.

Alpha-beta returns the *identical* result to plain negamax. It only skips branches that
provably cannot affect the value at the root, so a version that picks a different move is not a
faster search, it is a broken one - classically a sign error in the `-beta, -alpha` swap, which
produces a search that is still plausible, still fast, and quietly worse.

So the reference is written out here: an unpruned negamax, twenty lines, kept in the test suite
rather than in ai/. That is deliberate and follows what this repo already did once - the min/max
pair was deleted and its differential harness kept (see tests/chess/test_search_equivalence.py),
on the grounds that a second search shipped alongside the first is a second search to maintain.

The comparison is score for score, not merely best move. Two searches can agree on which move is
best and disagree about everything else, and the disagreement is the interesting part: it says
the pruning is unsound in positions where the best move happens to survive it anyway.

Measured from the empty board, leaves reached:

    depth   negamax   alpha-beta   alpha-beta      saving
                      centre-first left-to-right
    4         2,401           97          735       24.8x
    5        16,807          391        3,350       43.0x
    6       117,649          685       13,160      171.8x
    7       810,504        3,128       53,290      259.1x

Two things in that table. The saving compounds with depth, because pruned subtrees contain
pruned subtrees. And it is *ordering* that produces it: the same alpha-beta over the same
positions, differing only in which column `legal_moves` offers first, is between seven and
seventeen times worse. Alpha-beta with the moves in the wrong order is barely alpha-beta.

(At depth 7 the unpruned count is 810,504 rather than 7^7 because a game won at ply 7 has no
continuations - see tests/connect4/test_permutations.py.)
"""

import unittest
from typing import Callable, List, Tuple

from ai.search import MATE, _root_move_score, alpha_beta, terminal_score
from games.base import GameState
from games.connect4.bitboard import bit_count, drops
from games.connect4.board import Connect4
from games.connect4.constants import COLS, COLUMN_MASKS, RED, YELLOW
from games.connect4.evaluation import weighted_eval
from tests.connect4.corpus import positions

CORPUS_SIZE = 12
DEPTHS = (1, 2, 3, 4)


def negamax(state: GameState, depth: int, evaluate: Callable) -> int:
    """
    Negamax with no pruning at all: the answer alpha-beta has to reproduce.

    Mirrors `ai.search._negamax_ab` line for line with the window taken out, including asking
    about terminality before the horizon - a Connect 4 win at the last ply is a win, not a
    position to be evaluated - and including the depth term in the terminal score.
    """
    outcome = state.outcome
    if outcome is not None:
        return terminal_score(outcome, state.turn, depth)

    if depth == 0:
        return evaluate(state)

    best = None
    for move in state.legal_moves:
        state.make_move(move)
        score = -negamax(state, depth - 1, evaluate)
        state.unmake_move()
        if best is None or score > best:
            best = score

    if best is None:  # Nothing to play, so the board is full
        return terminal_score(state.outcome_without_moves, state.turn, depth)
    return best


def counted(evaluate: Callable) -> Tuple[Callable, List[int]]:
    """
    The evaluation, wrapped to record how often the search reached the horizon.

    Counting leaves this way rather than instrumenting `ai/search.py` keeps the measurement
    out of the thing being measured, and both searches call `evaluate` in the same place, so
    the two numbers mean the same thing.
    """
    calls = [0]

    def counting(state: GameState) -> int:
        calls[0] += 1
        return evaluate(state)

    return counting, calls


def root_scores(columns: List[int], depth: int, search: Callable) -> List[Tuple[int, int]]:
    """
    Every root move and the score `search` gives it, in generation order.

    `alpha_beta` scores each root move on a full window, independently of its siblings, so it
    returns the exact minimax value of each and not merely a bound - which is what makes an
    element-for-element comparison against unpruned negamax meaningful rather than optimistic.
    """
    scores = []
    for move in list(Connect4(columns).legal_moves):
        board = Connect4(columns)
        board.make_move(move)
        scores.append((move, -search(board, depth - 1, weighted_eval)))
    return scores


def zero_eval(state: GameState) -> int:
    return 0


def centre_weighted(state: Connect4) -> int:
    """
    Discs weighted by how central their column is, read from the side to move.

    Defined here rather than imported because measuring move ordering needs an evaluation that
    tells positions apart, and with a flat one it cannot: if every leaf scores the same, every
    order of moves is as good as every other and the comparison below reads 391 against 391.
    Keeping it local also stops the measurement moving when the real evaluation is tuned.
    """
    total = 0
    for column in range(COLS):
        weight = COLS // 2 - abs(column - COLS // 2)
        total += weight * bit_count(state.discs[YELLOW] & COLUMN_MASKS[column])
        total -= weight * bit_count(state.discs[RED] & COLUMN_MASKS[column])
    return total if state.turn else -total


class LeftToRight(Connect4):
    """
    Connect 4 with the move ordering taken away, for measuring what the ordering is worth.

    Alpha-beta's entire benefit is ordering: best move first takes the cost from about 7^d to
    7^(d/2), and worst first gains nothing whatever. The only way to say what centre-first is
    buying is to have something to compare it against.
    """

    @property
    def legal_moves(self):
        landing = drops(self.occupied)
        return (column for column in range(COLS) if landing & COLUMN_MASKS[column])


class TestEquivalence(unittest.TestCase):
    def test_pruning_gives_the_same_score_to_every_root_move(self):
        """
        The whole point. Not "the same best move" - every move, at every depth, exactly.
        """
        for columns in positions(CORPUS_SIZE):
            for depth in DEPTHS:
                self.assertEqual(
                    root_scores(columns, depth, negamax),
                    root_scores(columns, depth, _search_ab),
                    f'{columns} at depth {depth}',
                )

    def test_the_two_searches_pick_the_same_move(self):
        for columns in positions(CORPUS_SIZE, seed=1):
            for depth in DEPTHS:
                unpruned = _best_of(root_scores(columns, depth, negamax))
                self.assertEqual(
                    unpruned,
                    alpha_beta(Connect4(columns), depth=depth),
                    f'{columns} at depth {depth}',
                )

    def test_the_search_is_deterministic(self):
        """A comparison against a search that varied run to run would prove nothing."""
        for columns in positions(6, seed=2):
            self.assertEqual(
                root_scores(columns, 3, _search_ab), root_scores(columns, 3, _search_ab)
            )

    def test_ties_go_to_the_move_generated_first(self):
        """
        Which for Connect 4 is the centre column, because generation is centre-first. With
        nothing to choose between the moves the search takes the best opening move anyway.
        """
        self.assertEqual(COLS // 2, alpha_beta(Connect4(), depth=2, evaluate=zero_eval))


class TestPruningPays(unittest.TestCase):
    """
    That the pruning is doing something, and that the move ordering is what makes it pay.

    Asserted loosely on purpose: the exact numbers belong in the module docstring, where they
    can be read, rather than in an assertion that has to be edited whenever the evaluation
    changes what the search prefers.
    """

    def _leaves(self, state: GameState, depth: int, search: Callable) -> int:
        evaluate, calls = counted(centre_weighted)
        search(state, depth, evaluate)
        return calls[0]

    def test_alpha_beta_reaches_far_fewer_leaves_than_plain_negamax(self):
        for depth in (4, 5):
            unpruned = self._leaves(Connect4(), depth, negamax)
            pruned = self._leaves(Connect4(), depth, _search_ab)

            self.assertEqual(COLS ** depth, unpruned, 'the reference should search everything')
            self.assertLess(pruned * 10, unpruned, f'depth {depth}: {pruned} vs {unpruned}')

    def test_centre_first_ordering_beats_left_to_right(self):
        """
        The measurement stage 6 exists for, and the reason the ordering lives in `legal_moves`
        rather than in `ai/`. Same search, same evaluation, same position - the only difference
        is which column the game offers first.
        """
        for depth in (5, 6):
            centre = self._leaves(Connect4(), depth, _search_ab)
            left = self._leaves(LeftToRight(), depth, _search_ab)
            self.assertLess(centre * 5, left, f'depth {depth}: {centre} vs {left}')


class TestTerminalScores(unittest.TestCase):
    """
    Scoring at and beside a finish, which is where a sign convention goes wrong.

    The depth term is what stops the engine dawdling in a won position: a win now has to be
    worth more than the same win three moves later, or it will happily take the scenic route
    and give the opponent three chances to escape.
    """

    def test_a_win_in_one_scores_as_a_win(self):
        scores = dict(root_scores([1, 0, 2, 0, 3, 0], 2, _search_ab))
        self.assertEqual(MATE + 1, scores[4], 'column 4 completes the line')
        self.assertTrue(all(value < MATE for column, value in scores.items() if column != 4))

    def test_a_closer_win_outscores_a_further_one(self):
        """
        Both positions are won for Yellow and both are searched to the same depth. The only
        difference is how long the win takes, and that has to show up in the score - otherwise
        the engine has no reason to finish a game it has already won.

        The slower one is a double threat: Yellow plays to make a three open at both ends, Red
        can only block one end, and Yellow takes the other. Three plies rather than one.
        """
        immediate = _search_ab(Connect4([1, 0, 2, 0, 3, 0]), 5, weighted_eval)
        double_threat = _search_ab(Connect4.from_diagram('''
            .......
            .......
            .......
            .......
            ......R
            ..YY..R
        '''), 5, weighted_eval)

        self.assertEqual(MATE + 4, immediate, 'won on the first of five plies')
        self.assertEqual(MATE + 2, double_threat, 'won on the third of five plies')
        self.assertGreater(immediate, double_threat)

    def test_a_loss_is_worth_the_same_from_either_side(self):
        """
        Negamax reads every score from the point of view of whoever is to move, so a position
        one player calls a win the other must call a loss of exactly the same size.
        """
        board = Connect4([1, 0, 2, 0, 3, 0])
        winning = negamax(board, 2, weighted_eval)

        board.make_move(4)
        self.assertEqual(-winning, negamax(board, 1, weighted_eval))

    def test_a_full_board_is_a_draw_and_worth_nothing(self):
        from tests.connect4.corpus import DRAWN_GAME

        board = Connect4(DRAWN_GAME[:-1])
        board.make_move(DRAWN_GAME[-1])
        self.assertEqual(0, negamax(board, 3, weighted_eval))


def _search_ab(state: GameState, depth: int, evaluate: Callable) -> int:
    """`ai.search`'s alpha-beta on a full window, in the same shape as `negamax` above."""
    from ai.search import HIGH_BOUND, LOW_BOUND, _negamax_ab

    return _negamax_ab(state, depth, LOW_BOUND, HIGH_BOUND, evaluate)


def _best_of(scores: List[Tuple[int, int]]) -> int:
    """The move `alpha_beta` would pick from these: highest score, ties to the first."""
    best_move, best_score = scores[0]
    for move, score in scores[1:]:
        if score > best_score:
            best_move, best_score = move, score
    return best_move


def main():
    unittest.main()


if __name__ == '__main__':
    main()
