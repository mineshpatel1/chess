"""
The Rust alpha-beta against the Python one it copies.

The claim `ai/native.py` rests on is that the two searches are the same search, not merely a
faster one - `ai.search.alpha_beta` has no transposition table and no move ordering beyond
`CENTRE_FIRST`, so a version that returns a different answer has changed the player, not sped it
up. This is where that is checked rather than asserted, in the same shape
`tests/connect4/test_search_equivalence.py` already checks the pruning itself:

* **Every root move, not just the best one.** `root_scores` is compared element for element, so
  agreeing on the move while disagreeing on a sibling's score - which would say the pruning differs
  somewhere that happens not to matter here - cannot slip through.
* **Leaves reached, not only the scores.** Pinned into a fixture alongside the scores and checked
  by `rust/crates/c4-core/tests/alphabeta.rs` without an interpreter - agreeing on every score
  while walking a different number of leaves would mean the two prune differently and only agree
  by coincidence on this corpus.
* **The evaluation on its own.** `value()` is compared over a wide, uncorrelated sample of
  positions, independently of any search - `evaluate()` is the same primitive `root_scores` and
  `best_move` are built from, and testing it directly finds a threat-counting bug that a search
  comparison alone might average away.

Skipped rather than failed when the extension is not built, exactly as `tests/zero/test_fast.py`
skips when it is not. `python3 -m tests.connect4.test_native --write-fixture` regenerates the
pinned answers `rust/crates/c4-core/tests/alphabeta.rs` checks without a Python interpreter.
"""

import unittest

try:
    import zero_rs
    NATIVE = hasattr(zero_rs, 'best_move')
except ImportError:  # pragma: no cover - depends on the environment, not the code
    zero_rs = None
    NATIVE = False

from ai.native import alpha_beta as native_alpha_beta
from ai.native import available as native_available
from ai.search import MATE, _negamax_ab, alpha_beta as python_alpha_beta, LOW_BOUND, HIGH_BOUND
from games.connect4.board import Connect4
from games.connect4.constants import COLS, RED, YELLOW
from games.connect4.evaluation import value, weighted_eval
from games.tictactoe.board import TicTacToe
from tests.connect4.corpus import positions

needs_native = unittest.skipUnless(
    NATIVE, 'the Rust alpha-beta is not built (see rust/README.md)')

DEPTHS = (1, 2, 3, 4, 5, 6)
CORPUS_SIZE = 12


def counted_negamax_ab(board, depth):
    """
    `_negamax_ab` on a full window, counting evaluation calls the way
    `test_search_equivalence.counted` does - kept local so the count is never off by whatever a
    shared helper does differently.
    """
    calls = [0]

    def counting(state):
        calls[0] += 1
        return weighted_eval(state)

    return _negamax_ab(board, depth, LOW_BOUND, HIGH_BOUND, counting), calls[0]


def python_root_scores(columns, depth):
    """
    Every root move and its full-window score, in generation order, plus the leaves it took.

    The shape `ai.search.alpha_beta` actually searches in: a fresh window per root move, which is
    weaker pruning than carrying alpha across siblings and is why the leaf counts here are larger
    than the single-search table in `test_search_equivalence.py`'s docstring.
    """
    scores = []
    leaves = 0
    for move in list(Connect4(columns).legal_moves):
        board = Connect4(columns)
        board.make_move(move)
        score, calls = counted_negamax_ab(board, depth - 1)
        scores.append((move, -score))
        leaves += calls
    return scores, leaves


def rust_root_scores(columns, depth):
    board = Connect4(columns)
    scores = zero_rs.root_scores(board.discs[YELLOW], board.discs[RED], board.turn, depth)
    return [(column, score) for column, score in scores]


@needs_native
class TestTheSearchIsTheSameSearch(unittest.TestCase):
    """The same evaluator, the same window, the same order - so the same tree, not just answer."""

    def test_root_scores_match_move_for_move(self):
        for columns in positions(CORPUS_SIZE):
            for depth in DEPTHS:
                with self.subTest(columns=columns, depth=depth):
                    expected, _ = python_root_scores(columns, depth)
                    self.assertEqual(expected, rust_root_scores(columns, depth))

    def test_the_two_searches_pick_the_same_move(self):
        for columns in positions(CORPUS_SIZE, seed=2):
            for depth in DEPTHS:
                with self.subTest(columns=columns, depth=depth):
                    self.assertEqual(
                        python_alpha_beta(Connect4(columns), depth=depth),
                        native_alpha_beta(Connect4(columns), depth=depth),
                    )

    def test_ties_go_to_the_move_generated_first(self):
        # Which for Connect 4 is the centre column, because generation is centre-first.
        self.assertEqual(COLS // 2, native_alpha_beta(Connect4(), depth=2))

    def test_a_win_in_one_scores_as_a_win(self):
        scores = dict(rust_root_scores([1, 0, 2, 0, 3, 0], 2))
        self.assertEqual(MATE + 1, scores[4], 'column 4 completes the line')
        self.assertTrue(all(score < MATE for column, score in scores.items() if column != 4))

    def test_a_custom_evaluation_is_refused_rather_than_ignored(self):
        with self.assertRaises(ValueError):
            native_alpha_beta(Connect4(), depth=2, evaluate=lambda state: 0)


@needs_native
class TestTheEvaluationIsTheSameEvaluation(unittest.TestCase):
    """`value()` compared directly, independently of any search sitting on top of it."""

    def test_value_matches_over_a_wide_sample(self):
        for columns in positions(400, seed=3, plies=12):
            with self.subTest(columns=columns):
                board = Connect4(columns)
                self.assertEqual(
                    value(board),
                    zero_rs.evaluate(board.discs[YELLOW], board.discs[RED], board.turn),
                )

    def test_the_empty_board_and_full_columns_agree(self):
        empty = Connect4()
        self.assertEqual(
            value(empty), zero_rs.evaluate(empty.discs[YELLOW], empty.discs[RED], empty.turn))

        board = Connect4()
        for _ in range(6):
            board.make_move(3)
        self.assertEqual(
            value(board), zero_rs.evaluate(board.discs[YELLOW], board.discs[RED], board.turn))

    def test_an_unreachable_position_is_refused(self):
        with self.assertRaises(ValueError):
            zero_rs.evaluate(1, 1, True)  # The same cell claimed by both players


@needs_native
class TestChoosingAnEngine(unittest.TestCase):
    """`available()` checks the game and the scoring convention, mirroring `ai/zero/fast.py`."""

    def test_available_checks_the_game(self):
        self.assertTrue(native_available(Connect4))
        self.assertTrue(native_available())
        self.assertFalse(native_available(TicTacToe))


def write_fixture(path='rust/crates/c4-core/tests/fixtures/alphabeta_fixture.rs'):
    """Regenerates the pinned answers the Rust suite checks against without an interpreter."""
    lines = [
        "// The Python search's answers, generated once and pinned. Regenerate with",
        '// `python3 -m tests.connect4.test_native --write-fixture`; see tests/alphabeta.rs for',
        '// what they are compared against.',
        '',
        '/// A position, the depth it was searched to, its root scores and the leaves it took.',
        "pub type Pinned = (&'static [u8], i32, &'static [(u8, i32)], u64);",
        '',
        'pub const FIXTURE: &[Pinned] = &[',
    ]
    for columns in positions(CORPUS_SIZE):
        for depth in DEPTHS:
            scores, leaves = python_root_scores(columns, depth)
            played = ', '.join(str(column) for column in columns)
            scored = ', '.join(f'({column}, {score})' for column, score in scores)
            lines.append(f'    (&[{played}], {depth}, &[{scored}], {leaves}),')
    lines.append('];')

    with open(path, 'w') as fixture:
        fixture.write('\n'.join(lines) + '\n')
    return path


if __name__ == '__main__':  # pragma: no cover - a maintenance command, not a test
    import sys

    if '--write-fixture' in sys.argv:
        print(f'wrote {write_fixture()}')
    else:
        unittest.main()
