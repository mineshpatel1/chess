"""
The safety net for rewriting the search.

The search has no oracle but itself: perft pins move generation, and tests/test_search.py pins
a handful of positions whose right answer we happen to know, but neither would notice the
search quietly picking a different move in the middle of a game. So before the search is
rewritten, this compares implementations directly - the same positions, the same depths, and
every root move's score, not just the one that wins.

`root_scores` is the unit of comparison rather than the chosen move alone. Two searches can
agree on the best move while disagreeing about everything else, and that is a drift worth
catching early.
"""

import unittest
from typing import List, Tuple

from ai.algorithms import (
    _get_best_move,
    _negamax_root_move,
    alpha_beta,
    weighted_eval,
    relative_weighted_eval,
)
from games.chess.board import Board
from tests.corpus import positions, DECISIVE_POSITIONS

# Kept small enough to stay in the default suite. The equivalence run that gates the rewrite
# widens these considerably - see the module docstring in tests/test_all.py.
CORPUS_SIZE = 40
DEPTHS = (1, 2, 3)


def root_scores(fen: str, depth: int) -> List[Tuple[str, int]]:
    """
    Every root move and the score the current search gives it, in generation order.

    This deliberately mirrors what `alpha_beta` does internally - a full window per root move,
    each searched independently - so that comparing it against a replacement compares the real
    thing rather than a simplification of it.
    """
    scores = []
    for move in Board(fen).legal_moves:
        _, value, _ = _get_best_move(Board(fen), depth, move, weighted_eval)
        scores.append((move.uci, value))
    return scores


def negamax_root_scores(fen: str, depth: int) -> List[Tuple[str, int]]:
    """The same thing from the negamax search, for comparison against `root_scores`."""
    scores = []
    for move in Board(fen).legal_moves:
        _, value = _negamax_root_move(Board(fen), depth, move, relative_weighted_eval)
        scores.append((move.uci, value))
    return scores


def best_of(scores: List[Tuple[str, int]]) -> str:
    """The move `alpha_beta` would pick from these scores: highest, ties going to the first."""
    best_move, best_score = scores[0]
    for uci, score in scores[1:]:
        if score > best_score:
            best_move, best_score = uci, score
    return best_move


class TestHarness(unittest.TestCase):
    """
    Tests of the comparison machinery itself. If these fail, the equivalence result that gates
    the search rewrite means nothing.
    """

    def test_corpus_is_reproducible(self):
        self.assertEqual(positions(30, seed=1), positions(30, seed=1))

    def test_corpus_positions_all_have_moves(self):
        for fen in positions(CORPUS_SIZE):
            self.assertTrue(any(Board(fen).legal_moves), fen)

    def test_search_is_deterministic(self):
        """A comparison against a search that varies run to run would prove nothing."""
        for fen in positions(10):
            self.assertEqual(root_scores(fen, 2), root_scores(fen, 2), fen)

    def test_root_scores_agrees_with_alpha_beta(self):
        """
        The harness must measure the same thing the engine does. If `root_scores` and
        `alpha_beta` can disagree about the best move, then `root_scores` is not a faithful
        stand-in for the search and cannot vouch for a replacement.
        """
        for fen in positions(12):
            for depth in (1, 2, 3):
                self.assertEqual(
                    best_of(root_scores(fen, depth)),
                    alpha_beta(Board(fen), depth=depth).uci,
                    f'{fen} at depth {depth}',
                )


class TestNegamaxMatchesAlphaBeta(unittest.TestCase):
    """
    The gate on replacing the min/max pair with negamax.

    The two formulations should be the same search wearing different clothes, so they are held
    to agreeing on every root move's score - not merely on which move wins - across the whole
    corpus. Anything less would let a scoring change hide behind a coincidence of ordering.
    """

    def test_root_scores_are_identical(self):
        for fen in positions(CORPUS_SIZE):
            for depth in DEPTHS:
                self.assertEqual(
                    root_scores(fen, depth),
                    negamax_root_scores(fen, depth),
                    f'{fen} at depth {depth}',
                )

    def test_decisive_positions_agree(self):
        """Mate and stalemate scoring is where the sign conventions differ, so pin it hardest."""
        for fen in DECISIVE_POSITIONS:
            for depth in (1, 2, 3, 4):
                self.assertEqual(
                    root_scores(fen, depth),
                    negamax_root_scores(fen, depth),
                    f'{fen} at depth {depth}',
                )


def main():
    unittest.main()


if __name__ == '__main__':
    main()
