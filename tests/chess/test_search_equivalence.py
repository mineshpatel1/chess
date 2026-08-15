"""
The safety net for changing the search.

The search has no oracle but itself: perft pins move generation, and tests/test_search.py pins
a handful of positions whose right answer we happen to know, but neither would notice the
search quietly picking a different move in the middle of a game.

This module exists to make that noticeable. It scores *every root move*, not just the one that
wins, over a reproducible corpus - two searches can agree on the best move while disagreeing
about everything else, and that is a drift worth catching. Replacing the min/max pair with
negamax was gated on exactly this comparison: 37,167 root-move scores across 360 positions at
depths 1-4, zero mismatches. The old search is gone, so what remains is the machinery, plus
scores recorded against the surviving search so the next change to it has something to answer
to.

To run a wide comparison again, widen COMPARED and DEPTHS and expect minutes, not seconds: cost
grows as branching^depth, and corpus positions branch far wider than the opening does.
"""

import unittest
from typing import List, Tuple

from ai.search import _root_move_score, alpha_beta, MATE
from games.chess.board import ChessBoard
from games.chess.evaluation import weighted_eval
from tests.chess.corpus import positions

# Kept small enough to stay in the default suite; a real equivalence run widens all three, as the
# module docstring says. COMPARED is separate from CORPUS_SIZE because searching a position at
# every depth costs orders of magnitude more than checking that it has moves.
CORPUS_SIZE = 40
COMPARED = 8
DEPTHS = (1, 3)


def root_scores(fen: str, depth: int) -> List[Tuple[str, int]]:
    """
    Every root move and the score the search gives it, in generation order.

    This mirrors what `alpha_beta` does internally - a full window per root move, each
    searched independently - so that comparing it against a replacement compares the real
    thing rather than a simplification of it.
    """
    scores = []
    for move in ChessBoard(fen).legal_moves:
        _, value = _root_move_score(ChessBoard(fen), depth, move, weighted_eval)
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

    def test_corpus_positions_all_have_moves(self):
        for fen in positions(CORPUS_SIZE):
            self.assertTrue(any(ChessBoard(fen).legal_moves), fen)

    def test_root_scores_agrees_with_alpha_beta(self):
        """
        The harness must measure the same thing the engine does. If `root_scores` and
        `alpha_beta` can disagree about the best move, then `root_scores` is not a faithful
        stand-in for the search and cannot vouch for a replacement.

        A handful of positions at the shallowest and deepest depths the suite can afford.
        """
        for fen in positions(COMPARED):
            for depth in DEPTHS:
                self.assertEqual(
                    best_of(root_scores(fen, depth)),
                    alpha_beta(ChessBoard(fen), depth=depth).uci,
                    f'{fen} at depth {depth}',
                )


class TestDecisivePositions(unittest.TestCase):
    """
    Scores at and beside a finish, recorded rather than reasoned about.

    Mate and stalemate scoring is where a sign convention goes wrong, and where the old
    formulation and the new one had to be proven to agree. That comparison is gone with the
    old search, so these pin the surviving one against drifting on its own: a win is worth
    MATE plus the depth remaining, a draw is worth nothing, and both are worth the same from
    either side of the board.
    """

    def test_a_forced_mate_scores_as_a_win(self):
        # Ra8 is mate in one, and should be the only move worth a mate score
        scores = dict(root_scores('6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1', 3))
        self.assertEqual(scores['a1a8'], MATE + 2)
        self.assertTrue(all(v < MATE for m, v in scores.items() if m != 'a1a8'))

    def test_a_closer_mate_outscores_a_further_one(self):
        """Rc8 mates at once. Several Queen moves mate a move later and are generated first."""
        scores = dict(root_scores('7k/Q7/8/8/8/8/8/2R4K w - - 0 1', 4))
        mates = sorted((v for v in scores.values() if v > MATE), reverse=True)
        self.assertEqual(scores['c1c8'], mates[0])
        self.assertGreater(mates[0], mates[-1])

    def test_a_stalemate_scores_as_a_draw(self):
        """
        White is a Knight and four pawns down, so the stalemate at a5a6 is the best score on
        the board and a draw is worth exactly nothing.

        Searched at an even depth deliberately. There is no quiescence, so at odd depths the
        search gets the last move of the sequence and Rxb8 looks like a free Knight - it
        scores +40 at depths 1 and 3 and the recapture falls over the horizon. That is a known
        property of a fixed-depth search, not something for this test to trip over.
        """
        scores = dict(root_scores('kn5R/3p1p2/1P1p1p2/P2p1p2/3p1P2/3p4/3P4/7K w - - 0 1', 4))
        self.assertEqual(scores['a5a6'], 0)
        self.assertEqual(max(scores.values()), 0)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
