"""
Perft: every distinct sequence of legal moves of a given length, counted.

Connect 4 has no published reference table the way chess does, so the counts here are derived
rather than looked up - and it turns out the derivation is unusually clean, because the tree is
completely unconstrained for the first six plies:

  * A column takes ROWS discs, so none can be full before ply 6, and every position up to then
    offers all COLS moves. perft(d) = COLS ** d for d <= 6.

  * The earliest possible win is ply 7, the first player's fourth disc. A win at ply 7 is still
    a leaf and still counts, so wins do not reduce the count until ply 8.

  * At ply 7 the only thing missing is the seven six-ply prefixes that drop every disc into one
    column: each offers 6 moves instead of 7. perft(7) = COLS ** 7 - COLS = 823,536.

So depths 1 to 6 are checked against a formula computed here rather than pasted, and depth 7 is
checked against the one number in the table that argues for itself. Depth 8 - where wins start
truncating lines and hand derivation runs out - is recorded in the README instead, being some
seconds of work for a number nothing else depends on.
"""

import unittest

from ai.perft import divide, traverse_moves
from games.connect4.board import Connect4
from games.connect4.constants import COLS, CONNECT, ROWS


class TestPermutations(unittest.TestCase):
    def test_the_tree_is_unconstrained_for_the_first_six_plies(self):
        """
        Nothing can fill a column or win a game inside six moves, so every node branches COLS
        ways. Anything that generates a move too few or too many shows up immediately, and
        compounds with depth.
        """
        for depth in range(1, ROWS + 1):
            self.assertEqual(
                COLS ** depth,
                traverse_moves(Connect4(), depth, False),
                f'perft({depth})',
            )

    def test_the_first_full_column_shows_up_at_depth_seven(self):
        """
        `COLS ** 7 - COLS`: seven prefixes stack a whole column and then have one fewer move.
        This is the first depth whose count is not a power of seven, and so the first that says
        anything about move generation that the depths above it did not.
        """
        self.assertEqual(COLS ** 7 - COLS, traverse_moves(Connect4(), 7, False))
        self.assertEqual(823_536, COLS ** 7 - COLS)

    def test_no_game_can_be_won_before_the_seventh_ply(self):
        """
        The other half of the claim the counts above rest on. The first player's fourth disc
        arrives at ply 7, so nothing can be decided before then - and if anything were, the
        decided-position rule in ai/perft.py would be truncating lines and the powers of seven
        above would not hold.
        """
        self.assertEqual(0, self._decided_positions(Connect4(), CONNECT * 2 - 2))

    def test_the_seventh_ply_does_decide_some_games(self):
        """
        The complement, shown by example rather than by another full walk of the tree: at ply 7
        there is a decided position, so the bound above is exactly where it should be and not
        merely somewhere above it.
        """
        earliest = Connect4([0, 1, 0, 1, 0, 1, 0])
        self.assertEqual(CONNECT * 2 - 1, len(earliest.move_stack))
        self.assertIsNotNone(earliest.outcome)

    def _decided_positions(self, state, depth: int) -> int:
        """Distinct move sequences of `depth` plies that finish the game."""
        if state.outcome is not None:
            return 1 if depth == 0 else 0
        if depth == 0:
            return 0

        total = 0
        for move in state.legal_moves:
            state.make_move(move)
            total += self._decided_positions(state, depth - 1)
            state.unmake_move()
        return total

    def test_a_decided_position_has_no_continuations(self):
        """
        What the ai/perft.py change is for. Yellow has won along the bottom row and six columns
        are still playable, so a perft that did not ask would happily count the discs dropped
        after the game ended.
        """
        won = Connect4([0, 0, 1, 1, 2, 2, 3])
        self.assertIsNotNone(won.outcome)
        self.assertTrue(any(won.legal_moves), 'the point of this test is that moves remain')

        for depth in range(1, 4):
            self.assertEqual(0, traverse_moves(won, depth, False), f'depth {depth}')
        self.assertEqual({}, divide(won, 2))

    def test_a_position_decided_at_the_horizon_is_still_a_leaf(self):
        """
        The ordering half of the same change. A game that finishes at exactly `depth` plies is
        one sequence and counts as one, which is what chess gets for free from a checkmate
        generating no moves.
        """
        one_from_won = Connect4([0, 0, 1, 1, 2, 2])
        self.assertIsNone(one_from_won.outcome)
        self.assertEqual(COLS, traverse_moves(one_from_won, 1, False))

    def test_divide_accounts_for_the_whole_tree(self):
        counts = divide(Connect4(), 4)
        self.assertEqual(COLS, len(counts))
        self.assertEqual(COLS ** 4, sum(counts.values()))


def main():
    unittest.main()


if __name__ == '__main__':
    main()
