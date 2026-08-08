"""
The evaluation: what it counts, and the bounds the search needs it to respect.

None of this says the evaluation is any *good* - there is no oracle for that, which is what
`ai.match` is for and what the numbers in the module docstring of games/tictactoe/evaluation.py
came from. What is testable is that it counts what it claims to count, that it is symmetric
between the two players, and that it can never wander into the range `ai.search` reserves for
forced wins. The last of those is checked over every position in the game rather than a sample,
because it can be.
"""

import unittest

from ai.search import MATE
from games.tictactoe.board import TicTacToe
from games.tictactoe.constants import CROSS, NOUGHT
from games.tictactoe.evaluation import MAX_EVAL, bit_count, open_twos, value, weighted_eval
from tests.tictactoe.corpus import reachable_positions


class TestBitCount(unittest.TestCase):
    def test_it_counts_set_bits(self):
        self.assertEqual(0, bit_count(0))
        self.assertEqual(1, bit_count(0b1000))
        self.assertEqual(3, bit_count(0b111))
        self.assertEqual(9, bit_count(0b111111111))


class TestOpenTwos(unittest.TestCase):
    def test_an_empty_board_has_none(self):
        self.assertEqual(0, open_twos(0, 0))

    def test_two_in_a_line_with_the_third_free_is_one(self):
        state = TicTacToe([0, 4, 1])  # Crosses at 0 and 1, cell 2 free
        self.assertEqual(1, open_twos(state.marks[CROSS], state.marks[NOUGHT]))

    def test_a_line_the_opponent_is_in_is_dead(self):
        """The difference between counting threats and counting pairs."""
        state = TicTacToe([0, 2, 1])  # Crosses at 0 and 1, but Noughts hold cell 2
        self.assertEqual(0, open_twos(state.marks[CROSS], state.marks[NOUGHT]))

    def test_one_mark_is_not_a_threat(self):
        state = TicTacToe([4])
        self.assertEqual(0, open_twos(state.marks[CROSS], state.marks[NOUGHT]))

    def test_a_completed_line_is_not_counted_as_a_threat(self):
        """
        Three marks in a line is a won game, which is the search's business - `outcome` is
        checked before the evaluation is ever reached. Counting it here as well would let a
        finished position score like a merely promising one.
        """
        state = TicTacToe([0, 3, 1, 4, 2])  # Crosses hold the whole top row
        self.assertEqual(0, open_twos(state.marks[CROSS], state.marks[NOUGHT]))

    def test_threats_are_counted_one_per_line(self):
        state = TicTacToe.from_diagram("""
            O.X
            .X.
            ...
        """)
        # Crosses at 2 and 4: only 2-4-6 has two marks and a free third.
        self.assertEqual(1, open_twos(state.marks[CROSS], state.marks[NOUGHT]))

    def test_a_fork_counts_once_per_line_it_threatens(self):
        """
        Several threats at once is the position that actually wins games of tic-tac-toe, and the
        reason the term is a count rather than a flag.
        """
        forked = TicTacToe.from_diagram("""
            OOX
            OX.
            .XX
        """)
        # Crosses at 2, 4, 7 and 8 threaten 6-7-8, 2-5-8 and 2-4-6; every other line has a Nought
        # in it. Three is the most any position in the game allows.
        self.assertEqual(3, open_twos(forked.marks[CROSS], forked.marks[NOUGHT]))


class TestValue(unittest.TestCase):
    def test_an_empty_board_is_level(self):
        self.assertEqual(0, value(TicTacToe()))

    def test_a_threat_favours_the_player_who_has_it(self):
        crosses = TicTacToe([0, 4, 1])
        self.assertGreater(value(crosses), 0)

        noughts = TicTacToe([4, 0, 8, 1])  # Noughts at 0 and 1 with cell 2 free
        self.assertLess(value(noughts), 0)

    def test_swapping_the_marks_negates_the_value(self):
        """
        The symmetry that says the two players are being judged by the same standard. A term that
        counts one side more eagerly than the other gives an engine that plays one colour well.
        """
        for state in reachable_positions():
            mirrored = state.copy()
            mirrored.marks[CROSS], mirrored.marks[NOUGHT] = (
                state.marks[NOUGHT],
                state.marks[CROSS],
            )
            self.assertEqual(-value(state), value(mirrored), str(state))

    def test_the_value_stays_well_inside_the_mate_range(self):
        """
        Over every position in the game. An evaluation that can reach `ai.search.MATE` makes a
        merely good position indistinguishable from a forced win, and the engine stops finishing
        games it has won.

        The bound is not tight and does not need to be - the largest value any position actually
        reaches is 30, three threats at TWO each - but it is the number MAX_EVAL promises, and a
        term added later has to keep the promise rather than re-derive it.
        """
        largest = 0
        for state in reachable_positions():
            largest = max(largest, abs(value(state)))
            self.assertLessEqual(abs(value(state)), MAX_EVAL, str(state))

        self.assertEqual(30, largest)
        self.assertLess(MAX_EVAL, MATE)


class TestWeightedEval(unittest.TestCase):
    def test_it_reads_from_the_point_of_view_of_the_player_to_move(self):
        """
        The sign flip, in the one place it lives. Getting it wrong gives an engine that plays
        well at even depths and badly at odd ones, which is hard to notice by eye.
        """
        for state in reachable_positions():
            expected = value(state) if state.turn == CROSS else -value(state)
            self.assertEqual(expected, weighted_eval(state), str(state))

    def test_a_threat_is_good_news_for_whoever_holds_it(self):
        crosses_threaten = TicTacToe([0, 4, 1])  # Noughts to move, facing a threat
        self.assertEqual(NOUGHT, crosses_threaten.turn)
        self.assertLess(weighted_eval(crosses_threaten), 0)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
