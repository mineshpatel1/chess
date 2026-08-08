"""
The evaluation, tested for properties rather than for numbers.

An evaluation has no right answer to check against - that is the whole reason ai/match.py
exists - so pinning what it returns for a given position would only pin today's weights and
would have to be rewritten every time they are tuned. What can be pinned is the shape: that it
is fair to both players, that it does not care which way round the board is, that it stays
inside the range the search reserves for real wins, and that the threat masks it is built from
find exactly the cells they should.

The mirror test is the one that earns its keep. Reflecting the board left to right maps the
rising diagonals onto the falling ones, so a sign or shift-direction error in either of them
shows up here and almost nowhere else.
"""

import random
import unittest
from typing import List

from ai.search import MATE
from games.connect4.bitboard import drops, index
from games.connect4.board import Connect4
from games.connect4.constants import (
    COLS,
    DIRECTIONS,
    FULL_BOARD,
    HORIZONTAL,
    RED,
    VERTICAL,
    YELLOW,
)
from games.connect4.evaluation import (
    MAX_EVAL,
    threat_cells,
    threat_value,
    value,
    weighted_eval,
)
from tests.connect4.corpus import positions


def mirrored(columns: List[int]) -> List[int]:
    """The same game played into the reflected columns."""
    return [COLS - 1 - column for column in columns]


class TestSymmetry(unittest.TestCase):
    def test_the_two_players_are_valued_the_same_way(self):
        """
        Every position, played by the other side, must be worth exactly its negative. An
        evaluation that is kinder to one colour makes an engine that plays one side well.
        """
        for columns in positions(30):
            board = Connect4(columns)
            swapped = Connect4(columns)
            swapped.discs[YELLOW], swapped.discs[RED] = swapped.discs[RED], swapped.discs[YELLOW]

            self.assertEqual(-value(board), value(swapped), f'{columns}')

    def test_reflecting_the_board_changes_nothing(self):
        """
        Connect 4 is symmetric left to right, so the reflection of a position is worth what the
        position is worth. Reflection swaps the two diagonals, so this is what catches a shift
        that goes the wrong way in one of them.
        """
        for columns in positions(30, seed=5):
            self.assertEqual(
                value(Connect4(columns)),
                value(Connect4(mirrored(columns))),
                f'{columns} reflected',
            )

    def test_an_empty_board_is_level(self):
        self.assertEqual(0, value(Connect4()))

    def test_a_position_with_no_open_three_is_worth_exactly_nothing(self):
        """
        Not an accident, and the most important property in this module.

        With every move scoring the same, the search takes the first one generated, and
        `legal_moves` generates centre-first - so an evaluation that says nothing in a quiet
        position *is* the policy "take the middle unless there is a tactic". Measured over
        hundreds of games, that is worth more than any positional term tried on top of it: a
        centre-column bonus, the most obviously correct term of the lot, scored 0.425 against
        this same search evaluating everything as zero. Anything added here has to beat
        silence, and silence is a higher bar than it looks.
        """
        rng = random.Random(9)
        quiet = 0

        for _ in range(60):
            board = Connect4()
            while not board.is_game_over:
                empty = FULL_BOARD & ~board.occupied
                threats = any(
                    threat_cells(board.discs[player], direction, 3, empty)
                    for player in (YELLOW, RED)
                    for direction in DIRECTIONS
                )
                if not threats:
                    self.assertEqual(0, value(board), f'{board}')
                    quiet += 1
                board.make_move(rng.choice(list(board.legal_moves)))

        self.assertGreater(quiet, 100, 'not enough quiet positions to have tested anything')

    def test_the_evaluation_speaks_for_the_player_to_move(self):
        """
        Negamax reads every score from the point of view of whoever is on move, so the same
        position must be worth opposite amounts to the two players. Missing this flip is the
        classic way to build an engine that is strong at even depths and weak at odd ones.
        """
        for columns in positions(20, seed=6):
            board = Connect4(columns)
            absolute = value(board)
            self.assertEqual(absolute if board.turn else -absolute, weighted_eval(board))

            board.make_move(next(iter(board.legal_moves)))
            self.assertEqual(
                value(board) if board.turn else -value(board), weighted_eval(board)
            )


class TestBounds(unittest.TestCase):
    def test_no_position_is_valued_anywhere_near_a_win(self):
        """
        `ai.search` scores a forced win as MATE plus the depth remaining. An evaluation that
        could reach it would make a good position indistinguishable from a won one, and the
        engine would stop bothering to finish games.
        """
        rng = random.Random(0)
        worst = 0

        for _ in range(200):
            board = Connect4()
            while not board.is_game_over:
                board.make_move(rng.choice(list(board.legal_moves)))
                worst = max(worst, abs(value(board)))

        self.assertLess(worst, MAX_EVAL, 'the declared budget was exceeded')
        self.assertLess(MAX_EVAL, MATE // 10, 'the budget is not far enough inside MATE')


class TestThreatCells(unittest.TestCase):
    """
    The masks the value is built out of, checked cell by cell.

    `threat_cells` is `runs` one length short, plus a shift to the cell that would finish the
    job, so it inherits the sentinel protection that stops a run wrapping between columns - and
    the last case here is what confirms that it really does.
    """

    def _cells(self, board: Connect4, player, direction, length):
        empty = FULL_BOARD & ~board.occupied
        return threat_cells(board.discs[player], direction, length, empty)

    def test_a_three_open_at_both_ends_offers_two_cells(self):
        board = Connect4([1, 0, 2, 0, 3, 0])  # Yellow on columns 1, 2, 3 of the bottom row
        cells = self._cells(board, YELLOW, HORIZONTAL, 3)

        self.assertEqual((1 << index(4, 0)), cells & (1 << index(4, 0)))
        self.assertEqual(0, cells & (1 << index(0, 0)), 'column 0 is taken by Red')

    def test_a_three_with_a_wall_at_one_end_offers_one(self):
        """Yellow on columns 0, 1, 2 of the bottom row: nothing to the left of column 0."""
        board = Connect4([0, 6, 1, 6, 2, 6])
        cells = self._cells(board, YELLOW, HORIZONTAL, 3)

        self.assertEqual(1 << index(3, 0), cells)

    def test_a_vertical_three_at_the_top_of_a_column_offers_nothing(self):
        """
        The sentinel case, in the evaluation rather than in win detection. The cell above a
        three at the top of a column is a sentinel, and `empty` is carved out of FULL_BOARD,
        which has no sentinels in it - so the shift lands on nothing.
        """
        board = Connect4.from_diagram('''
            ...Y...
            ...Y...
            ...Y...
            ...R...
            ...R...
            ...R...
        ''')
        self.assertEqual(0, self._cells(board, YELLOW, VERTICAL, 3))

    def test_a_vertical_three_lower_down_offers_the_cell_above_it(self):
        board = Connect4([3, 0, 3, 0, 3, 0])
        self.assertEqual(1 << index(3, 3), self._cells(board, YELLOW, VERTICAL, 3))

    def test_a_threat_cell_is_never_off_the_board(self):
        rng = random.Random(1)
        for _ in range(100):
            board = Connect4()
            while not board.is_game_over:
                board.make_move(rng.choice(list(board.legal_moves)))
                empty = FULL_BOARD & ~board.occupied
                for player in (YELLOW, RED):
                    for direction in DIRECTIONS:
                        for length in (2, 3):
                            cells = threat_cells(board.discs[player], direction, length, empty)
                            self.assertEqual(cells, cells & FULL_BOARD, f'{board}')
                            self.assertEqual(0, cells & board.occupied, f'{board}')


class TestThreatValue(unittest.TestCase):
    def test_a_live_threat_is_worth_more_than_a_distant_one(self):
        """
        The same three discs, scored twice, with only playability changed.

        `threat_value` takes raw masks, so the comparison can hold the position fixed and vary
        nothing but which cells are reachable this move - which is the only honest way to price
        a single term. Two boards differing in where the discs sit would differ in half a dozen
        other ways at the same time.
        """
        three = sum(1 << index(column, 0) for column in (1, 2, 3))
        empty = FULL_BOARD & ~three

        on_the_ground = threat_value(three, empty, drops(three))
        in_the_air = threat_value(three, empty, 0)  # As if nothing could be played at all

        self.assertGreater(on_the_ground, in_the_air)
        self.assertGreater(in_the_air, 0, 'a distant threat is worth less, not nothing')

    def test_nothing_on_the_board_is_worth_nothing(self):
        self.assertEqual(0, threat_value(0, FULL_BOARD, 0))

    def test_a_diagonal_threat_outweighs_a_vertical_one(self):
        """
        A vertical three is answered by playing on top of it, which costs the opponent a move
        they were going to spend somewhere anyway. A diagonal has to be answered in one
        particular column.
        """
        vertical = Connect4([3, 0, 3, 0, 3, 0])
        diagonal = Connect4([0, 1, 1, 2, 2, 6, 2])

        self.assertGreater(abs(value(diagonal)), abs(value(vertical)))


def main():
    unittest.main()


if __name__ == '__main__':
    main()
