"""
Win detection, and the boundary cases the sentinel row exists to catch.

`is_win` is four shift chains over one integer, and the thing that can go wrong with it is not
that it misses a win - a broken version would fail the first test written - but that it finds
one that is not there, by letting a run walk off the top of one column and into the bottom of
the next. Those are the cases nobody writes down by hand, so most of this module does not try:
it derives the right answer from (column, row) arithmetic and checks every start cell in every
direction against it.

The three cases the specification calls out are written out longhand as well, because a test
that says *why* is worth having next to one that says how many.
"""

import random
import unittest
from typing import List, Optional, Set, Tuple

from games.base import has_moves
from games.connect4.bitboard import (
    cells,
    completions,
    has_run,
    index,
    is_win,
    runs,
)
from games.connect4.board import Connect4
from games.connect4.constants import (
    COLS,
    CONNECT,
    DIAGONAL_DOWN,
    DIAGONAL_UP,
    DIRECTIONS,
    FULL_BOARD,
    HORIZONTAL,
    RED,
    ROWS,
    SIZE,
    STRIDE,
    VERTICAL,
    YELLOW,
)
from games.base import DRAW, win

# The (dcolumn, drow) each index step means. This is the definition the deltas are derived
# from, restated independently so the test does not assume the constant it is checking.
STEPS = {
    VERTICAL: (0, 1),
    HORIZONTAL: (1, 0),
    DIAGONAL_UP: (1, 1),
    DIAGONAL_DOWN: (1, -1),
}


def line_from(column: int, row: int, dcolumn: int, drow: int) -> Optional[List[Tuple[int, int]]]:
    """
    The CONNECT cells of a run starting here, or None if it leaves the board.

    Deliberately in (column, row) terms and not in bit terms: this is the oracle, so it has to
    be wrong in different ways from the thing it is checking.
    """
    line = [(column + step * dcolumn, row + step * drow) for step in range(CONNECT)]
    if all(0 <= c < COLS and 0 <= r < ROWS for c, r in line):
        return line
    return None


def naive_is_win(occupied: Set[Tuple[int, int]]) -> bool:
    """Whether a set of (column, row) cells holds a line, by scanning a grid the slow way."""
    for column in range(COLS):
        for row in range(ROWS):
            for dcolumn, drow in STEPS.values():
                line = line_from(column, row, dcolumn, drow)
                if line and all(cell in occupied for cell in line):
                    return True
    return False


def occupied_cells(bb: int) -> Set[Tuple[int, int]]:
    """A bitboard as a set of (column, row), which is how the oracle above thinks."""
    return {divmod(cell, STRIDE) for cell in cells(bb)}


class TestRuns(unittest.TestCase):
    """The shift primitive on its own, before anything reads a win out of it."""

    def test_a_run_is_marked_at_its_lowest_cell(self):
        position = sum(1 << index(0, row) for row in range(CONNECT))
        self.assertEqual(1 << index(0, 0), runs(position, VERTICAL))

    def test_five_in_a_line_contains_two_runs_of_four(self):
        position = sum(1 << index(0, row) for row in range(5))
        self.assertEqual(2, len(list(cells(runs(position, VERTICAL)))))

    def test_the_halving_works_at_every_length(self):
        """
        The step doubles and is clamped, so the shifts have to total length - 1 whether that
        number is a power of two or not. Lengths 3 and 5 are the ones that clamp.
        """
        for length in range(2, 7):
            position = sum(1 << index(3, row) for row in range(length))
            for shorter in range(2, length + 1):
                self.assertTrue(has_run(position, VERTICAL, shorter), f'{length} holds {shorter}')
            self.assertFalse(has_run(position, VERTICAL, length + 1), f'{length} is not longer')

    def test_a_gap_breaks_a_run(self):
        position = (1 << index(0, 0)) | (1 << index(0, 1)) | (1 << index(0, 3))
        self.assertFalse(has_run(position, VERTICAL))

    def test_an_empty_board_holds_nothing(self):
        self.assertFalse(is_win(0))
        for delta in DIRECTIONS:
            self.assertEqual(0, runs(0, delta))


class TestExhaustiveWins(unittest.TestCase):
    """
    Every possible run of four, in every direction, from every cell.

    SIZE * 4 = 196 cases, which is the whole space. Three hand-written boundary cases prove
    three things; this proves that no start cell and no direction anywhere on the board - or on
    the sentinel row - can produce an answer the geometry does not agree with.
    """

    def test_every_line_of_four_is_recognised_exactly_when_it_is_real(self):
        checked = wins = 0

        for start in range(SIZE):
            column, row = divmod(start, STRIDE)

            for delta, (dcolumn, drow) in STEPS.items():
                # The mask a naive implementation would build, cut down to the real board.
                # Anything that walked off the board loses cells here and cannot be four.
                position = sum(1 << (start + step * delta) for step in range(CONNECT))
                position &= FULL_BOARD

                expected = row < ROWS and line_from(column, row, dcolumn, drow) is not None
                self.assertEqual(
                    expected,
                    is_win(position),
                    f'{CONNECT} from cell {start} (column {column}, row {row}) by {delta}',
                )
                checked += 1
                wins += expected

        self.assertEqual(SIZE * len(STEPS), checked)
        # 69 lines of four fit on a 7x6 board: 24 horizontal, 21 vertical, 12 of each diagonal.
        self.assertEqual(69, wins)

    def test_it_agrees_with_a_naive_grid_scan_over_random_games(self):
        """A second opinion from an implementation that shares none of the bit tricks."""
        rng = random.Random(0)
        for _ in range(50):  # The scan is ~170 line checks per call; this is the suite budget
            board = Connect4()
            while not board.is_game_over:
                board.make_move(rng.choice(list(board.legal_moves)))
                for player in (YELLOW, RED):
                    discs = board.discs[player]
                    self.assertEqual(
                        naive_is_win(occupied_cells(discs)), is_win(discs), f'{board}'
                    )


class TestCompletions(unittest.TestCase):
    """
    The cells that would finish a line, which is `is_win` asked one move earlier.

    Worth its own exhaustive check for the same reason `is_win` gets one, and rather more
    urgently: `ai.oracle` returns WIN without searching when a winning move exists, so a cell
    marked here that does not really complete a line is an exact solver returning a wrong answer
    in silence. The oracle below is `is_win` itself, which shares the shift trick but not the
    shape of the computation - `runs` halves its way up a chain, `completions` ANDs a fixed set of
    offsets - and, more to the point, is already pinned by the two exhaustive tests above.
    """

    def test_a_cell_completes_a_line_exactly_when_adding_it_would_win(self):
        """
        Every cell of every board with three discs on a line, which is the whole of the space
        that matters: a cell completes a line for `position` if and only if putting a disc there
        makes `is_win` true. Both directions, over all 49 cells of all 196 candidate lines.
        """
        checked = marked = 0

        for start in range(SIZE):
            for delta in STEPS:
                line = sum(1 << (start + step * delta) for step in range(CONNECT)) & FULL_BOARD

                for gap in cells(line):
                    three = line ^ (1 << gap)
                    completing = completions(three)

                    for cell in cells(FULL_BOARD):
                        # Cells already held are skipped: `completions` says nothing about
                        # occupancy and may mark them, and the solver intersects with `drops`
                        # anyway. Sentinel cells are skipped by iterating FULL_BOARD, because the
                        # oracle would be wrong about those rather than `completions` - `is_win`
                        # does no masking, so it happily reads a line straight through one. That
                        # is what test_it_never_marks_a_sentinel_cell covers instead.
                        if three >> cell & 1:
                            continue

                        expected = is_win(three | 1 << cell)
                        self.assertEqual(
                            expected,
                            bool(completing >> cell & 1),
                            f'cell {cell} against three from {start} by {delta}',
                        )
                        checked += 1
                        marked += expected

        self.assertGreater(marked, 0, 'no completion was found anywhere, so this proved nothing')

    def test_it_marks_nothing_for_an_empty_board(self):
        self.assertEqual(0, completions(0))

    def test_it_never_marks_a_sentinel_cell(self):
        """
        The wrap check, and the reason the result is masked with FULL_BOARD.

        A left shift can carry a chain up out of a column; if the mask were omitted, a completion
        could be reported on a cell that is not on the board at all - and `drops` would never
        offer it, so the fault would only ever show up as a wrong value.
        """
        rng = random.Random(0)
        for _ in range(200):
            board = Connect4()
            for _ in range(rng.randrange(0, 30)):
                if board.outcome is not None:
                    break
                board.make_move(rng.choice(list(board.legal_moves)))

            for player in (YELLOW, RED):
                self.assertEqual(0, completions(board.discs[player]) & ~FULL_BOARD)

    def test_it_agrees_with_playing_the_move_over_random_games(self):
        """`Connect4.winning_moves` against the make-a-move-and-look version it replaces."""
        rng = random.Random(1)
        for _ in range(60):
            board = Connect4()
            while board.outcome is None and has_moves(board.legal_moves):
                claimed = list(board.winning_moves)

                actual = []
                for column in list(board.legal_moves):
                    board.make_move(column)
                    outcome = board.outcome
                    board.unmake_move()
                    if outcome is not None:
                        actual.append(column)

                self.assertEqual(sorted(actual), sorted(claimed), str(board))
                board.make_move(rng.choice(list(board.legal_moves)))


class TestSentinelBoundaries(unittest.TestCase):
    """
    The three cases the sentinel row exists for, written out rather than generated.

    Each is asserted both ways round: the same shape of run, placed where it is legal, must
    still register. A win detector that is simply broken everywhere would pass the negative
    half of these on its own.
    """

    def test_a_vertical_run_may_not_span_a_column_edge(self):
        """
        Cells 5, 6, 7, 8 are four consecutive indices, and a naive vertical check would call
        them a win. They are the top of column 0, a sentinel, and the bottom two of column 1.
        """
        spanning = (1 << 5) | (1 << 7) | (1 << 8)  # Cell 6 is the sentinel and cannot be set
        self.assertFalse(is_win(spanning))
        self.assertFalse(has_run(spanning, VERTICAL))

        # The same four discs stacked inside one column do win
        legal = sum(1 << index(0, row) for row in range(2, CONNECT + 2))
        self.assertTrue(has_run(legal, VERTICAL))

    def test_a_rising_diagonal_may_not_wrap_off_the_top_of_a_column(self):
        """
        A / diagonal climbs a row per column, so one starting high enough runs out of board.
        From column 0 row 3 it wants rows 3, 4, 5, 6 - and row 6 is the sentinel.
        """
        wrapping = sum(1 << (index(0, 3) + step * DIAGONAL_UP) for step in range(CONNECT))
        self.assertFalse(is_win(wrapping & FULL_BOARD))

        # Starting one row lower it fits, and wins
        legal = sum(1 << (index(0, 2) + step * DIAGONAL_UP) for step in range(CONNECT))
        self.assertTrue(has_run(legal & FULL_BOARD, DIAGONAL_UP))

    def test_a_falling_diagonal_may_not_wrap_off_row_zero(self):
        """
        A \\ diagonal drops a row per column, so one starting too low falls through the floor.
        From column 0 row 2 it wants rows 2, 1, 0, -1, and -1 is the top of the column before.
        """
        wrapping = sum(1 << (index(0, 2) + step * DIAGONAL_DOWN) for step in range(CONNECT))
        self.assertFalse(is_win(wrapping & FULL_BOARD))

        # Starting one row higher it fits, and wins
        legal = sum(1 << (index(0, 3) + step * DIAGONAL_DOWN) for step in range(CONNECT))
        self.assertTrue(has_run(legal & FULL_BOARD, DIAGONAL_DOWN))

    def test_a_horizontal_run_may_not_leave_the_right_hand_edge(self):
        """Not in the specification's list, but the same failure one direction over."""
        wrapping = sum(1 << (index(COLS - 2, 0) + step * HORIZONTAL) for step in range(CONNECT))
        self.assertFalse(is_win(wrapping & FULL_BOARD))

        legal = sum(1 << (index(COLS - CONNECT, 0) + step * HORIZONTAL) for step in range(CONNECT))
        self.assertTrue(has_run(legal & FULL_BOARD, HORIZONTAL))


class TestOutcome(unittest.TestCase):
    """
    What the search actually asks. `outcome` is the property chess does not have: Connect 4 is
    won with the board half empty and moves still on offer, so without it the search would play
    on through a finished game.
    """

    def test_a_win_in_each_direction_is_reported(self):
        for name, columns in (
            ('vertical', [3, 4, 3, 4, 3, 4, 3]),
            ('horizontal', [0, 0, 1, 1, 2, 2, 3]),
            ('rising diagonal', [0, 1, 1, 2, 2, 3, 2, 3, 3, 6, 3]),
            ('falling diagonal', [3, 2, 2, 1, 1, 0, 1, 0, 0, 6, 0]),
        ):
            board = Connect4(columns)
            self.assertEqual(win(YELLOW), board.outcome, f'{name}{board}')
            self.assertEqual(win(YELLOW), board.result, f'{name}{board}')
            self.assertTrue(board.is_game_over, name)

    def test_a_game_still_running_has_no_outcome(self):
        for columns in ([], [3], [3, 3], [3, 4, 3, 4, 3]):
            board = Connect4(columns)
            self.assertIsNone(board.outcome, f'{columns}{board}')
            self.assertIsNone(board.result, f'{columns}{board}')

    def test_three_in_a_row_is_not_a_win(self):
        self.assertIsNone(Connect4([0, 0, 1, 1, 2, 2]).outcome)

    def test_the_second_player_can_win_too(self):
        board = Connect4([6, 3, 6, 4, 6, 5, 0, 2])
        self.assertEqual(win(RED), board.outcome)

    def test_nothing_to_play_is_a_draw(self):
        self.assertEqual(DRAW, Connect4().outcome_without_moves)

    def test_only_the_player_who_just_moved_is_tested(self):
        """
        The saving that halves the cost of the property the search touches at every node. It
        holds because the position where it would matter cannot be built - and `from_diagram`
        is what makes sure of that, so this checks the two agree.
        """
        board = Connect4([0, 1, 0, 1, 0, 1, 0])
        self.assertEqual(YELLOW, not board.turn, 'Yellow made the last move')
        self.assertEqual(win(YELLOW), board.outcome)

        with self.assertRaises(ValueError):
            Connect4.from_diagram(
                '.......\n.......\nY......\nY......\nY......\nY......'
            )  # Yellow to move, and already won


def main():
    unittest.main()


if __name__ == '__main__':
    main()
