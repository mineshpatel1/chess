"""
The constants and the bit tricks, tested without a board in sight.

Everything in games/connect4/constants.py is derived from ROWS, COLS and CONNECT, and the point
of deriving it is that a differently shaped board needs no other edit. A test that pastes the
same expressions back in would prove nothing about that, so the derivation tests here rebuild
the masks a different way - by iterating cells and setting bits - and check the two agree.
"""

import unittest

from games.connect4.bitboard import bit_count, cells, drops, index, landing_square
from games.connect4.constants import (
    BOTTOM_ROW,
    CENTRE_FIRST,
    COLS,
    COLUMN_MASKS,
    CONNECT,
    DIRECTIONS,
    FULL_BOARD,
    ROWS,
    SENTINEL_ROW,
    SIZE,
    STRIDE,
    TOP_ROW,
)


class TestConstants(unittest.TestCase):
    def test_the_layout_leaves_one_spare_cell_per_column(self):
        self.assertEqual(ROWS + 1, STRIDE)
        self.assertEqual(COLS * STRIDE, SIZE)
        self.assertEqual(49, SIZE)  # 7 columns of 6 playable cells and a sentinel

    def test_the_board_is_every_playable_cell_and_nothing_else(self):
        """Built here by iterating cells, rather than by the shift-and-or constants.py uses."""
        expected = 0
        for column in range(COLS):
            for row in range(ROWS):
                expected |= 1 << (column * STRIDE + row)

        self.assertEqual(expected, FULL_BOARD)
        self.assertEqual(ROWS * COLS, bit_count(FULL_BOARD))

    def test_the_sentinel_row_is_disjoint_from_the_board(self):
        """
        The invariant everything else rests on. A sentinel cell inside FULL_BOARD would let a
        run wrap from one column into the next, and would make both win detection and move
        generation lie in ways that surface much later as a baffling bug.
        """
        self.assertEqual(0, FULL_BOARD & SENTINEL_ROW)
        self.assertEqual(COLS, bit_count(SENTINEL_ROW))

    def test_column_masks_tile_the_board(self):
        self.assertEqual(COLS, len(COLUMN_MASKS))
        for column, mask in enumerate(COLUMN_MASKS):
            self.assertEqual(ROWS, bit_count(mask), f'column {column}')
            self.assertEqual(mask, mask & FULL_BOARD, f'column {column} leaves the board')

        union = 0
        for mask in COLUMN_MASKS:
            self.assertEqual(0, union & mask, 'columns overlap')
            union |= mask
        self.assertEqual(FULL_BOARD, union)

    def test_the_bottom_and_top_rows_hold_one_cell_per_column(self):
        for name, row in (('bottom', BOTTOM_ROW), ('top', TOP_ROW)):
            self.assertEqual(COLS, bit_count(row), name)
            self.assertEqual(row, row & FULL_BOARD, name)
            for mask in COLUMN_MASKS:
                self.assertEqual(1, bit_count(row & mask), name)

        self.assertEqual(0, BOTTOM_ROW & TOP_ROW)

    def test_the_directions_are_column_and_row_steps(self):
        """
        A straight line of constant (dcolumn, drow) is a constant step in index, which is why
        a diagonal needs no special handling - only a bigger constant.
        """
        for dcolumn, drow in ((0, 1), (1, 0), (1, 1), (1, -1)):
            self.assertIn(dcolumn * STRIDE + drow, DIRECTIONS)
        self.assertEqual(4, len(set(DIRECTIONS)))

    def test_the_move_order_runs_outwards_from_the_centre(self):
        self.assertEqual((3, 2, 4, 1, 5, 0, 6), CENTRE_FIRST)
        self.assertEqual(sorted(range(COLS)), sorted(CENTRE_FIRST))

        distances = [abs(2 * column - (COLS - 1)) for column in CENTRE_FIRST]
        self.assertEqual(sorted(distances), distances, 'not ordered by distance from centre')

    def test_a_differently_shaped_board_would_need_no_other_edit(self):
        """
        The masks re-derived for several board shapes, to check that the expressions in
        constants.py depend on ROWS and COLS and on nothing else. Changing the real constants
        is not something a test can do - the module is imported once - so this checks the
        arithmetic instead, against the shape the module was actually built with among others.
        """
        for rows, cols in ((6, 7), (4, 5), (8, 8), (3, 3)):
            stride = rows + 1
            columns = [((1 << rows) - 1) << (column * stride) for column in range(cols)]
            board = 0
            for mask in columns:
                board |= mask

            self.assertEqual(rows * cols, bit_count(board), f'{rows}x{cols}')
            sentinels = 0
            for column in range(cols):
                sentinels |= 1 << (column * stride + rows)
            self.assertEqual(0, board & sentinels, f'{rows}x{cols} sentinels are playable')

            if (rows, cols) == (ROWS, COLS):
                self.assertEqual(FULL_BOARD, board)
                self.assertEqual(COLUMN_MASKS, columns)


class TestIndexing(unittest.TestCase):
    def test_index_counts_up_each_column(self):
        self.assertEqual(0, index(0, 0))
        self.assertEqual(5, index(0, ROWS - 1))
        self.assertEqual(STRIDE, index(1, 0))
        self.assertEqual(SIZE - STRIDE + ROWS - 1, index(COLS - 1, ROWS - 1))

    def test_cells_yields_the_indices_that_are_set(self):
        self.assertEqual([], list(cells(0)))
        self.assertEqual([0, 5, 41], list(cells((1 << 0) | (1 << 5) | (1 << 41))))
        self.assertEqual(list(range(SIZE)), list(cells((1 << SIZE) - 1)))

    def test_bit_count_counts_bits(self):
        self.assertEqual(0, bit_count(0))
        self.assertEqual(1, bit_count(1 << 30))
        self.assertEqual(SIZE, bit_count((1 << SIZE) - 1))


class TestDrops(unittest.TestCase):
    """
    The carry trick, which is the whole of move generation.

    Adding BOTTOM_ROW rings a carry up every column at once; it settles in the first gap, and
    in a full column it settles in the sentinel, where FULL_BOARD discards it. That last part
    is the one worth testing hardest: without the sentinel the carry would spill into the
    bottom cell of the next column and offer a move that is not there.
    """

    def test_an_empty_board_lands_every_disc_on_the_bottom_row(self):
        self.assertEqual(BOTTOM_ROW, drops(0))

    def test_a_disc_lands_directly_on_the_one_below_it(self):
        occupied = 1 << index(3, 0)
        self.assertEqual(1 << index(3, 1), drops(occupied) & COLUMN_MASKS[3])

    def test_a_full_column_offers_nothing(self):
        occupied = COLUMN_MASKS[0]
        self.assertEqual(0, drops(occupied) & COLUMN_MASKS[0])
        self.assertEqual(0, landing_square(occupied, 0))

    def test_a_full_column_does_not_leak_into_its_neighbour(self):
        """
        The sentinel's reason to exist, stated as a test. Column 0 is full and column 1 empty,
        so the carry out of column 0 has somewhere to go wrong; it must still land column 1 on
        its own bottom cell and nowhere else.
        """
        occupied = COLUMN_MASKS[0]
        landing = drops(occupied)

        self.assertEqual(0, landing & COLUMN_MASKS[0])
        self.assertEqual(1 << index(1, 0), landing & COLUMN_MASKS[1])
        self.assertEqual(0, landing & SENTINEL_ROW, 'the carry escaped into a sentinel')
        self.assertEqual(COLS - 1, bit_count(landing))

    def test_every_column_full_offers_nothing_at_all(self):
        self.assertEqual(0, drops(FULL_BOARD))

    def test_landing_squares_stack_a_column_from_the_bottom(self):
        occupied = 0
        for row in range(ROWS):
            bit = landing_square(occupied, 4)
            self.assertEqual(1 << index(4, row), bit, f'row {row}')
            occupied |= bit

        self.assertEqual(COLUMN_MASKS[4], occupied)
        self.assertEqual(0, landing_square(occupied, 4))


class TestConnectLength(unittest.TestCase):
    def test_a_win_is_four_in_a_line(self):
        """Guards the tests below, which are written in terms of CONNECT being 4."""
        self.assertEqual(4, CONNECT)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
