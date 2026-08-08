"""
The board itself: its masks, its move generation, and the invariants a hand-built position has
to satisfy before the rest of the suite is allowed to trust it.

The masks are re-derived here from the geometry rather than imported and compared to themselves,
the way tests/connect4/test_bitboard.py re-derives its own. A test that builds its expectation
with the code under test can only show that code is consistent with itself, which is exactly the
property a wrong constant already has.
"""

import unittest

from games.tictactoe.board import IllegalMove, TicTacToe, is_win
from games.tictactoe.constants import (
    CELLS,
    CENTRE_FIRST,
    CROSS,
    FULL_BOARD,
    LINE,
    MARK_ICONS,
    NOUGHT,
    SIDE,
    WIN_MASKS,
    index,
    lines_through,
)
from tests.tictactoe.corpus import DRAWN_GAME, LAST_CELL_IS_ZERO, NOUGHT_WIN, ROW_WIN


def cells_of(mask: int):
    """The cell indices set in a mask, as a sorted list."""
    return sorted(cell for cell in range(CELLS) if mask >> cell & 1)


class TestGeometry(unittest.TestCase):
    def test_the_board_is_the_right_size(self):
        self.assertEqual(9, CELLS)
        self.assertEqual(0b111111111, FULL_BOARD)

    def test_cells_are_numbered_in_reading_order(self):
        """Row 0 at the top, so the numbers are the ones the board prints."""
        self.assertEqual(0, index(0, 0))
        self.assertEqual(2, index(2, 0))
        self.assertEqual(4, index(1, 1))
        self.assertEqual(8, index(2, 2))

    def test_there_are_eight_winning_lines(self):
        self.assertEqual(2 * SIDE + 2, len(WIN_MASKS))
        self.assertEqual(len(WIN_MASKS), len(set(WIN_MASKS)), 'a line is repeated')

    def test_every_winning_line_is_three_cells(self):
        for line in WIN_MASKS:
            self.assertEqual(LINE, len(cells_of(line)), format(line, '09b'))

    def test_the_winning_lines_are_the_ones_a_person_would_write_down(self):
        """
        The eight lines, spelled out independently of the comprehensions that build them. This is
        the whole win condition, so it is worth stating once in the least clever way available.
        """
        expected = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # columns
            [0, 4, 8], [2, 4, 6],             # diagonals
        ]
        self.assertEqual(sorted(expected), sorted(cells_of(line) for line in WIN_MASKS))

    def test_the_centre_is_on_four_lines_the_corners_three_the_edges_two(self):
        """What CENTRE_FIRST is derived from, checked before the ordering that depends on it."""
        self.assertEqual(4, lines_through(4))
        for corner in (0, 2, 6, 8):
            self.assertEqual(3, lines_through(corner), f'corner {corner}')
        for edge in (1, 3, 5, 7):
            self.assertEqual(2, lines_through(edge), f'edge {edge}')

    def test_move_ordering_is_centre_then_corners_then_edges(self):
        self.assertEqual((4, 0, 2, 6, 8, 1, 3, 5, 7), CENTRE_FIRST)

    def test_move_ordering_covers_every_cell_once(self):
        self.assertEqual(sorted(CENTRE_FIRST), list(range(CELLS)))


class TestIsWin(unittest.TestCase):
    def test_every_line_is_a_win(self):
        for line in WIN_MASKS:
            self.assertTrue(is_win(line), format(line, '09b'))

    def test_a_line_with_a_spare_mark_is_still_a_win(self):
        self.assertTrue(is_win(WIN_MASKS[0] | 1 << 4))

    def test_an_empty_board_is_not_a_win(self):
        self.assertFalse(is_win(0))

    def test_two_marks_of_a_line_are_not_a_win(self):
        for line in WIN_MASKS:
            for cell in cells_of(line):
                self.assertFalse(is_win(line ^ 1 << cell), f'{format(line, "09b")} less {cell}')

    def test_the_only_winning_masks_are_supersets_of_a_line(self):
        """
        Exhaustive over all 512 sets of cells, which is what makes it worth doing at all: there
        is no sampling here and no chance of a lucky pass.
        """
        for marks in range(1 << CELLS):
            expected = any(marks & line == line for line in WIN_MASKS)
            self.assertEqual(expected, is_win(marks), format(marks, '09b'))


class TestMoveGeneration(unittest.TestCase):
    def test_a_new_board_offers_every_cell(self):
        self.assertEqual(sorted(CENTRE_FIRST), sorted(TicTacToe().legal_moves))

    def test_generation_follows_the_centre_first_order(self):
        self.assertEqual(list(CENTRE_FIRST), list(TicTacToe().legal_moves))

    def test_a_played_cell_is_no_longer_offered(self):
        state = TicTacToe([4])
        self.assertNotIn(4, list(state.legal_moves))
        self.assertEqual(CELLS - 1, len(list(state.legal_moves)))

    def test_a_won_board_still_offers_moves(self):
        """The property chess does not have, and the reason `outcome` has to exist."""
        state = TicTacToe(ROW_WIN)
        self.assertIsNotNone(state.outcome)
        self.assertTrue(list(state.legal_moves))

    def test_cell_zero_alone_is_still_a_move(self):
        """
        The regression. Cell 0 is falsy, and `result` used to ask `any(legal_moves)`, so this
        position was reported as a draw with a move still on the board.
        """
        state = TicTacToe(LAST_CELL_IS_ZERO)
        self.assertEqual([0], list(state.legal_moves))
        self.assertIsNone(state.result)
        self.assertFalse(state.is_game_over)

    def test_an_illegal_cell_is_refused_by_the_constructor(self):
        with self.assertRaises(IllegalMove):
            TicTacToe([4, 4])
        with self.assertRaises(IllegalMove):
            TicTacToe([9])
        with self.assertRaises(IllegalMove):
            TicTacToe([-1])


class TestPlayingAndUndoing(unittest.TestCase):
    def test_a_move_marks_the_cell_for_the_player_who_made_it(self):
        state = TicTacToe([4])
        self.assertEqual(1 << 4, state.marks[CROSS])
        self.assertEqual(0, state.marks[NOUGHT])
        self.assertEqual(NOUGHT, state.turn)

    def test_undo_restores_the_position_exactly(self):
        state = TicTacToe([4, 0, 8])
        before = (state.marks[CROSS], state.marks[NOUGHT], state.turn, list(state.move_stack))

        state.make_move(2)
        state.unmake_move()

        after = (state.marks[CROSS], state.marks[NOUGHT], state.turn, list(state.move_stack))
        self.assertEqual(before, after)

    def test_a_whole_game_can_be_unwound(self):
        state = TicTacToe(DRAWN_GAME)
        for _ in DRAWN_GAME:
            state.unmake_move()

        self.assertEqual(0, state.marks[CROSS])
        self.assertEqual(0, state.marks[NOUGHT])
        self.assertEqual(CROSS, state.turn)
        self.assertEqual([], state.move_stack)


class TestCopy(unittest.TestCase):
    def test_a_copy_starts_equal(self):
        state = TicTacToe([4, 0])
        clone = state.copy()
        self.assertEqual(state.signature, clone.signature)
        self.assertEqual(state.turn, clone.turn)
        self.assertEqual(state.move_stack, clone.move_stack)

    def test_moving_a_copy_leaves_the_original_alone(self):
        state = TicTacToe([4, 0])
        before = state.signature

        clone = state.copy()
        clone.make_move(8)

        self.assertNotEqual(before, clone.signature)
        self.assertEqual(before, state.signature)

    def test_a_copy_can_be_unwound_as_far_as_the_original(self):
        """The move stack comes with the copy, which is what makes this true."""
        clone = TicTacToe([4, 0, 8]).copy()
        for _ in range(3):
            clone.unmake_move()
        self.assertEqual(TicTacToe().signature, clone.signature)


class TestSignature(unittest.TestCase):
    def test_the_signature_is_the_two_masks(self):
        self.assertEqual('0/0', TicTacToe().signature)
        self.assertEqual(f'{1 << 4}/0', TicTacToe([4]).signature)

    def test_positions_reached_differently_share_a_signature(self):
        """The board is the whole of the state, so the order the marks arrived in is not part."""
        self.assertEqual(TicTacToe([4, 0]).signature, TicTacToe([4, 0]).signature)
        self.assertNotEqual(TicTacToe([4, 0]).signature, TicTacToe([0, 4]).signature)


class TestDiagrams(unittest.TestCase):
    def test_a_diagram_round_trips_through_the_board(self):
        state = TicTacToe.from_diagram("""
            X.O
            .X.
            O..
        """)
        self.assertEqual(1 << 0 | 1 << 4, state.marks[CROSS])
        self.assertEqual(1 << 2 | 1 << 6, state.marks[NOUGHT])
        self.assertEqual(CROSS, state.turn)

    def test_the_player_to_move_is_derived_from_the_marks(self):
        even = TicTacToe.from_diagram('X.O\n...\n...')
        self.assertEqual(CROSS, even.turn)

        odd = TicTacToe.from_diagram('X..\n...\n...')
        self.assertEqual(NOUGHT, odd.turn)

    def test_a_diagram_of_the_wrong_shape_is_refused(self):
        with self.assertRaises(ValueError):
            TicTacToe.from_diagram('X.O\n...')
        with self.assertRaises(ValueError):
            TicTacToe.from_diagram('X.OO\n...\n...')

    def test_an_unrecognised_cell_is_refused(self):
        with self.assertRaises(ValueError):
            TicTacToe.from_diagram('X.?\n...\n...')

    def test_an_impossible_number_of_marks_is_refused(self):
        with self.assertRaises(ValueError):
            TicTacToe.from_diagram('X.X\nX..\n...')  # Crosses two moves ahead
        with self.assertRaises(ValueError):
            TicTacToe.from_diagram('OO.\n...\n...')  # Noughts ahead of Crosses

    def test_a_won_position_is_allowed_when_the_winner_has_just_moved(self):
        """The ordinary case, and the one the rejection below has to be careful not to catch."""
        state = TicTacToe.from_diagram("""
            XXX
            OO.
            ...
        """)
        self.assertEqual(NOUGHT, state.turn)
        self.assertEqual(CROSS, state.result.winner)

    def test_a_position_the_player_to_move_has_already_won_is_refused(self):
        """
        The invariant `outcome` rests on. Crosses have the top row and three marks answered by
        three, so it is somehow their turn again - which play cannot reach. `result` delegates to
        `outcome`, and `outcome` only ever tests the player who just moved, so nothing downstream
        would notice.
        """
        with self.assertRaises(ValueError):
            TicTacToe.from_diagram("""
                XXX
                OO.
                ..O
            """)


class TestRendering(unittest.TestCase):
    def test_an_empty_board_prints_its_cell_numbers(self):
        """The board is its own guide to what to type, which is why nothing labels it."""
        self.assertEqual(
            '\n[0][1][2]\n[3][4][5]\n[6][7][8]',
            str(TicTacToe()),
        )

    def test_marks_replace_the_numbers(self):
        rendered = str(TicTacToe([4, 0]))
        self.assertIn(MARK_ICONS[CROSS], rendered)
        self.assertIn(MARK_ICONS[NOUGHT], rendered)
        self.assertNotIn('[4]', rendered)
        self.assertNotIn('[0]', rendered)

    def test_the_top_row_prints_first(self):
        rendered = str(TicTacToe([0])).strip().splitlines()
        self.assertIn(MARK_ICONS[CROSS], rendered[0])
        self.assertEqual('[6][7][8]', rendered[2])

    def test_a_second_player_win_renders_both_marks(self):
        rendered = str(TicTacToe(NOUGHT_WIN))
        self.assertEqual(3, rendered.count(MARK_ICONS[NOUGHT]))
        self.assertEqual(3, rendered.count(MARK_ICONS[CROSS]))


def main():
    unittest.main()


if __name__ == '__main__':
    main()
