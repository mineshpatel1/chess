"""
The board: generation, make, unmake, copy, and the diagram loader.

The heart of this module is the fuzz test. Connect 4's invariants are all of the form "this can
never happen", and the cheapest way to believe a "never" is to play a great many random games
and check it after every single move. Thousands of playouts run in well under a second here,
which is a luxury the chess suite does not have.
"""

import random
import unittest

from games.connect4.bitboard import bit_count, index
from games.connect4.board import Connect4, IllegalMove
from games.connect4.constants import (
    CENTRE_FIRST,
    COLS,
    COLUMN_MASKS,
    FULL_BOARD,
    RED,
    ROWS,
    SENTINEL_ROW,
    YELLOW,
)

PLAYOUTS = 500


def playout(rng: random.Random) -> Connect4:
    """A board part-way through a random game, stopping at a win or a full board."""
    board = Connect4()
    while not board.is_game_over:
        board.make_move(rng.choice(list(board.legal_moves)))
    return board


class TestInvariants(unittest.TestCase):
    """
    The three things that must be true of every position this engine can ever hold.

    A set sentinel bit is the worst of them: it does not break anything where it happens, it
    makes move generation and win detection quietly wrong somewhere else, several moves later.
    """

    def test_a_random_game_never_sets_a_sentinel_bit(self):
        rng = random.Random(0)
        for _ in range(PLAYOUTS):
            board = Connect4()
            while not board.is_game_over:
                board.make_move(rng.choice(list(board.legal_moves)))
                occupied = board.occupied
                self.assertEqual(0, occupied & SENTINEL_ROW, f'sentinel set{board}')
                self.assertEqual(0, occupied & ~FULL_BOARD, f'disc off the board{board}')
                self.assertEqual(0, board.discs[YELLOW] & board.discs[RED], f'overlap{board}')

    def test_discs_alternate_and_are_never_lost(self):
        rng = random.Random(1)
        for _ in range(PLAYOUTS):
            board = Connect4()
            while not board.is_game_over:
                board.make_move(rng.choice(list(board.legal_moves)))
                yellow, red = bit_count(board.discs[YELLOW]), bit_count(board.discs[RED])

                self.assertEqual(len(board.move_stack), yellow + red)
                self.assertIn(yellow - red, (0, 1), 'a player moved twice')
                self.assertEqual(YELLOW if yellow == red else RED, board.turn)

    def test_unmaking_a_whole_game_empties_the_board(self):
        """
        Undo has to survive a full game, not one move at a time. The search walks a single
        board up and down the tree rather than copying at every node, so an unmake that
        restores nearly everything corrupts every branch searched after it.
        """
        rng = random.Random(2)
        for _ in range(PLAYOUTS):
            board = playout(rng)
            moves = len(board.move_stack)

            for _ in range(moves):
                board.unmake_move()

            self.assertEqual(0, board.discs[YELLOW])
            self.assertEqual(0, board.discs[RED])
            self.assertEqual(YELLOW, board.turn)
            self.assertEqual([], board.move_stack)

    def test_every_move_is_reversible_from_every_position(self):
        rng = random.Random(3)
        for _ in range(50):
            board = Connect4()
            while not board.is_game_over:
                before = (board.discs[YELLOW], board.discs[RED], board.turn)
                for column in list(board.legal_moves):
                    board.make_move(column)
                    board.unmake_move()
                    after = (board.discs[YELLOW], board.discs[RED], board.turn)
                    self.assertEqual(before, after, f'column {column} did not undo{board}')
                board.make_move(rng.choice(list(board.legal_moves)))


class TestMoveGeneration(unittest.TestCase):
    def test_an_empty_board_offers_every_column_centre_first(self):
        self.assertEqual(list(CENTRE_FIRST), list(Connect4().legal_moves))

    def test_a_full_column_drops_out_of_the_move_list(self):
        board = Connect4([0] * ROWS)
        self.assertNotIn(0, board.legal_moves)
        self.assertEqual(COLS - 1, len(list(board.legal_moves)))

    def test_discs_stack_upwards(self):
        board = Connect4()
        for row in range(ROWS):
            board.make_move(2)
            self.assertTrue(board.occupied & (1 << index(2, row)), f'row {row}')
        self.assertEqual(COLUMN_MASKS[2], board.occupied)

    def test_a_full_board_offers_nothing(self):
        board = Connect4()
        for column in range(COLS):
            for _ in range(ROWS):
                board.make_move(column)

        self.assertEqual(FULL_BOARD, board.occupied)
        self.assertEqual([], list(board.legal_moves))

    def test_playing_a_full_column_is_rejected_by_the_constructor(self):
        with self.assertRaises(IllegalMove):
            Connect4([0] * (ROWS + 1))

    def test_columns_played_reads_the_move_stack_back(self):
        columns = [3, 3, 2, 4, 1, 0, 3, 6, 6]
        self.assertEqual(columns, Connect4(columns).columns_played)


class TestCopy(unittest.TestCase):
    def test_a_copy_is_the_same_position(self):
        board = Connect4([3, 3, 4, 2])
        clone = board.copy()

        self.assertEqual(board.signature, clone.signature)
        self.assertEqual(board.turn, clone.turn)
        self.assertEqual(list(board.legal_moves), list(clone.legal_moves))

    def test_moving_a_copy_leaves_the_original_alone(self):
        board = Connect4([3, 3, 4, 2])
        before = board.signature
        clone = board.copy()

        for column in list(clone.legal_moves)[:3]:
            clone.make_move(column)

        self.assertNotEqual(before, clone.signature)
        self.assertEqual(before, board.signature)

    def test_a_copy_does_not_share_the_move_stack(self):
        """
        The conformance suite cannot catch this one: it compares signature, turn and legal
        moves, and a shared stack changes none of them until something unmakes. `alpha_beta`
        searches copies and unmakes on them, so a shared list would reach back into the
        original mid-search.
        """
        board = Connect4([3, 3, 4])
        clone = board.copy()
        clone.make_move(1)
        clone.unmake_move()
        clone.unmake_move()

        self.assertEqual(3, len(board.move_stack))
        self.assertEqual([3, 3, 4], board.columns_played)


class TestSignature(unittest.TestCase):
    def test_the_same_cells_in_different_hands_are_different_positions(self):
        yellow_first = Connect4([0, 1])
        red_first = Connect4([1, 0])

        self.assertEqual(yellow_first.occupied, red_first.occupied)
        self.assertNotEqual(yellow_first.signature, red_first.signature)

    def test_the_separator_keeps_the_two_boards_apart(self):
        """Without it, `f'{1}{23}'` and `f'{12}{3}'` would be the same string."""
        self.assertIn('/', Connect4().signature)


class TestDiagrams(unittest.TestCase):
    def test_a_diagram_describes_the_same_position_as_the_moves_that_reach_it(self):
        played = Connect4([3, 3, 2, 4, 1])
        drawn = Connect4.from_diagram('''
            .......
            .......
            .......
            .......
            ...R...
            .YYYR..
        ''')

        self.assertEqual(played.signature, drawn.signature)
        self.assertEqual(played.turn, drawn.turn)

    def test_a_diagram_ignores_its_own_indentation(self):
        self.assertEqual(Connect4().signature, Connect4.from_diagram('''
            .......
            .......
            .......
            .......
            .......
            .......
        ''').signature)

    def test_a_diagram_infers_whose_turn_it_is(self):
        self.assertEqual(YELLOW, Connect4.from_diagram(
            '.......\n.......\n.......\n.......\n...R...\n...Y...').turn)
        self.assertEqual(RED, Connect4.from_diagram(
            '.......\n.......\n.......\n.......\n.......\n...Y...').turn)

    def test_a_diagram_that_play_could_not_reach_is_rejected(self):
        """
        `outcome` tests only the board of the player who just moved, which is sound precisely
        because these positions cannot be built. Without the check the saving would be a bug:
        `result` delegates to `outcome`, so the conformance suite would agree with it and see
        nothing wrong.
        """
        floating = '.......\n.......\n.......\n.......\n...Y...\n.......'
        too_many = '.......\n.......\n.......\n.......\n.......\nYYY....'
        mover_won = '.......\n.......\n.......\n.......\nRRRR...\nYYYY...'

        for diagram, reason in (
            (floating, 'a disc floating above a gap'),
            (too_many, 'three Yellow discs and no Red'),
            (mover_won, 'the player to move has already won'),
        ):
            with self.assertRaises(ValueError, msg=reason):
                Connect4.from_diagram(diagram)

    def test_a_diagram_of_the_wrong_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            Connect4.from_diagram('.......\n.......')
        with self.assertRaises(ValueError):
            Connect4.from_diagram('....\n....\n....\n....\n....\n....')
        with self.assertRaises(ValueError):
            Connect4.from_diagram('.......\n.......\n.......\n.......\n.......\n...X...')


class TestRendering(unittest.TestCase):
    def test_the_board_prints_a_row_per_row_and_a_column_header(self):
        lines = str(Connect4()).strip().splitlines()
        self.assertEqual(ROWS + 1, len(lines))
        self.assertEqual(list(range(COLS)), [int(c) for c in lines[-1].split()])

    def test_the_bottom_row_of_the_picture_is_row_zero(self):
        board = Connect4([0])
        lines = str(board).strip().splitlines()
        self.assertIn('●', lines[ROWS - 1])
        self.assertNotIn('●', lines[0])


def main():
    unittest.main()


if __name__ == '__main__':
    main()
