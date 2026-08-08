"""
What the network is shown, and the two properties that make it learnable at all.

Both were wrong in the 2021 attempt, and both fail silently: an encoding mistake does not raise,
it produces a network that trains and cannot play. So they are pinned here rather than trusted.

Needs no PyTorch - the encoder deals in nested lists of ints, which is the whole point of keeping
tensors out of `games/`.
"""

import unittest

from games.tictactoe.board import TicTacToe
from games.tictactoe.constants import CELLS, CROSS, NOUGHT, SIDE, index
from games.tictactoe.encoding import TRANSFORMS, TicTacToeEncoder as Encoder


class TestShape(unittest.TestCase):
    def test_the_board_is_one_signed_plane(self):
        """
        One plane, not two. Two binary planes - "mine" and "theirs" - is what AlphaZero uses and
        what this was first; one signed plane measurably beat it here, at every checkpoint of an
        otherwise identical run. See the docstring of `TicTacToeEncoder.planes`.
        """
        self.assertEqual((1, SIDE, SIDE), Encoder.PLANE_SHAPE)

    def test_there_is_one_action_per_cell_and_not_two(self):
        """
        Nine, not eighteen. The 2021 encoding gave each player its own block of nine outputs,
        wrote every training target into the first block and read the second player's moves out
        of the second - so the second player's half of the policy head was never trained and
        stayed at its random initialisation. One shared action space has no second block.
        """
        self.assertEqual(CELLS, Encoder.POLICY_SIZE)

    def test_the_game_declares_its_encoder(self):
        self.assertIs(Encoder, TicTacToe.ENCODER)


class TestPlanes(unittest.TestCase):
    def test_the_mover_is_positive_and_the_opponent_negative(self):
        state = TicTacToe([4, 0])  # Crosses centre, Noughts corner, Crosses to move
        (board,) = Encoder.planes(state)

        self.assertEqual(1, board[1][1], 'the mover holds the centre')
        self.assertEqual(-1, board[0][0], 'the opponent holds the corner')
        self.assertEqual(0, board[0][1], 'and that cell is empty')

    def test_a_cell_cannot_be_both_players_at_once(self):
        """
        The constraint the signed encoding gets for free and two planes have to learn: one cell,
        one value. That is the leading guess for why it trains faster on a board this small.
        """
        for state in (TicTacToe(), TicTacToe([4, 0]), TicTacToe([4, 0, 8, 2])):
            (board,) = Encoder.planes(state)
            for row in board:
                for cell in row:
                    self.assertIn(cell, (-1, 0, 1))

    def test_the_same_shape_looks_the_same_to_whoever_is_playing_it(self):
        """
        Perspective-relative, which is what lets one network learn both seats. A player holding
        the centre against a corner sees the identical tensor whether it is Crosses or Noughts.
        """
        crosses_to_move = TicTacToe([4, 0])  # Crosses hold the centre, Noughts a corner

        # Build the mirror explicitly: Noughts holding the centre against a Crosses corner.
        mirror = TicTacToe()
        mirror.marks[NOUGHT] = 1 << index(1, 1)
        mirror.marks[CROSS] = 1 << index(0, 0)
        mirror.turn = NOUGHT

        self.assertEqual(Encoder.planes(crosses_to_move), Encoder.planes(mirror))

    def test_an_empty_board_is_all_zeroes(self):
        self.assertEqual([[[0] * SIDE] * SIDE], Encoder.planes(TicTacToe()))

    def test_planes_are_read_in_the_same_order_the_board_prints(self):
        state = TicTacToe([2])  # Crosses take the top right, so Noughts are to move
        (board,) = Encoder.planes(state)
        self.assertEqual(-1, board[0][2], 'cell 2 is row 0, column 2, and belongs to the opponent')


class TestActions(unittest.TestCase):
    def test_a_move_round_trips_through_its_action(self):
        for state in (TicTacToe(), TicTacToe([4, 0]), TicTacToe([4, 0, 8, 2])):
            for move in state.legal_moves:
                self.assertEqual(move, Encoder.action_move(Encoder.action_index(move)))

    def test_actions_cover_the_whole_policy_exactly_once(self):
        indices = sorted(Encoder.action_index(move) for move in TicTacToe().legal_moves)
        self.assertEqual(list(range(Encoder.POLICY_SIZE)), indices)

    def test_both_players_use_the_same_actions(self):
        """A cell is a cell whoever plays it, which is the whole of the fix."""
        crosses = TicTacToe([])       # Crosses to move
        noughts = TicTacToe([4])      # Noughts to move
        shared = {Encoder.action_index(m) for m in crosses.legal_moves}
        self.assertTrue({Encoder.action_index(m) for m in noughts.legal_moves} <= shared)


class TestSymmetries(unittest.TestCase):
    def test_there_are_eight(self):
        self.assertEqual(8, len(TRANSFORMS))
        self.assertEqual(8, len(set(TRANSFORMS)))

    def test_each_is_a_permutation_of_the_cells(self):
        for transform in TRANSFORMS:
            self.assertEqual(sorted(transform), list(range(CELLS)))

    def test_the_identity_is_among_them(self):
        self.assertIn(tuple(range(CELLS)), TRANSFORMS)

    def test_they_are_derived_rather_than_typed(self):
        """
        Re-derived here independently of encoding.py, the way test_board.py re-derives the win
        masks. A permutation written by hand is one that can be wrong in a way nothing catches -
        and a wrong symmetry does not crash, it teaches the network the mirror image of the game.
        """
        expected = set()
        for reflect in (False, True):
            for turns in range(4):
                permutation = [0] * CELLS
                for row in range(SIDE):
                    for column in range(SIDE):
                        r, c = (row, SIDE - 1 - column) if reflect else (row, column)
                        for _ in range(turns):
                            r, c = c, SIDE - 1 - r
                        permutation[index(column, row)] = index(c, r)
                expected.add(tuple(permutation))

        self.assertEqual(expected, set(TRANSFORMS))

    def test_a_symmetry_moves_the_board_and_the_policy_together(self):
        """
        The property that matters. Applying a transform to one and not the other trains the
        network to answer a different position from the one it was shown.
        """
        state = TicTacToe([0, 4])  # Crosses corner, Noughts centre, Crosses to move
        planes = Encoder.planes(state)
        policy = [1.0 if cell == 0 else 0.0 for cell in range(CELLS)]  # all mass on cell 0

        for moved, spun in Encoder.symmetries(planes, policy):
            hot = spun.index(1.0)
            # `== 1` rather than truthiness: the opponent's cells are -1, which is also truthy.
            mine = [r * SIDE + c for r in range(SIDE) for c in range(SIDE) if moved[0][r][c] == 1]
            self.assertEqual([hot], mine, 'the policy did not follow the mark it belongs to')

    def test_a_symmetric_position_maps_to_itself(self):
        """The centre is fixed by every transform, so all eight variants are the same board."""
        planes = Encoder.planes(TicTacToe([4]))
        variants = {str(moved) for moved, _ in Encoder.symmetries(planes, [0.0] * CELLS)}
        self.assertEqual(1, len(variants))

    def test_a_corner_reaches_all_four_corners(self):
        planes = Encoder.planes(TicTacToe([0, 4]))
        policy = [1.0 if cell == 0 else 0.0 for cell in range(CELLS)]
        landed = {spun.index(1.0) for _, spun in Encoder.symmetries(planes, policy)}
        self.assertEqual({0, 2, 6, 8}, landed)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
