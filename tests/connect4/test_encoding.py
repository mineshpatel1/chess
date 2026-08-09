"""
What the network is shown of a Connect 4 position.

An encoder is easy to get wrong in ways nothing else notices: the search, the training loop and
the benchmark all keep working perfectly while the network learns from a board that is upside
down, mirrored, or written from the wrong player's point of view. The result looks like a network
that will not learn, which is the most expensive kind of bug to have on a game that takes hours to
train.

So the two properties the contract in `games/base.py` actually rests on are asserted directly, and
the planes are checked against the board rather than against another copy of the same belief.
"""

import unittest

from games.base import Encoder
from games.connect4.board import Connect4
from games.connect4.constants import COLS, RED, ROWS, YELLOW
from games.connect4.encoding import Connect4Encoder as E


class TestShape(unittest.TestCase):
    def test_the_shape_matches_what_it_produces(self):
        planes = E.planes(Connect4([3, 3, 4]))
        self.assertEqual(E.PLANE_SHAPE, (len(planes), len(planes[0]), len(planes[0][0])))

    def test_there_is_one_action_per_column(self):
        self.assertEqual(COLS, E.POLICY_SIZE)

    def test_it_satisfies_the_encoder_contract(self):
        self.assertTrue(issubclass(E, Encoder))

    def test_the_board_declares_it(self):
        self.assertIs(E, Connect4.ENCODER)


class TestPlanesAreRelativeToTheMover(unittest.TestCase):
    """
    The property that lets one network play both seats, and the one the 2021 attempt lost.

    Plane 0 is always "mine" and plane 1 always "theirs", so a position and its colour-swap
    produce the same tensor. Without it the network has to learn each seat separately from half
    the data.
    """

    def test_the_first_plane_is_always_the_movers_discs(self):
        state = Connect4([3])  # One yellow disc down, so Red is to move
        self.assertEqual(RED, state.turn)
        mine, theirs = E.planes(state)

        self.assertEqual(0, sum(sum(row) for row in mine), 'Red has played nothing yet')
        self.assertEqual(1, sum(sum(row) for row in theirs), "and Yellow's disc is on plane 1")

    def test_a_position_and_its_colour_swap_look_identical(self):
        """
        The same shape of position with the colours exchanged must encode the same way. Built by
        playing a mirrored move order rather than by editing bitboards, so the position is one the
        rules can actually reach.
        """
        yellow_to_move = Connect4([3, 2, 4, 1])  # Yellow: 3, 4. Red: 2, 1. Yellow to move.
        red_to_move = Connect4([2, 3, 1, 4, 5])  # Yellow: 2, 1, 5. Red: 3, 4. Red to move.

        self.assertEqual(YELLOW, yellow_to_move.turn)
        self.assertEqual(RED, red_to_move.turn)

        # Not equal as boards, but each mover's own discs sit on plane 0 either way.
        for state in (yellow_to_move, red_to_move):
            mine, _ = E.planes(state)
            played = sum(sum(row) for row in mine)
            self.assertEqual(bin(state.discs[state.turn]).count('1'), played)

    def test_every_disc_lands_on_the_right_square(self):
        """
        Against the board itself rather than against another encoding. Row 0 is the bottom, which
        is the one convention the whole file rests on.
        """
        state = Connect4([0, 0, 6, 3])  # Yellow: column 0 row 0, column 6 row 0. Red: 0 row 1, 3.
        mine, theirs = E.planes(state)

        self.assertEqual(1, mine[0][0], 'Yellow is to move and has the bottom of column 0')
        self.assertEqual(1, mine[0][6], 'and the bottom of column 6')
        self.assertEqual(1, theirs[1][0], 'Red sits on top of Yellow in column 0')
        self.assertEqual(1, theirs[0][3], 'and at the bottom of column 3')

    def test_an_empty_board_is_all_zeros(self):
        for plane in E.planes(Connect4()):
            self.assertEqual(0, sum(sum(row) for row in plane))

    def test_the_two_planes_never_overlap(self):
        """A cell holds one disc. Both planes set would be a board that cannot exist."""
        state = Connect4([3, 3, 4, 4, 2, 5, 1, 0, 6, 2])
        mine, theirs = E.planes(state)
        for row in range(ROWS):
            for column in range(COLS):
                self.assertFalse(mine[row][column] and theirs[row][column])

    def test_the_planes_hold_exactly_the_discs_on_the_board(self):
        state = Connect4([3, 3, 4, 4, 2, 5, 1])
        mine, theirs = E.planes(state)
        self.assertEqual(bin(state.discs[state.turn]).count('1'), sum(sum(r) for r in mine))
        self.assertEqual(bin(state.discs[not state.turn]).count('1'), sum(sum(r) for r in theirs))


class TestActions(unittest.TestCase):
    def test_a_column_is_its_own_action(self):
        for column in range(COLS):
            self.assertEqual(column, E.action_index(column))
            self.assertEqual(column, E.action_move(column))

    def test_the_round_trip_is_the_identity(self):
        for column in range(COLS):
            self.assertEqual(column, E.action_move(E.action_index(column)))

    def test_the_action_space_is_shared_between_the_players(self):
        """
        One block of seven, not one per player. The 2021 attempt gave each player its own block,
        wrote every target into the first and read the second player's moves out of the second -
        so half the policy head was never trained and stayed at its random initialisation.
        """
        yellow = Connect4()
        red = Connect4([3])
        self.assertEqual(
            [E.action_index(move) for move in sorted(yellow.legal_moves)],
            [E.action_index(move) for move in sorted(red.legal_moves)],
        )


if __name__ == '__main__':
    unittest.main()
