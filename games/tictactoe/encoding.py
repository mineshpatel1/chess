"""
How a tic-tac-toe position is shown to a neural network.

Two planes and nine actions, which is as small as this gets. What matters is not the size but the
two choices behind it, both of which a previous attempt in this project got wrong:

**The planes are relative to the player to move.** Plane 0 is the mover's marks and plane 1 the
opponent's, so a position and its colour-swap produce the same tensor. The network learns one
player's problem and every game teaches it about both seats. `TicTacToe.model_input` in the 2021
branch did this correctly - it was the only part of that encoding that was right.

**There are nine actions, not eighteen.** A cell is a cell whoever is playing it. The 2021 code
gave each player its own block of nine outputs, wrote every training target into the first block
and read the second player's moves out of the second block, so the second player's half of the
policy head was never trained at all and stayed at its random initialisation. One shared action
space cannot develop that fault, because there is no second block to disagree with.

Note that nothing here tells the network whose turn it is in absolute terms, and nothing needs
to: with perspective-relative planes the question never arises. The mark counts would answer it
anyway - the mover has as many marks as the opponent when Crosses is to move, and one fewer when
Noughts is - but no part of the network is asked to care.
"""

from typing import Iterator, Tuple

from games.base import Encoder, Planes, Policy
from games.tictactoe.constants import CELLS, SIDE, index


def _dihedral() -> Tuple[Tuple[int, ...], ...]:
    """
    The eight symmetries of a square, as permutations of cell indices.

    Derived by composing a quarter turn and a reflection rather than typed out, for the reason
    constants.py derives its win masks: a permutation written by hand is a permutation that can
    be wrong in a way nothing catches, and this one silently mistrains a network rather than
    failing. tests/tictactoe/test_encoding.py re-derives them independently.

    Each permutation maps a source cell to its destination, so a transform is applied the same
    way to a board and to a policy over that board - which is what keeps the two in step.
    """
    def turn(row: int, column: int) -> Tuple[int, int]:
        """A quarter turn clockwise."""
        return column, SIDE - 1 - row

    def flip(row: int, column: int) -> Tuple[int, int]:
        return row, SIDE - 1 - column

    transforms = []
    for reflected in (False, True):
        for quarter_turns in range(4):
            permutation = [0] * CELLS
            for row in range(SIDE):
                for column in range(SIDE):
                    r, c = (flip(row, column) if reflected else (row, column))
                    for _ in range(quarter_turns):
                        r, c = turn(r, c)
                    permutation[index(column, row)] = index(c, r)
            transforms.append(tuple(permutation))

    assert len(set(transforms)) == 8, 'the dihedral group of a square has exactly eight elements'
    return tuple(transforms)


TRANSFORMS = _dihedral()


class TicTacToeEncoder(Encoder):
    PLANE_SHAPE = (1, SIDE, SIDE)
    POLICY_SIZE = CELLS

    @staticmethod
    def planes(state) -> Planes:
        """
        One 3x3 plane, signed: +1 where the mover has a mark, -1 where the opponent has, 0 empty.

        The obvious alternative, and what this was first, is two binary planes - "mine" and
        "theirs" - which is what AlphaZero uses and what every board-game implementation reaches
        for. Measured over 150 generations, identical in every other respect, one signed plane
        won at every checkpoint:

            generation        50       100      150
            signed, 9 in    96.97%   97.54%   97.65%
            2-plane, 18 in  93.94%   96.22%   96.46%

        The two-plane version at generation 50 was where the signed one started, and at 150 it
        had still not caught up. Both runs were re-run and reproduced to the digit, so this is an
        effect rather than a seed.

        Why is worth a guess, and it is only a guess: two planes give the first layer eighteen
        inputs of which exactly nine are ever non-zero, and the fact that a cell cannot be both
        mine and theirs has to be *learned* from data. One signed cell has that constraint built
        into the representation, and on a board this small - with about six thousand parameters
        to spend - not having to learn it is apparently worth more than the extra freedom of
        separate channels.

        That argument does not obviously survive a bigger board with more piece types, where
        "signed" stops being expressive enough. This is the tic-tac-toe encoder; a game whose
        pieces need more than one bit should measure it again rather than inherit the conclusion.

        Read off `state.marks` rather than the printed board, so this cannot drift from the
        position if the rendering ever changes.
        """
        mine, theirs = state.marks[state.turn], state.marks[not state.turn]
        return [[
            [
                1 if mine >> index(column, row) & 1
                else -1 if theirs >> index(column, row) & 1
                else 0
                for column in range(SIDE)
            ]
            for row in range(SIDE)
        ]]

    @staticmethod
    def action_index(move: int) -> int:
        """A move already is its own action: cell 4 is action 4, for either player."""
        return move

    @staticmethod
    def action_move(index_: int) -> int:
        return index_

    @staticmethod
    def symmetries(planes: Planes, policy: Policy) -> Iterator[Tuple[Planes, Policy]]:
        """
        The same position seen eight ways, which is eight training examples for the price of one.

        Worth having here beyond the free data: a network shown only the positions self-play
        happens to reach can learn that the top-left corner is different from the bottom-right.
        It is not, and the symmetries say so.

        Duplicates are not filtered. A symmetric position - the empty board, or a lone centre
        mark - maps to itself under several transforms and is emitted several times, which
        weights it more heavily. That is a real cost and a small one: it is a handful of
        positions, and the alternative is deduplicating on every example forever.
        """
        for permutation in TRANSFORMS:
            moved: Planes = []
            for plane in planes:
                flat = [plane[row][column] for row in range(SIDE) for column in range(SIDE)]
                turned = [0] * CELLS
                for source, destination in enumerate(permutation):
                    turned[destination] = flat[source]
                moved.append([turned[row * SIDE:(row + 1) * SIDE] for row in range(SIDE)])

            spun: Policy = [0.0] * CELLS
            for source, destination in enumerate(permutation):
                spun[destination] = policy[source]

            yield moved, spun
