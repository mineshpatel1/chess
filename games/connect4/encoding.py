"""
How a Connect 4 position is shown to a neural network.

Two 6x7 planes and seven actions. The planes are **relative to the player to move** - plane 0 is
the mover's discs and plane 1 the opponent's - so a position and its colour-swap produce the same
tensor, the network learns one player's problem, and every game teaches it about both seats. The
action space is single and shared: a column is a column whoever is dropping into it.

Both of those are the contract in `games/base.py` rather than choices made here, and both are
choices the 2021 attempt got wrong - it gave each player its own block of nine outputs and trained
only one of them.

**Two binary planes rather than the one signed plane tic-tac-toe uses.** That is a departure, and a
deliberate one. The signed encoding won on tic-tac-toe by a clear margin, but its own docstring
says why that does not carry: the measurement was over an MLP on a 3x3 board, where the constraint
"a cell cannot be both mine and theirs" has to be learned from data and building it into the
representation is worth more than the extra freedom of separate channels. A convolution over two
channels is a different question - the two planes are what the filters slide over, and a filter
looking for three of mine with a gap wants "mine" as a channel it can weight independently of
"theirs". This is the standard AlphaZero encoding and the right starting assumption; it is also
cheap to re-measure once there is a training curve to compare, which is the only honest way to
settle it.

Nothing here tells the network whose turn it is and nothing needs to. With perspective-relative
planes the question never arises, and the disc counts would answer it anyway.

No symmetries are declared. Connect 4 has exactly one - the left-right mirror - and
`games/base.py` explains why the default claims none: augmenting *tells* the network the board is
symmetric rather than leaving it to notice, and on tic-tac-toe the measurement said it bought
nothing while costing eight times the examples. A convolutional trunk has translation equivariance
but not reflection, so the answer may genuinely differ here - `Encoder.symmetries` is where it
would go, and it deserves its own run rather than this conclusion inherited either way.
"""

from typing import List

from games.base import Encoder, Planes
from games.connect4.constants import COLS, ROWS
from games.connect4.bitboard import index


class Connect4Encoder(Encoder):
    PLANE_SHAPE = (2, ROWS, COLS)
    POLICY_SIZE = COLS

    @staticmethod
    def planes(state) -> Planes:
        """
        Two planes: the mover's discs, then the opponent's, bottom row first.

        Read off the bitboards rather than the printed board, so this cannot drift from the
        position if the rendering ever changes. Row 0 is the bottom of the board, matching
        `games/connect4/constants.py` - which way up it is does not matter to a network, but
        picking one and saying so is what stops a future reader guessing.
        """
        boards = (state.discs[state.turn], state.discs[not state.turn])
        return [
            [
                [1 if board >> index(column, row) & 1 else 0 for column in range(COLS)]
                for row in range(ROWS)
            ]
            for board in boards
        ]

    @staticmethod
    def action_index(move: int) -> int:
        """A move already is its own action: column 3 is action 3, for either player."""
        return move

    @staticmethod
    def action_move(index_: int) -> int:
        return index_
