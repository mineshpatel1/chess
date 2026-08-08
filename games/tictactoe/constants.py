"""
The shape of a tic-tac-toe board, and every mask derived from it.

A cell is addressed as `index = row * SIDE + column`, with row 0 at the *top*, so the indices
run in reading order and are what the board prints in its empty squares:

     0 1 2
     3 4 5
     6 7 8

Connect 4 numbers its rows from the bottom because discs fall; nothing falls here, so reading
order is the only ordering a person has to be told about, and a move is just the number they
can already see.

There is no bitboard module beside this one, which is the whole of what makes this the simplest
game in the project. Connect 4 needs sentinel padding because it walks along lines by adding a
constant to an index and would otherwise wrap from one column into the next; a 3x3 board has
eight lines in total, so they are simply enumerated once here and tested with a single `&`.

Nothing is written down that can be derived. SIDE is the only edit a differently shaped board
would need - WIN_MASKS, FULL_BOARD and the move ordering all follow from it, and
tests/tictactoe/test_board.py re-derives them independently to keep that true.
"""

from typing import Iterable, Tuple

from games.base import Player

SIDE = 3
CELLS = SIDE * SIDE
LINE = SIDE  # Marks in a row that wins

# True is the first player, the one a positive evaluation favours, matching Player in games.base,
# WHITE in chess and YELLOW in Connect 4. Crosses conventionally go first.
Mark = Player
CROSS: Mark = True
NOUGHT: Mark = False

MARKS: Tuple[Mark, Mark] = (CROSS, NOUGHT)
MARK_ICONS = {CROSS: '✕', NOUGHT: '○'}
MARK_NAMES = {CROSS: 'Crosses', NOUGHT: 'Noughts'}


def index(column: int, row: int) -> int:
    """The index of a cell, row 0 being the top."""
    return row * SIDE + column


def mask(cells: Iterable[int]) -> int:
    """A set of cells as a bitboard.

    `sum` rather than `|` because the cells handed to it are always distinct, which makes the
    two the same and reads better - the same choice games/connect4/constants.py makes.
    """
    return sum(1 << cell for cell in cells)


# The eight ways to win, built rather than pasted. A line in direction -d covers the same cells
# as one in +d, so three rows, three columns and two diagonals is all of them.
ROW_MASKS: Tuple[int, ...] = tuple(
    mask(index(column, row) for column in range(SIDE)) for row in range(SIDE)
)
COLUMN_MASKS: Tuple[int, ...] = tuple(
    mask(index(column, row) for row in range(SIDE)) for column in range(SIDE)
)
DIAGONAL_MASKS: Tuple[int, ...] = (
    mask(index(step, step) for step in range(SIDE)),
    mask(index(SIDE - 1 - step, step) for step in range(SIDE)),
)
WIN_MASKS: Tuple[int, ...] = ROW_MASKS + COLUMN_MASKS + DIAGONAL_MASKS

FULL_BOARD = (1 << CELLS) - 1


def lines_through(cell: int) -> int:
    """How many of the eight winning lines pass through a cell."""
    return sum(1 for line in WIN_MASKS if line >> cell & 1)


# Cells in order of how many lines run through them: the centre (four), then the corners (three),
# then the edges (two).
#
# The order is deliberate and load-bearing, for the reasons Connect 4's CENTRE_FIRST is.
# `ai.search` iterates moves in generation order and breaks ties by it, so putting the cells that
# do the most work first is what makes alpha-beta's pruning pay, and it also means a search with
# nothing to choose between opens in the centre - which is the right opening move.
#
# Deriving it from the line count rather than from distance to the middle matters: the corners
# are further from the centre than the edges are, but they are on three lines to the edges' two,
# and it is the lines that make them the better move.
CENTRE_FIRST: Tuple[int, ...] = tuple(
    sorted(range(CELLS), key=lambda cell: (-lines_through(cell), cell))
)
