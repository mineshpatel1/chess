"""
The shape of a Connect 4 board, and every mask derived from it.

A cell is addressed as `index = column * STRIDE + row`, with row 0 at the bottom. STRIDE is one
greater than the number of rows, so every column carries a spare cell above it that is never
occupied - the *sentinel*. The board therefore looks like this, indices rising up each column:

     6 13 20 27 34 41 48    <- sentinel, always zero
     5 12 19 26 33 40 47
     4 11 18 25 32 39 46
     3 10 17 24 31 38 45
     2  9 16 23 30 37 44
     1  8 15 22 29 36 43
     0  7 14 21 28 35 42

The sentinel is the whole reason this layout is worth the extra bits. Two things in this engine
walk along a straight line by adding a constant to an index - the carry in `drops`, which finds
where a disc lands, and the shifts in `runs`, which find four in a row - and both would happily
walk off the top of one column and into the bottom of the next if the columns were packed
tightly. Because the sentinel sits between them, any such walk lands on a cell that is
permanently empty and is cleared by FULL_BOARD, so it dies there instead.

That is a property of the four directions rather than a coincidence. Each of them advances the
row by at most one per step, so a chain leaving the six-row band must land on `column * STRIDE +
ROWS` - a sentinel - before it can reach any real cell of the neighbouring column. Masking with
FULL_BOARD after a shift or a carry is therefore sufficient to prevent every wrap, and no
direction needs a special case.

Nothing here is written down that can be derived. Changing ROWS, COLS or CONNECT should be the
only edit a differently shaped game needs, and tests/connect4/test_bitboard.py re-derives the
masks independently to keep that true.
"""

from typing import List, Tuple

from games.base import Player

ROWS = 6
COLS = 7
CONNECT = 4  # Discs in a line that wins

STRIDE = ROWS + 1  # The sentinel row that isolates one column from the next
SIZE = COLS * STRIDE

# True is the first player, the one a positive evaluation favours, matching Player in games.base
# and WHITE in chess.
Disc = Player
YELLOW: Disc = True
RED: Disc = False

DISCS: Tuple[Disc, Disc] = (YELLOW, RED)
DISC_ICONS = {YELLOW: '●', RED: '○'}
DISC_NAMES = {YELLOW: 'Yellow', RED: 'Red'}

# One mask per column, covering its real cells and not its sentinel.
COLUMN_MASKS: List[int] = [((1 << ROWS) - 1) << (column * STRIDE) for column in range(COLS)]

# Every playable cell. Sentinels are excluded by construction rather than by subtraction, which
# is what makes `& FULL_BOARD` a safe way to discard a walk that left the board.
#
# `sum` rather than `|` throughout: the masks being summed are disjoint by construction, one
# cell per column, so the two are the same and this reads better.
FULL_BOARD = sum(COLUMN_MASKS)
SENTINEL_ROW = sum(1 << (column * STRIDE + ROWS) for column in range(COLS))
BOTTOM_ROW = sum(1 << (column * STRIDE) for column in range(COLS))
TOP_ROW = sum(1 << (column * STRIDE + ROWS - 1) for column in range(COLS))

# Any straight line of constant (dcolumn, drow) is a constant step in index, so a diagonal is
# not a special case - only a larger constant.
#
#   direction    (dcolumn, drow)   delta
#   vertical         (0, +1)       1
#   horizontal       (+1, 0)       STRIDE
#   diagonal /       (+1, +1)      STRIDE + 1
#   diagonal \       (+1, -1)      STRIDE - 1
#
# Four is all that is needed: a run in direction -d covers the same cells as one in +d.
VERTICAL = 1
HORIZONTAL = STRIDE
DIAGONAL_UP = STRIDE + 1
DIAGONAL_DOWN = STRIDE - 1
DIRECTIONS: Tuple[int, ...] = (VERTICAL, HORIZONTAL, DIAGONAL_UP, DIAGONAL_DOWN)

# Columns nearest the middle first. The search iterates moves in generation order and breaks
# ties by it, so generating this way is both the move ordering that makes alpha-beta pay and the
# reason a search with nothing to choose between takes the centre.
CENTRE_FIRST: Tuple[int, ...] = tuple(
    sorted(range(COLS), key=lambda column: (abs(2 * column - (COLS - 1)), column))
)
