"""
The bit tricks a Connect 4 position is made of, with no board class attached.

Kept separate from board.py so that evaluation.py can reach `runs` at runtime. The evaluation
counts open threes, which are runs of three with an empty cell on the end, so it needs the same
primitive win detection does - and games/chess/evaluation.py shows what happens otherwise: the
board imports its default evaluation, so the evaluation cannot import the board back and has to
hide behind TYPE_CHECKING. That works for chess because chess's evaluation only ever reads
attributes. Ours needs a function, so the shared half lives somewhere neither of them owns.

Everything here takes and returns plain ints. A Bitboard is a set of cells, whoever they belong
to; nothing in this module knows which player is which.
"""

from typing import Iterator

from games.connect4.constants import (
    BOTTOM_ROW,
    COLS,
    COLUMN_MASKS,
    CONNECT,
    DIRECTIONS,
    FULL_BOARD,
    ROWS,
    STRIDE,
)

Bitboard = int


def bit_count(bb: Bitboard) -> int:
    """
    The number of cells in a bitboard.

    `int.bit_count` would do this, but it arrived in Python 3.10 and this project supports 3.7,
    which is why games/chess/bitboard.py hand-rolls it too. Counting the characters of the
    binary string beats Kernighan's loop for the sparse masks used here: it is one pass in C
    rather than one Python iteration per set bit.
    """
    return bin(bb).count('1')


def cells(bb: Bitboard) -> Iterator[int]:
    """The indices set in a bitboard, lowest first."""
    while bb:
        low = bb & -bb
        yield low.bit_length() - 1
        bb ^= low


def index(column: int, row: int) -> int:
    """The index of a cell, row 0 being the bottom."""
    return column * STRIDE + row


def drops(occupied: Bitboard) -> Bitboard:
    """
    Where a disc would land in each column, as one bitboard.

    Adding BOTTOM_ROW sets off a carry in every column at once. In a column with discs in it the
    carry ripples up through them and settles in the first gap; in an empty column there is
    nothing to carry and the bottom cell is set directly. Either way the result has exactly one
    bit per non-full column, sitting on the cell a disc played there would occupy.

    A full column carries all the way into its sentinel, which is what the sentinel is for: the
    carry stops there instead of spilling into the bottom of the next column, and masking with
    FULL_BOARD then discards it. So a full column simply contributes nothing, and no height
    array has to be kept in step with the board to know that.
    """
    return (occupied + BOTTOM_ROW) & FULL_BOARD


def landing_square(occupied: Bitboard, column: int) -> Bitboard:
    """Where a disc played in `column` would land, or 0 if the column is full."""
    return drops(occupied) & COLUMN_MASKS[column]


def runs(position: Bitboard, delta: int, length: int = CONNECT) -> Bitboard:
    """
    The lowest cell of every run of `length` cells spaced `delta` apart in `position`.

    `position & (position >> delta)` marks the lower cell of every adjacent pair. Doing the same
    thing to *that* at twice the spacing marks the lower cell of every run of four, because a
    run of four is two overlapping pairs two apart. So the work halves each time and the whole
    thing costs about log2(length) shifts rather than length of them.

    `step` doubles to track how far the surviving marks already reach, and is clamped so the
    shifts total exactly `length - 1`:

        length 2 -> shifts of d                 (1 AND)
        length 3 -> shifts of d, d              (2 ANDs)
        length 4 -> shifts of d, 2d             (2 ANDs)
        length 5 -> shifts of d, 2d, d          (3 ANDs)

    No masking is needed. Shifting right discards bits below zero, and a run that would cross a
    column boundary has to pass through a sentinel cell, which is never set - so a chain that
    leaves the board dies rather than wrapping. See the module docstring in constants.py.
    """
    mask = position
    remaining = length - 1
    step = 1

    while remaining and mask:
        shift = min(step, remaining)
        mask &= mask >> (shift * delta)
        remaining -= shift
        step *= 2

    return mask


def has_run(position: Bitboard, delta: int, length: int = CONNECT) -> bool:
    """Whether `position` holds a run of `length` in the direction `delta`."""
    return bool(runs(position, delta, length))


def is_win(position: Bitboard) -> bool:
    """Whether one player's discs, given alone, contain a line of CONNECT."""
    return any(has_run(position, delta) for delta in DIRECTIONS)


def bitboard_to_str(bb: Bitboard, marker: str = 'x') -> str:
    """A picture of which cells are set, for reading a mask in a failing test."""
    lines = []
    for row in range(ROWS - 1, -1, -1):
        cells_in_row = ''.join(
            f'[{marker}]' if bb >> index(column, row) & 1 else '[ ]' for column in range(COLS)
        )
        lines.append(f'{row + 1} {cells_in_row}')
    lines.append('   ' + '  '.join(str(column + 1) for column in range(COLS)) + ' ')
    return '\n'.join(lines)
