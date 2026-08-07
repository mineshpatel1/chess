"""
A Connect 4 position, and the rules for getting from one to the next.

The position is two integers, one per player, over the sentinel-padded layout described in
constants.py. Two boards rather than the position/mask pairing a lot of Connect 4 engines use,
for two reasons: making and unmaking a move becomes `self.discs[player] ^= bit`, which is its
own inverse and so cannot be got wrong by undoing in the wrong order; and the evaluation wants
to ask about one player's discs on their own anyway.
"""

from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Optional

from games.base import DRAW, GameState, Outcome, win
from games.connect4.bitboard import (
    Bitboard,
    bit_count,
    drops,
    index,
    is_win,
    landing_square,
)
from games.connect4.constants import (
    CENTRE_FIRST,
    COLS,
    COLUMN_MASKS,
    DISC_ICONS,
    FULL_BOARD,
    RED,
    ROWS,
    STRIDE,
    YELLOW,
    Disc,
)
from games.connect4.evaluation import weighted_eval


class IllegalMove(Exception):
    """A column that is not on the board, or one that is already full."""


class Connect4(GameState):
    # Seven moves at the root, and an evaluation cheap enough that a node costs less than
    # handing it to another process would. Starting a pool here would lose every time.
    PARALLEL_ROOT = False

    DEFAULT_EVAL = staticmethod(weighted_eval)

    def __init__(self, columns: Iterable[int] = ()) -> None:
        """A position, reached by playing `columns` from an empty board."""
        self.discs: Dict[Disc, Bitboard] = {YELLOW: 0, RED: 0}
        self.turn: Disc = YELLOW
        self.move_stack: List[Bitboard] = []

        for column in columns:
            if column not in self.legal_moves:
                raise IllegalMove(f'column {column} cannot be played here')
            self.make_move(column)

    # ---- position ----------------------------------------------------------------------

    @property
    def occupied(self) -> Bitboard:
        """
        Every cell holding a disc.

        Derived rather than cached. Caching it means make_move and unmake_move have to keep it
        in step, and a cache that is right almost always is worse than no cache at all here -
        move generation and win detection would both quietly lie. Measure before adding one.
        """
        return self.discs[YELLOW] | self.discs[RED]

    @property
    def legal_moves(self) -> Iterator[int]:
        """
        Every column with room in it, nearest the middle first.

        The order is deliberate and load-bearing. `ai.search` iterates moves in generation
        order and breaks ties by it, so centre-first generation is what makes alpha-beta's
        pruning pay - best-move-first takes the cost from about 7^d to 7^(d/2), and worst-first
        gains nothing at all. It also means a search with nothing to choose between takes the
        centre column, which happens to be the right opening move.
        """
        landing = drops(self.occupied)
        return (column for column in CENTRE_FIRST if landing & COLUMN_MASKS[column])

    @property
    def signature(self) -> str:
        """
        The two disc masks, which are the whole of the position.

        Connect 4 carries none of the state chess hides behind an identical-looking board - no
        castling rights, no en passant square, no clock - so nothing is lost by not printing it.
        Overridden only because the inherited default renders the whole grid, and the
        conformance suite asks for this once per move of every position it walks.
        """
        return f'{self.discs[YELLOW]}/{self.discs[RED]}'

    def copy(self) -> 'Connect4':
        """
        An independent board at the same position.

        The move stack comes too. Chess drops its history here because rebuilding from a FEN is
        cheaper than copying it, but ours is a list of small ints and copying it costs nothing,
        so a copy stays undoable to the same depth the original was.
        """
        clone = Connect4()
        clone.discs[YELLOW] = self.discs[YELLOW]
        clone.discs[RED] = self.discs[RED]
        clone.turn = self.turn
        clone.move_stack = list(self.move_stack)
        return clone

    # ---- playing -----------------------------------------------------------------------

    def make_move(self, column: int) -> None:
        """
        Drops a disc into `column` and hands over the turn.

        Assumes the column is legal - this is the search's hot path, and `legal_moves` is where
        legality is decided. A full column yields a landing square of 0, so an illegal move here
        is a silent no-op on the boards and a corrupt turn, which is exactly what the
        `landing_square` assertion is for when running under -O0.

        `^=` rather than `|=`, throughout and on purpose. Addition and OR both happen to work
        while the target cell is provably empty and fail silently when it is not; XOR is its own
        inverse, so unmake_move is the same line again.
        """
        bit = landing_square(self.occupied, column)
        assert bit, f'column {column} is full'

        self.discs[self.turn] ^= bit
        self.move_stack.append(bit)
        self.turn = not self.turn

    def unmake_move(self) -> None:
        """Takes the last disc back off the board."""
        bit = self.move_stack.pop()
        self.turn = not self.turn
        self.discs[self.turn] ^= bit

    # ---- results -----------------------------------------------------------------------

    @property
    def outcome(self) -> Optional[Outcome]:
        """
        The winner, if the last move made one.

        This is the property chess does not have. Chess ends by a player running out of moves,
        so it can leave this as the inherited None; Connect 4 is won with the board half empty
        and legal moves still on offer, so without this the search would play straight through a
        finished game.

        Only one board is tested, because only one player can have just won: the one who moved.
        That halves the cost of the property the search touches at every single node. It is
        sound because no constructor here can build a position where the player *to* move
        already has four - `from_diagram` rejects those, and play cannot reach one.

        At the start the mover is Red, whose board is 0, and an empty board has no run, so no
        special case is needed for a game that has not begun.
        """
        mover = not self.turn
        return win(mover) if is_win(self.discs[mover]) else None

    @property
    def outcome_without_moves(self) -> Outcome:
        """Nowhere to play means the grid is full, and a full grid nobody won is a draw."""
        return DRAW

    # ---- reading and writing positions -------------------------------------------------

    @classmethod
    def from_diagram(cls, diagram: str) -> 'Connect4':
        """
        A position from a picture of one, top row first, for tests that want to be read.

        `.` is empty, `Y` and `R` are discs:

            Connect4.from_diagram('''
                .......
                .......
                .......
                .......
                ...R...
                ..RYY..
            ''')

        This is the one sanctioned way to set the boards other than by playing, so it carries
        the checks that playing gets for free. A diagram with a floating disc, with the wrong
        number of each colour, or in which the player *to move* has already won would break the
        assumption `outcome` rests on, and would do it invisibly: `result` delegates to
        `outcome`, so the conformance suite could not catch it either.
        """
        rows = [line.strip() for line in diagram.strip().splitlines()]
        rows = [row for row in rows if row]
        if len(rows) != ROWS:
            raise ValueError(f'expected {ROWS} rows, got {len(rows)}')

        board = cls()
        for height, line in enumerate(rows):
            if len(line) != COLS:
                raise ValueError(f'expected {COLS} columns, got {len(line)} in {line!r}')

            row = ROWS - 1 - height  # Diagrams read downwards; rows count upwards
            for column, char in enumerate(line):
                if char in 'Yy':
                    board.discs[YELLOW] |= 1 << index(column, row)
                elif char in 'Rr':
                    board.discs[RED] |= 1 << index(column, row)
                elif char not in '.-_ ':
                    raise ValueError(f'unrecognised cell {char!r}')

        board._validate()
        return board

    def _validate(self) -> None:
        """Checks a hand-built position is one that play could actually have reached."""
        yellow, red = self.discs[YELLOW], self.discs[RED]

        if yellow & red:
            raise ValueError('a cell holds two discs')
        if (yellow | red) & ~FULL_BOARD:
            raise ValueError('a disc is off the board')

        played = bit_count(yellow), bit_count(red)
        if played[0] not in (played[1], played[1] + 1):
            raise ValueError(f'Yellow has played {played[0]} discs and Red {played[1]}')
        self.turn = YELLOW if played[0] == played[1] else RED

        occupied = self.occupied
        for column in range(COLS):
            # Discs stack from the bottom, so a column read down to bit 0 must be a solid run.
            stack = (occupied & COLUMN_MASKS[column]) >> (column * STRIDE)
            if stack != (1 << bit_count(stack)) - 1:
                raise ValueError(f'column {column} has a disc floating above a gap')

        if is_win(self.discs[self.turn]):
            raise ValueError('the player to move has already won')

    @property
    def columns_played(self) -> List[int]:
        """The columns of the moves on the stack, in the order they were played."""
        return [bit.bit_length() // STRIDE for bit in self.move_stack]

    def parse_move(self, text: str) -> int:
        """A column number, as typed. Columns are numbered from 0, as the board prints them."""
        try:
            column = int(text.strip())
        except ValueError:
            raise ValueError(f'{text!r} is not a column number') from None

        if column not in self.legal_moves:
            if 0 <= column < COLS:
                raise ValueError(f'column {column} is full')
            raise ValueError(f'there is no column {column}')
        return column

    def __str__(self) -> str:
        """The board as a picture, top row first, with the column numbers under it."""
        lines = []
        for row in range(ROWS - 1, -1, -1):
            cells = ''
            for column in range(COLS):
                bit = 1 << index(column, row)
                if self.discs[YELLOW] & bit:
                    cells += f'[{DISC_ICONS[YELLOW]}]'
                elif self.discs[RED] & bit:
                    cells += f'[{DISC_ICONS[RED]}]'
                else:
                    cells += '[ ]'
            lines.append(cells)
        lines.append(' ' + '  '.join(str(column) for column in range(COLS)) + ' ')
        return '\n' + '\n'.join(lines)
