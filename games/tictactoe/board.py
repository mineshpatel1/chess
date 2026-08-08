"""
A tic-tac-toe position, and the rules for getting from one to the next.

The position is two integers, one per player, over the nine cells described in constants.py. Two
boards rather than one array of three-valued cells, for the reasons Connect 4 gives: making and
unmaking a move becomes `self.marks[player] ^= bit`, which is its own inverse and so cannot be
got wrong by undoing in the wrong order, and win detection becomes one `&` against each of the
eight lines.

This is the smallest game in the project by some distance. It has no bitboard module, no shift
tricks and no move objects - a move is the number of the cell, which is the number the board
prints in that cell - and it exists to show how little the GameState contract actually asks for.
Nine cells also mean the whole game tree fits inside a search, which is what SOLVED_DEPTH is
about: the engine here is not merely good, it is perfect, and tests/tictactoe/test_perfect_play.py
proves it against an independent solver.
"""

from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Optional

from games.base import DRAW, GameState, Outcome, win
from games.tictactoe.constants import (
    CELLS,
    CENTRE_FIRST,
    CROSS,
    FULL_BOARD,
    MARK_ICONS,
    NOUGHT,
    SIDE,
    WIN_MASKS,
    Mark,
    index,
)
from games.tictactoe.encoding import TicTacToeEncoder
from games.tictactoe.evaluation import bit_count, weighted_eval


class IllegalMove(Exception):
    """A cell that is not on the board, or one that is already taken."""


def is_win(marks: int) -> bool:
    """Whether a set of marks contains a whole line.

    The entire win condition, and the reason this game needs no bitboard module: eight constant
    masks and an `&`, where Connect 4 needs a padded layout and a halving shift to find a run of
    four anywhere on a grid it cannot enumerate.
    """
    return any(marks & line == line for line in WIN_MASKS)


class TicTacToe(GameState):
    # Nine moves at the root and an evaluation that returns a constant, so a node costs less than
    # handing it to another process would. Starting a pool here would lose every time.
    PARALLEL_ROOT = False

    # Nine cells, so nine plies is the whole game from an empty board and no position can be
    # deeper. A search this deep has seen every leaf and is playing perfectly, which is what
    # play.py uses this for.
    SOLVED_DEPTH = CELLS

    DEFAULT_EVAL = staticmethod(weighted_eval)

    # How a network sees this game, for ai.zero. Two perspective-relative planes and nine shared
    # actions - see games/tictactoe/encoding.py for why both of those matter.
    ENCODER = TicTacToeEncoder

    def __init__(self, cells: Iterable[int] = ()) -> None:
        """A position, reached by playing `cells` from an empty board."""
        self.marks: Dict[Mark, int] = {CROSS: 0, NOUGHT: 0}
        self.turn: Mark = CROSS
        self.move_stack: List[int] = []

        for cell in cells:
            if cell not in self.legal_moves:
                raise IllegalMove(f'cell {cell} cannot be played here')
            self.make_move(cell)

    # ---- position ----------------------------------------------------------------------

    @property
    def occupied(self) -> int:
        """
        Every cell holding a mark.

        Derived rather than cached, for the reason Connect 4 gives: a cache that make_move and
        unmake_move have to keep in step is a cache that will one day be right almost always,
        and move generation would quietly lie rather than fail. Measure before adding one.
        """
        return self.marks[CROSS] | self.marks[NOUGHT]

    @property
    def legal_moves(self) -> Iterator[int]:
        """
        Every empty cell, the ones on the most winning lines first.

        The order is load-bearing rather than cosmetic - see CENTRE_FIRST in constants.py. It is
        what makes alpha-beta pay, and it is also the whole of this game's opening book, because
        the search breaks ties by generation order and every opening move ties.
        """
        occupied = self.occupied
        return (cell for cell in CENTRE_FIRST if not occupied >> cell & 1)

    @property
    def signature(self) -> str:
        """
        The two mark masks, which are the whole of the position.

        Tic-tac-toe carries none of the state chess hides behind an identical-looking board - no
        castling rights, no en passant square, no clock - so nothing is lost by not printing it.
        Overridden only because the inherited default renders the whole grid, and the conformance
        suite asks for this once per move of every position it walks.
        """
        return f'{self.marks[CROSS]}/{self.marks[NOUGHT]}'

    def copy(self) -> 'TicTacToe':
        """
        An independent board at the same position.

        The move stack comes too, so a copy stays undoable to the same depth the original was.
        Chess drops its history here because rebuilding from a FEN is cheaper than copying it;
        ours is at most nine small ints.
        """
        clone = type(self)()  # type(self), so a subclass copies to its own kind
        clone.marks[CROSS] = self.marks[CROSS]
        clone.marks[NOUGHT] = self.marks[NOUGHT]
        clone.turn = self.turn
        clone.move_stack = list(self.move_stack)
        return clone

    # ---- playing -----------------------------------------------------------------------

    def make_move(self, cell: int) -> None:
        """
        Puts the mover's mark in `cell` and hands over the turn.

        Assumes the cell is legal - this is the search's hot path, and `legal_moves` is where
        legality is decided. `^=` rather than `|=`, for the reason Connect 4 gives: both happen
        to work while the target cell is provably empty and fail silently when it is not, and
        XOR being its own inverse makes unmake_move the same line again.
        """
        assert not self.occupied >> cell & 1, f'cell {cell} is taken'

        self.marks[self.turn] ^= 1 << cell
        self.move_stack.append(cell)
        self.turn = not self.turn

    def unmake_move(self) -> None:
        """Takes the last mark back off the board."""
        cell = self.move_stack.pop()
        self.turn = not self.turn
        self.marks[self.turn] ^= 1 << cell

    # ---- results -----------------------------------------------------------------------

    @property
    def outcome(self) -> Optional[Outcome]:
        """
        The winner, if the last move made one.

        This is the property chess does not have. Chess ends by a player running out of moves, so
        it can leave this as the inherited None; tic-tac-toe is won with up to four cells still
        empty, so without this the search would play straight through a finished game.

        Only one board is tested, because only one player can have just won: the one who moved.
        That halves the cost of the property the search touches at every single node. It is sound
        because no constructor here can build a position where the player *to* move already has a
        line - `from_diagram` rejects those, and play cannot reach one.

        At the start the mover is Noughts, whose board is 0, and no line is a subset of nothing,
        so a game that has not begun needs no special case.
        """
        mover = not self.turn
        return win(mover) if is_win(self.marks[mover]) else None

    @property
    def outcome_without_moves(self) -> Outcome:
        """Nowhere to play means the grid is full, and a full grid nobody won is a draw."""
        return DRAW

    # ---- reading and writing positions -------------------------------------------------

    @classmethod
    def from_diagram(cls, diagram: str) -> 'TicTacToe':
        """
        A position from a picture of one, top row first, for tests that want to be read.

        `.` is empty, `X` and `O` are marks:

            TicTacToe.from_diagram('''
                X.O
                .X.
                O.X
            ''')

        This is the one sanctioned way to set the boards other than by playing, so it carries the
        checks that playing gets for free. A diagram with the wrong number of each mark, or in
        which the player *to move* has already won, would break the assumption `outcome` rests
        on, and would do it invisibly: `result` delegates to `outcome`, so the conformance suite
        could not catch it either.
        """
        rows = [line.strip() for line in diagram.strip().splitlines()]
        rows = [row for row in rows if row]
        if len(rows) != SIDE:
            raise ValueError(f'expected {SIDE} rows, got {len(rows)}')

        board = cls()
        for row, text in enumerate(rows):
            if len(text) != SIDE:
                raise ValueError(f'expected {SIDE} columns, got {len(text)} in {text!r}')

            for column, char in enumerate(text):
                if char in 'Xx':
                    board.marks[CROSS] |= 1 << index(column, row)
                elif char in 'Oo0':
                    board.marks[NOUGHT] |= 1 << index(column, row)
                elif char not in '.-_ ':
                    raise ValueError(f'unrecognised cell {char!r}')

        board._validate()
        return board

    def _validate(self) -> None:
        """Checks a hand-built position is one that play could actually have reached."""
        crosses, noughts = self.marks[CROSS], self.marks[NOUGHT]

        if crosses & noughts:
            raise ValueError('a cell holds two marks')
        if (crosses | noughts) & ~FULL_BOARD:
            raise ValueError('a mark is off the board')

        played = bit_count(crosses), bit_count(noughts)
        if played[0] not in (played[1], played[1] + 1):
            raise ValueError(f'Crosses has played {played[0]} marks and Noughts {played[1]}')
        self.turn = CROSS if played[0] == played[1] else NOUGHT

        if is_win(self.marks[self.turn]):
            raise ValueError('the player to move has already won')

    def parse_move(self, text: str) -> int:
        """A cell number, as typed. Cells are numbered from 0, as the board prints them."""
        try:
            cell = int(text.strip())
        except ValueError:
            raise ValueError(f'{text.strip()!r} is not a cell number') from None

        if cell not in self.legal_moves:
            if 0 <= cell < CELLS:
                raise ValueError(f'cell {cell} is taken')
            raise ValueError(f'there is no cell {cell}')
        return cell

    def __str__(self) -> str:
        """
        The board as a picture, top row first.

        An empty cell prints its own number, so the board is its own guide to what to type and
        there is nothing to label underneath it the way Connect 4 labels its columns.
        """
        lines = []
        for row in range(SIDE):
            cells = ''
            for column in range(SIDE):
                cell = index(column, row)
                bit = 1 << cell
                if self.marks[CROSS] & bit:
                    cells += f'[{MARK_ICONS[CROSS]}]'
                elif self.marks[NOUGHT] & bit:
                    cells += f'[{MARK_ICONS[NOUGHT]}]'
                else:
                    cells += f'[{cell}]'
            lines.append(cells)
        return '\n' + '\n'.join(lines)
