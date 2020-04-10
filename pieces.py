from __future__ import annotations

from typing import Iterable, List, Tuple, Set, TYPE_CHECKING

import log
from position import Position
from constants import WHITE, BLACK, PIECE_ICONS, PIECE_NAMES, CARDINALS, DIAGONALS

if TYPE_CHECKING:
    from board import Board


class Piece:
    TYPE = None

    def __init__(self, pos: Position, colour: str = WHITE):
        assert colour in [WHITE, BLACK], "Only white and black pieces allowed."
        self.colour = colour
        self.pos = pos

    def _moves(self, board: Board) -> Iterable[Position]:
        return []

    def _repeat_move(self, board: Board, move_patterns: List[Tuple[int, int]]) -> List[Position]:
        moves = []

        for x, y in move_patterns:
            _pos = Position(self.pos.file + x, self.pos.rank + y)
            while _pos.in_board:
                square = board.squares[_pos.index]

                if square.is_occupied:
                    if square.piece.colour != self.colour:
                        moves.append(_pos)
                    break

                moves.append(_pos)
                _pos = Position(_pos.file + x, _pos.rank + y)
        return moves

    def _occupied_same_team(self, pos: Position, board: Board) -> bool:
        square = board.squares[pos.index]
        if square.is_occupied:
            return square.piece.colour == self.colour
        else:
            return False

    def _occupied_opposite_team(self, pos: Position, board: Board) -> bool:
        square = board.squares[pos.index]
        if square.is_occupied:
            return square.piece.colour != self.colour
        else:
            return False

    def same_team(self, other) -> bool:
        return self.colour == other.colour

    def legal_moves(self, board: Board) -> Set[Position]:
        return set(filter(
            lambda pos: pos.in_board,
            self._moves(board),
        ))

    @property
    def name(self) -> str:
        return f'{self.colour.title()} {PIECE_NAMES[self.TYPE]}'

    @property
    def code(self) -> str:
        return self.TYPE.upper() if self.colour == WHITE else self.TYPE.lower()

    @property
    def icon(self) -> str:
        return PIECE_ICONS[self.code]

    def __str__(self) -> str:
        return f'{self.icon}: {self.pos}'

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other) -> bool:
        return (
            self.TYPE == other.TYPE and
            self.colour == other.colour and
            self.pos == other.pos
        )

    def __hash__(self):
        return hash((self.TYPE, self.pos, self.colour))

    def __lt__(self, other) -> bool:
        return self.pos < other.pos

    def __le__(self, other) -> bool:
        return self.pos <= other.pos

    def __gt__(self, other) -> bool:
        return self.pos > other.pos

    def __ge__(self, other) -> bool:
        return self.pos >= other.pos


class Pawn(Piece):
    TYPE = 'p'

    def _moves(self, board: Board) -> Iterable[Position]:
        direction = 1 if self.colour == WHITE else -1

        forwards = [
            Position(self.pos.file, self.pos.rank + direction),  # Move one square forward
        ]

        # Move 2 squares if at original rank
        if self.colour == WHITE and self.pos.rank == 1:
            forwards.append(
                Position(self.pos.file, self.pos.rank + 2)
            )
        elif self.colour == BLACK and self.pos.rank == 6:
            forwards.append(
                Position(self.pos.file, self.pos.rank - 2)
            )

        forwards = filter(
            lambda pos: (
                pos.in_board and
                not board.squares[pos.index].is_occupied
            ),
            forwards,
        )

        # If the forward-diagonal pieces are of the opposite colour, these are legal attack moves.
        attacks = filter(
            lambda pos: (
                pos.in_board and
                self._occupied_opposite_team(pos, board)
            ),
            [
                Position(self.pos.file + 1, self.pos.rank + direction),
                Position(self.pos.file - 1, self.pos.rank + direction),
            ]
        )

        return list(forwards) + list(attacks)


class Rook(Piece):
    TYPE = 'r'

    def _moves(self, board: Board) -> Iterable[Position]:
        moves = self._repeat_move(board, CARDINALS)
        return moves


class Knight(Piece):
    TYPE = 'n'

    def _moves(self, board: Board) -> Iterable[Position]:
        moves = []
        for i in [1, -1, 2, -2]:
            for j in [1, -1, 2, -2]:
                if (abs(i) + abs(j)) == 3:
                    pos = Position(self.pos.file + i, self.pos.rank + j)
                    if (
                        pos.in_board and
                        not self._occupied_same_team(pos, board)
                    ):
                        moves.append(pos)
        return moves


class Bishop(Piece):
    TYPE = 'b'

    def _moves(self, board: Board) -> Iterable[Position]:
        moves = self._repeat_move(board, DIAGONALS)
        return moves


class Queen(Piece):
    TYPE = 'q'

    def _moves(self, board: Board) -> Iterable[Position]:
        moves = self._repeat_move(board, CARDINALS + DIAGONALS)
        return moves


class King(Piece):
    TYPE = 'k'

    def _moves(self, board: Board) -> Iterable[Position]:
        moves = []
        for x, y in CARDINALS + DIAGONALS:
            pos = Position(self.pos.file + x, self.pos.rank + y)
            if pos.in_board and not self._occupied_same_team(pos, board):
                moves.append(pos)
        return moves

