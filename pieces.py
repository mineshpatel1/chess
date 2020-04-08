from typing import Iterable, Set

import log
from position import Position

PIECE_NAMES = {
    'p': 'Pawn',
    'r': 'Rook',
    'n': 'Knight',
    'b': 'Bishop',
    'q': 'Queen',
    'k': 'King',
}

PIECE_ICONS = {
    'P': '♙',
    'R': '♖',
    'N': '♘',
    'B': '♗',
    'Q': '♕',
    'K': '♔',
    'p': '♟',
    'r': '♜',
    'n': '♞',
    'b': '♝',
    'q': '♛',
    'k': '♚',
}


class Piece:
    TYPE = None

    def __init__(self, is_white: bool, pos: Position):
        self.is_white = is_white
        self.pos = pos

    @property
    def code(self) -> str:
        return self.TYPE.upper() if self.is_white else self.TYPE.lower()

    @property
    def icon(self) -> str:
        return PIECE_ICONS[self.code]

    @property
    def _moves(self) -> Iterable[Position]:
        return []

    @property
    def legal_moves(self) -> Set[Position]:
        return set(filter(lambda x: x.valid, self._moves))

    def __str__(self) -> str:
        return f'{self.icon}: {self.pos}'

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other) -> bool:
        log.info((self.TYPE, other.TYPE))
        return (
            self.TYPE == other.TYPE and
            self.is_white == other.is_white and
            self.pos == other.pos
        )

    def __hash__(self):
        return hash((self.TYPE, self.is_white, self.pos))

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

    @property
    def _moves(self) -> Iterable[Position]:
        direction = 1 if self.is_white else -1

        moves = [
            Position(self.pos.file, self.pos.rank + direction),  # Move one square forward
        ]

        # Move 2 squares if at original rank
        if self.is_white and self.pos.rank == 1:
            moves.append(
                Position(self.pos.file, self.pos.rank + 2)
            )
        elif not self.is_white and self.pos.rank == 6:
            moves.append(
                Position(self.pos.file, self.pos.rank - 2)
            )

        return moves


class Rook(Piece):
    TYPE = 'r'


class Knight(Piece):
    TYPE = 'n'


class Bishop(Piece):
    TYPE = 'b'


class Queen(Piece):
    TYPE = 'q'


class King(Piece):
    TYPE = 'k'
