from __future__ import annotations

from typing import Optional, Iterable, List, Tuple, Set, TYPE_CHECKING

from engine.position import Position

from engine.constants import (
    WHITE,
    BLACK,
    PIECE_ICONS,
    PIECE_NAMES,
    PAWN,
    ROOK,
    KNIGHT,
    BISHOP,
    QUEEN,
    KING,
    CARDINALS,
    DIAGONALS,
    QUEENSIDE,
    KINGSIDE,
)

if TYPE_CHECKING:
    from engine.board import Board


class Piece:
    TYPE = None
    CASTLE_POSITIONS = {
        WHITE: {},
        BLACK: {},
    }

    def __init__(self, pos: Position, colour: str = WHITE):
        assert colour in [WHITE, BLACK], "Only white and black pieces allowed."
        self.colour = colour
        self.original_pos = pos
        self.pos = pos
        self.move_history = []

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
        return f'{self.colour.title()} {PIECE_NAMES[self.type]}'

    @property
    def type(self):
        return self.TYPE

    @property
    def code(self) -> str:
        return self.type.upper() if self.colour == WHITE else self.type.lower()

    @property
    def icon(self) -> str:
        return PIECE_ICONS[self.code]

    @property
    def can_castle(self) -> bool:
        if self.type in (ROOK, KING):  # Only Rooks and Kings can castle
            if len(self.move_history) > 0:
                return False

            return self.pos in self.CASTLE_POSITIONS[self.colour]
        else:
            return False

    @property
    def rook_type(self) -> Optional[str]:
        return None

    def __str__(self) -> str:
        return f'{self.icon}: {self.pos}'

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other) -> bool:
        return (
            self.type == other.type and
            self.colour == other.colour and
            self.original_pos == other.original_pos
        )

    def __hash__(self):
        return hash((self.type, self.original_pos, self.colour))

    def __lt__(self, other) -> bool:
        return self.pos < other.pos

    def __le__(self, other) -> bool:
        return self.pos <= other.pos

    def __gt__(self, other) -> bool:
        return self.pos > other.pos

    def __ge__(self, other) -> bool:
        return self.pos >= other.pos


class Pawn(Piece):
    TYPE = PAWN

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
        potential_attacks = (
            Position(self.pos.file + 1, self.pos.rank + direction),
            Position(self.pos.file - 1, self.pos.rank + direction),
        )

        attacks = []
        for pos in potential_attacks:
            if pos.in_board and self._occupied_opposite_team(pos, board):
                attacks.append(pos)
            elif board.en_passant is not None:
                if pos == board.en_passant:
                    attacks.append(pos)

        return list(forwards) + list(attacks)


class Rook(Piece):
    TYPE = ROOK
    CASTLE_POSITIONS = {
        WHITE: {
            Position(0, 0): QUEENSIDE,
            Position(7, 0): KINGSIDE,
        },
        BLACK: {
            Position(0, 7): QUEENSIDE,
            Position(7, 7): KINGSIDE,
        }
    }

    def _moves(self, board: Board) -> Iterable[Position]:
        moves = self._repeat_move(board, CARDINALS)
        return moves

    @property
    def rook_type(self) -> Optional[str]:
        if len(self.move_history) > 0:
            return None
        else:
            return self.CASTLE_POSITIONS[self.colour].get(self.pos)


class Knight(Piece):
    TYPE = KNIGHT

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
    TYPE = BISHOP

    def _moves(self, board: Board) -> Iterable[Position]:
        moves = self._repeat_move(board, DIAGONALS)
        return moves


class Queen(Piece):
    TYPE = QUEEN

    def _moves(self, board: Board) -> Iterable[Position]:
        moves = self._repeat_move(board, CARDINALS + DIAGONALS)
        return moves


class King(Piece):
    TYPE = KING

    CASTLE_POSITIONS = {
        WHITE: {
            Position(4, 0): True,
        },
        BLACK: {
            Position(4, 7): True,
        }
    }

    def _moves(self, board: Board) -> Iterable[Position]:
        moves = []
        for x, y in CARDINALS + DIAGONALS:
            pos = Position(self.pos.file + x, self.pos.rank + y)
            if pos.in_board and not self._occupied_same_team(pos, board):
                moves.append(pos)

        # Add moves for castling if possible
        if self.can_castle:
            for rook in filter(
                lambda p: p.colour == self.colour and p.type == ROOK and p.can_castle,
                board.pieces,
            ):

                if rook.rook_type == KINGSIDE:
                    _num_squares, direction = 2, 1
                elif rook.rook_type == QUEENSIDE:
                    _num_squares, direction = 3, -1
                else:
                    continue

                can_castle = True
                for i in range(_num_squares):
                    _castle_pos = Position(self.pos.file + (direction * (i + 1)), self.pos.rank)
                    if board.squares[_castle_pos.index].is_occupied:
                        can_castle = False
                        break

                if can_castle:
                    moves.append(Position(self.pos.file + (direction * 2), self.pos.rank))

        return moves
