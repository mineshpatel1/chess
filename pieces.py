from typing import Iterable, Set

from position import Position


class Piece:
    def __init__(self, is_white: bool, pos: Position):
        self.is_white = is_white
        self.pos = pos

    @property
    def _moves(self) -> Iterable[Position]:
        return []

    @property
    def valid_moves(self) -> Set[Position]:
        return set(filter(lambda x: x.valid, self._moves))


class Pawn(Piece):
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
