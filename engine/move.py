from __future__ import annotations
from typing import Union, Optional

from engine.constants import PieceType, QUEEN, ROOK, BISHOP, KNIGHT
from engine.square import Square, file_rank_to_index, char_to_file


class Move:
    @staticmethod
    def from_uci(uci: str) -> Move:
        assert len(uci) in (4, 5), "Invalid UCI"
        pos_1 = uci[:2]
        from_sq = file_rank_to_index(char_to_file(pos_1[0]), int(pos_1[1]) - 1)
        pos_2 = uci[2:4]
        to_sq = file_rank_to_index(char_to_file(pos_2[0]), int(pos_2[1]) - 1)

        promotion = None
        if len(uci) > 4:
            promotion = uci[4]
        return Move(from_sq, to_sq, promotion=promotion)

    def __init__(
            self,
            from_square: Union[Square, int],
            to_square: Union[Square, int],
            is_castling: bool = False,
            promotion: Optional[PieceType] = None,
    ):
        self.from_square = Square(from_square)
        self.to_square = Square(to_square)
        self.is_castling = is_castling
        assert promotion in (None, QUEEN, ROOK, BISHOP, KNIGHT)
        self.promotion = promotion

    @property
    def uci(self) -> str:
        promotion = self.promotion if self.promotion else ''
        return f'{str(self.from_square).lower()}{str(self.to_square).lower()}{promotion}'

    def __str__(self) -> str:
        return f'{self.uci}'

    def __repr__(self) -> str:
        return f"'{str(self)}'"

    def __hash__(self) -> int:
        return hash((
            self.from_square,
            self.to_square
        ))

    def __eq__(self, other) -> bool:
        return (
            other.from_square == self.from_square and
            other.to_square == self.to_square
        )