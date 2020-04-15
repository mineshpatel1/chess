from __future__ import annotations

from engine.square import Square, file_rank_to_index, char_to_file


class Move:
    @staticmethod
    def from_uci(uci: str) -> Move:
        assert len(uci) == 4, "Invalid UCI"
        pos_1 = uci[:2]
        from_sq = file_rank_to_index(char_to_file(pos_1[0]), int(pos_1[1]) - 1)
        pos_2 = uci[2:4]
        to_sq = file_rank_to_index(char_to_file(pos_2[0]), int(pos_2[1]) - 1)
        return Move(Square(from_sq), Square(to_sq))

    def __init__(
            self,
            from_square: Square,
            to_square: Square,
            is_castling: bool = False,
    ):
        self.from_square = Square(from_square)
        self.to_square = Square(to_square)
        self.is_castling = is_castling

    @property
    def uci(self):
        return f'{str(self.from_square).lower()}{str(self.to_square).lower()}'

    def __str__(self):
        return f'{self.uci}'

    def __repr__(self):
        return f"'{str(self)}'"

    def __hash__(self):
        return hash((
            self.from_square,
            self.to_square
        ))

    def __eq__(self, other):
        return (
            other.from_square == self.from_square and
            other.to_square == self.to_square
        )