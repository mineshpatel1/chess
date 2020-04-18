from typing import Tuple

from engine.constants import (
    WHITE,
    BLACK,
    FILE_NAMES,
    RANK_NAMES,

    PAWN_POSITION_BASE_VALUES,
    KNIGHT_POSITION_BASE_VALUES,
    BISHOP_POSITION_BASE_VALUES,
    ROOK_POSITION_BASE_VALUES,
    QUEEN_POSITION_BASE_VALUES,
    KING_POSITION_BASE_VALUES,
)


def index_to_file_rank(idx: int) -> Tuple[int, int]:
    """Converts an integer position into the corresponding file and rank (in that order)."""
    assert 0 <= idx < 64, "Board index must be between 0 and 63."
    return idx % 8, int(idx / 8)


def file_rank_to_index(file: int, rank: int) -> int:
    assert 0 <= file < 8 and 0 <= rank < 8, "Rank and file must be between 0 and 7."
    return (rank * 8) + (file % 8)


def file_to_char(file: int) -> str:
    return ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'][file]


def char_to_file(char: str) -> int:
    return {
        'A': 0,
        'B': 1,
        'C': 2,
        'D': 3,
        'E': 4,
        'F': 5,
        'G': 6,
        'H': 7,
    }[char.upper()]


class Square(int):
    @staticmethod
    def from_file_rank(file: int, rank: int):
        return Square(file_rank_to_index(file, rank))

    @staticmethod
    def from_coord(coord: str):
        file = char_to_file(coord[0].upper())
        rank = int(coord[1]) - 1
        return Square(file_rank_to_index(file, rank))

    @property
    def file(self):
        return self % 8

    @property
    def rank(self):
        return int(self / 8)

    @property
    def colour(self):
        return WHITE if self % 2 == 0 else BLACK

    @property
    def name(self):
        return FILE_NAMES[self.file] + RANK_NAMES[self.rank]

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)


def square_distance(a: Square, b: Square) -> int:
    """
    Gets the distance (i.e., the number of king steps) from square *a* to *b*.
    """
    a, b = Square(a), Square(b)
    return max(abs(a.file - b.file), abs(a.rank - b.rank))


def mirror_square(square: Square, vertical: bool = True) -> Square:
    """Returns a square position as if the board was mirrored vertically."""
    if vertical:
        return Square(square ^ 56)
    else:
        return Square(square ^ 7)


def _mirror_list(_list):
    """Returns a list of 64 elements representing the board as if the board were mirrored vertically."""
    _new_list = [0] * len(_list)
    for i, val in enumerate(_list):
        _new_list[SQUARES_VFLIP[i]] = _list[i]
    return _new_list


SQUARES = [
    A1, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8,
] = [Square(i) for i in range(64)]

SQUARES_VFLIP = [mirror_square(sq, True) for sq in SQUARES]

PAWN_POSITION_VALUES = {
    WHITE: _mirror_list(PAWN_POSITION_BASE_VALUES),
    BLACK: PAWN_POSITION_BASE_VALUES,
}

KNIGHT_POSITION_VALUES = {
    WHITE: _mirror_list(KNIGHT_POSITION_BASE_VALUES),
    BLACK: KNIGHT_POSITION_BASE_VALUES,
}

BISHOP_POSITION_VALUES = {
    WHITE: _mirror_list(BISHOP_POSITION_BASE_VALUES),
    BLACK: BISHOP_POSITION_BASE_VALUES,
}

ROOK_POSITION_VALUES = {
    WHITE: _mirror_list(ROOK_POSITION_BASE_VALUES),
    BLACK: ROOK_POSITION_BASE_VALUES,
}

QUEEN_POSITION_VALUES = {
    WHITE: _mirror_list(QUEEN_POSITION_BASE_VALUES),
    BLACK: QUEEN_POSITION_BASE_VALUES,
}

KING_POSITION_VALUES = {
    WHITE: _mirror_list(KING_POSITION_BASE_VALUES),
    BLACK: KING_POSITION_BASE_VALUES,
}
