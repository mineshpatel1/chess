from typing import Tuple

from engine.constants import (
    WHITE,
    BLACK,
    FILE_NAMES,
    RANK_NAMES,
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