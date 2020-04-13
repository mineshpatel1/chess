from typing import Tuple

from engine.constants import WHITE, BLACK


class Position:
    def __init__(
        self,
        file: int,
        rank: int,
    ):
        """
        Defines a position on the board. Optionally can specify rank and file integers or the index in the 1-D array.
        """

        self.file = file
        self.rank = rank

    @property
    def colour(self) -> str:
        odd_rank = self.rank % 2
        return BLACK if (self.index + odd_rank) % 2 == 0 else WHITE

    @property
    def index(self) -> int:
        return file_rank_to_index(self.file, self.rank)

    @property
    def in_board(self):
        return 0 <= self.file < 8 and 0 <= self.rank < 8

    def __str__(self) -> str:
        return f'{file_to_char(self.file)}{self.rank + 1}'

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other) -> bool:
        return self.file == other.file and self.rank == other.rank

    def __hash__(self) -> int:
        return hash((self.file, self.rank))

    def __lt__(self, other) -> bool:
        return self.index < other.index

    def __le__(self, other) -> bool:
        return self.index <= other.index

    def __gt__(self, other) -> bool:
        return self.index > other.index

    def __ge__(self, other) -> bool:
        return self.index >= other.index


def index_to_rankfile(idx: int) -> Tuple[int, int]:
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


def from_index(index: int) -> Position:
    file, rank = index_to_rankfile(index)
    return Position(file, rank)


def from_coord(coord: str) -> Position:
    assert len(coord) == 2, "Coordinates must be specified as a letter and integer, e.g. A5."
    file = char_to_file(coord[0])
    rank = int(coord[1]) - 1
    return Position(file, rank)
