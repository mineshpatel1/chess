from typing import Tuple


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
    def index(self) -> int:
        return rankfile_to_index(self.file, self.rank)

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
    """Converts an integer position into the corresponding rank and file (in that order)."""
    assert 0 <= idx < 64, "Board index must be between 0 and 63."
    return idx % 8, int(idx / 8)


def rankfile_to_index(file: int, rank: int) -> int:
    assert 0 <= file < 8 and 0 <= rank < 8, "Rank and file must be between 0 and 7."
    return (rank * 8) + (file % 8)


def file_to_char(file: int) -> str:
    return ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'][file]


def from_index(index: int) -> Position:
    file, rank = index_to_rankfile(index)
    return Position(file, rank)
