from typing import Tuple

STARTING_STATE = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'


class Colour:
    WHITE = 'WHITE'
    BLACK = 'BLACK'


class Position:
    @staticmethod
    def _index_to_rankfile(idx: int) -> Tuple[int, int]:
        """Converts an integer position into the corresponding rank and file (in that order)."""
        assert 0 <= idx < 64, "Board index must be between 0 and 63."
        return idx % 8, int(idx / 8)

    @staticmethod
    def _rankfile_to_index(file: int, rank: int) -> int:
        assert 0 <= file < 8 and 0 <= rank < 8, "Rank and file must be between 0 and 7."
        return (rank * 8) + (file % 8)

    @staticmethod
    def _file_to_char(file: int) -> str:
        return ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'][file]

    def __init__(
        self,
        file: int = None,
        rank: int = None,
        index: int = None,
    ):
        """
        Defines a position on the board. Optionally can specify rank and file integers or the index in the 1-D array.
        """

        if index is not None:
            self.index = index
            self.file, self.rank = self._index_to_rankfile(index)
        else:
            self.file = file
            self.rank = rank
            self.index = self._rankfile_to_index(file, rank)

    def __str__(self) -> str:
        return f'{self._file_to_char(self.file)}{self.rank + 1}'

    def __repr__(self) -> str:
        return str(self)


class Piece:
    def __init__(self, is_white: bool, pos: Position):
        self.is_white = is_white
        self.pos = pos

    @property
    def possible_moves(self):
        return None


class Pawn:
    @property
    def possible_moves(self):
        pass


class Cell:
    def __init__(self):
        self.state = None


class Board:
    def __init__(self, state=STARTING_STATE):
        self.cells = []
        for _ in range(64):
            self.cells.append(None)

