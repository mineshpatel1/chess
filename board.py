from typing import List, Set, Union

import log
import position
from position import Position
from pieces import Piece, Rook, Knight, Bishop, Queen, King, Pawn


FEN_PIECES = {
    'r': Rook,
    'n': Knight,
    'b': Bishop,
    'q': Queen,
    'k': King,
    'p': Pawn,
}
STARTING_STATE = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'


class Board:
    def __init__(self, state: str = STARTING_STATE):
        self.pieces = from_fen(state)

    @property
    def squares(self) -> List[Union[Piece, Position]]:
        all_squares = []
        i = 0
        for piece in sorted(self.pieces):
            while i < piece.pos.index:
                all_squares.append(Position(*position.index_to_rankfile(i)))
                i += 1
            all_squares.append(piece)
            i += 1
        return all_squares

    def __str__(self) -> str:
        by_rank = {}
        for piece in self.squares:
            rank = piece.rank if isinstance(piece, Position) else piece.pos.rank
            by_rank[rank] = by_rank.get(rank, [])
            by_rank[rank].append(piece)

        board_str = ""
        for rank in sorted(by_rank.keys(), reverse=True):
            board_str += f'\n{rank + 1} '
            for piece in by_rank[rank]:
                if isinstance(piece, Position):
                    board_str += '[ ]'
                else:
                    board_str += f'[{piece.icon}]'
        board_str += '\n   A  B  C  D  E  F  G  H '
        return board_str


def from_fen(fen: str = STARTING_STATE) -> Set[Piece]:
    rank = 7
    file = 0
    pieces = set()

    for char in fen.split(' ')[0]:
        if char.isdigit():
            file += int(char)
        elif char == '/':
            rank -= 1
            file = 0
        else:
            assert char.lower() in list(FEN_PIECES.keys()), f'{char} is not a valid piece in FEN notation.'
            piece = FEN_PIECES[char.lower()](
                is_white=char.isupper(),
                pos=Position(file, rank),
            )
            pieces.add(piece)
            file += 1
    return pieces
