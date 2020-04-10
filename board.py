from typing import Dict, List, Set, Union

import log
import position
from position import Position
from pieces import Piece, Rook, Knight, Bishop, Queen, King, Pawn
from constants import WHITE, BLACK, STARTING_STATE
from exceptions import IllegalMove

FEN_PIECES = {
    'r': Rook,
    'n': Knight,
    'b': Bishop,
    'q': Queen,
    'k': King,
    'p': Pawn,
}


class Square:
    def __init__(self, pos: Position, piece: Piece = None):
        self.pos = pos
        self.piece = piece

    @property
    def is_occupied(self):
        return self.piece is not None

    def __str__(self):
        if self.is_occupied:
            return str(self.piece)
        else:
            return str(self.pos)

class Board:
    def __init__(self, state: str = STARTING_STATE):
        self.pieces = from_fen(state)

    def move(self, start_pos: Position, end_pos: Position):
        """Moves a piece from the start position to the end position if legal."""
        if start_pos == end_pos or not start_pos.in_board or not end_pos.in_board:
            raise IllegalMove(f"{start_pos} to {end_pos} is an illegal move.")

        squares = self.squares
        start_square = squares[start_pos.index]
        assert start_square.is_occupied, f"No piece at {start_pos}"

        if not end_pos in start_square.piece.legal_moves(self):
            raise IllegalMove(f"{start_pos} to {end_pos} is an illegal move.")

        end_square = squares[end_pos.index]
        if end_square.is_occupied:
            self.pieces.remove(end_square.piece)
        start_square.piece.pos = end_pos  # Move piece

    def is_in_check(self, colour: str = WHITE) -> bool:
        """Checks if the associated King is in check on the board."""
        king = filter(
            lambda p: p.TYPE == 'k' and p.colour == colour,
            self.pieces
        )
        king = list(king)[0]

        # Loop through legal moves of opposing side and return true if they contain the King
        for piece in filter(
            lambda p: p.colour != colour,
            self.pieces
        ):
            if king.pos in piece.legal_moves(self):
                return True
        return False

    @property
    def _squares_by_rank(self) -> Dict[int, List[Square]]:
        """All squares of the board, organised by rank."""
        by_rank = {}
        for square in self.squares:
            rank = square.pos.rank
            by_rank[rank] = by_rank.get(rank, [])
            by_rank[rank].append(square)
        return by_rank

    @property
    def squares(self) -> List[Square]:
        """All squares of the board, in order from A1, B1... G8, H8."""
        all_squares = []
        i = 0
        for piece in sorted(self.pieces):
            while i < piece.pos.index:
                all_squares.append(
                    Square(pos=position.from_index(i))
                )
                i += 1
            all_squares.append(Square(pos=position.from_index(i), piece=piece))
            i += 1
        return all_squares

    @property
    def fen(self) -> str:
        """
        Returns the board's current state in FE Notation.
        (https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation)
        """
        fen_str = ''
        by_rank = self._squares_by_rank
        for rank in sorted(by_rank.keys(), reverse=True):
            blank_counter = 0
            for square in by_rank[rank]:
                if square.is_occupied:
                    if blank_counter > 0:
                        fen_str += str(blank_counter)
                        blank_counter = 0
                    fen_str += square.piece.code
                else:
                    blank_counter += 1

            if blank_counter > 0:
                fen_str += str(blank_counter)

            if rank > 0:
                fen_str += '/'
        return fen_str

    def __str__(self) -> str:
        by_rank = self._squares_by_rank
        board_str = ''
        for rank in sorted(by_rank.keys(), reverse=True):
            board_str += f'\n{rank + 1} '
            for square in by_rank[rank]:
                if square.is_occupied:
                    board_str += f'[{square.piece.icon}]'
                else:
                    board_str += '[ ]'
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
                pos=Position(file, rank),
                colour=WHITE if char.isupper() else BLACK,
            )
            pieces.add(piece)
            file += 1
    return pieces
