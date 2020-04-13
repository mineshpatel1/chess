from typing import List, Tuple, Optional, Iterable

import log
from engine.position import file_rank_to_index
from engine.constants import (
    Colour,
    WHITE,
    BLACK,

    FILE_NAMES,
    RANK_NAMES,
    STARTING_STATE,

    PieceType,
    PAWN,
    ROOK,
    KNIGHT,
    BISHOP,
    QUEEN,
    KING,
    PIECE_ICONS,
    PIECE_NAMES,
    PIECE_TYPES,
    PIECE_POINTS,
)


def binary_str(i: int):
    """Returns a base 2 integer as a binary string of length 64"""
    s = '{0:b}'.format(i)
    s = ('0' * (64 - len(s))) + s  # Pad with zeroes
    assert len(s) == 64
    return s


Bitboard = int


def bitboard_to_squares(bb):
        while bb:
            r = bb.bit_length() - 1
            yield Square(r)
            bb ^= BB_SQUARES[r]


def bitboard_to_str(bb):
    """Prints a visual representation of the occupation represented by the input bitboard integer."""
    board_str = ''
    rank = 8
    for sq in SQUARES_VFLIP:
        if rank > sq.rank:
            board_str += f'\n{sq.rank + 1} '
            rank = sq.rank

        if binary_str(bb)[63 - sq] == '1':
            board_str += '[•]'
        else:
            board_str += '[ ]'
    board_str += '\n   A  B  C  D  E  F  G  H '
    return board_str


class Square(int):
    @staticmethod
    def from_file_rank(file: int, rank: int):
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


def mirror_square(square: Square, vertical: bool = True) -> Square:
    """Returns a square position as if the board was mirrored vertically."""
    if vertical:
        return Square(square ^ 56)
    else:
        return Square(square ^ 7)


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

BB_DIRECTIONS = {
    'n': 8,
    'ne': 9,
    'e': 1,
    'se': -7,
    's': -8,
    'sw': -9,
    'w': -1,
    'nw': 7,
}

# 64-bit representations of board positions
BB_EMPTY = 0
BB_BOARD = (2 ** 64) - 1
BB_WHITE_SQUARES = 6172840429334713770
BB_BLACK_SQUARES = 12273903644374837845

BB_SQUARES = [
    BB_A1, BB_B1, BB_C1, BB_D1, BB_E1, BB_F1, BB_G1, BB_H1,
    BB_A2, BB_B2, BB_C2, BB_D2, BB_E2, BB_F2, BB_G2, BB_H2,
    BB_A3, BB_B3, BB_C3, BB_D3, BB_E3, BB_F3, BB_G3, BB_H3,
    BB_A4, BB_B4, BB_C4, BB_D4, BB_E4, BB_F4, BB_G4, BB_H4,
    BB_A5, BB_B5, BB_C5, BB_D5, BB_E5, BB_F5, BB_G5, BB_H5,
    BB_A6, BB_B6, BB_C6, BB_D6, BB_E6, BB_F6, BB_G6, BB_H6,
    BB_A7, BB_B7, BB_C7, BB_D7, BB_E7, BB_F7, BB_G7, BB_H7,
    BB_A8, BB_B8, BB_C8, BB_D8, BB_E8, BB_F8, BB_G8, BB_H8,
] = [1 << sq for sq in SQUARES]

BB_FILES = [
    BB_FILE_A,
    BB_FILE_B,
    BB_FILE_C,
    BB_FILE_D,
    BB_FILE_E,
    BB_FILE_F,
    BB_FILE_G,
    BB_FILE_H,
] = [
    # Create a set of the first file (A) and then iteratively shift all positions eastward 1 square
    (BB_A1 | BB_A2 | BB_A3 | BB_A4 | BB_A5 | BB_A6 | BB_A7 | BB_A8) << (i * BB_DIRECTIONS['e'])
    for i in range(8)
]

BB_RANKS = [
    BB_RANK_1,
    BB_RANK_2,
    BB_RANK_3,
    BB_RANK_4,
    BB_RANK_5,
    BB_RANK_6,
    BB_RANK_7,
    BB_RANK_8,
] = [
    # Create a set of the first rank (1) and then iteratively shift all positions northward 1 square
    (BB_A1 | BB_B1 | BB_C1 | BB_D1 | BB_E1 | BB_F1 | BB_G1 | BB_H1) << (i * BB_DIRECTIONS['n'])
    for i in range(8)
]


def _gen_moves(intervals) -> List[Bitboard]:
    bbs = []
    for sq in SQUARES:
        moves = BB_EMPTY
        for i, j in intervals:
            _file = sq.file + i
            _rank = sq.rank + j
            if 0 <= _file < 8 and 0 <= _rank < 8:  # Checks within the board
                _sq = file_rank_to_index(sq.file + i, sq.rank + j)
                moves |= BB_SQUARES[_sq]
        bbs.append(moves)
    return bbs


KNIGHT_MOVES = _gen_moves(((1, 2), (1, -2), (-1, 2), (-1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1)))
KING_MOVES = _gen_moves(((1, 1), (0, 1), (1, 0), (-1, -1), (-1, 0), (0, -1), (1, -1), (-1, 1)))


class Piece:
    TYPE = None
    BASE_VALUE = 0
    CASTLE_POSITIONS = {
        WHITE: {},
        BLACK: {},
    }

    def __init__(self, piece_type: PieceType, colour: str = WHITE):
        assert colour in [WHITE, BLACK], f"Invalid colour: {colour} chosen."
        assert piece_type in PIECE_TYPES, f"Invalid piece type: {piece_type} chosen."
        self.colour = colour
        self.type = piece_type

    @property
    def name(self) -> str:
        return f'{self.colour.title()} {PIECE_NAMES[self.type]}'

    @property
    def code(self) -> str:
        return self.type.upper() if self.colour == WHITE else self.type.lower()

    @property
    def icon(self) -> str:
        return PIECE_ICONS[self.code]

    @property
    def value(self) -> int:
        modifier = 1 if self.colour == WHITE else -1
        return modifier * PIECE_POINTS[self.type]

    def __str__(self) -> str:
        return f'{self.icon}'

    def __repr__(self) -> str:
        return str(self)


class Move:
    def __init__(self, from_square: Square, to_square: Square):
        self.from_square = Square(from_square)
        self.to_square = Square(to_square)

    @property
    def uci(self):
        return f'{str(self.from_square).lower()}{str(self.to_square).lower()}'

    def __str__(self):
        return f'{self.from_square} -> {self.to_square}'


class Board:
    def __init__(self, fen: str = STARTING_STATE):
        self._clear()
        self._set_from_fen(fen)

    def _clear(self):
        """Defines an empty bitbaord."""
        self.pawns = {
            WHITE: BB_EMPTY,
            BLACK: BB_EMPTY,
        }
        self.knights = {
            WHITE: BB_EMPTY,
            BLACK: BB_EMPTY,
        }
        self.bishops = {
            WHITE: BB_EMPTY,
            BLACK: BB_EMPTY,
        }
        self.rooks = {
            WHITE: BB_EMPTY,
            BLACK: BB_EMPTY,
        }
        self.queens = {
            WHITE: BB_EMPTY,
            BLACK: BB_EMPTY,
        }
        self.kings = {
            WHITE: BB_EMPTY,
            BLACK: BB_EMPTY,
        }

        self.occupied_colour = {
            WHITE: BB_EMPTY,
            BLACK: BB_EMPTY,
        }
        self.occupied = BB_EMPTY

    def _set_from_fen(self, fen: str):
        rank = 7
        file = 0
        components = fen.split(' ')
        for char in components[0]:
            if char.isdigit():
                file += int(char)
            elif char == '/':
                rank -= 1
                file = 0
            else:
                assert char.lower() in PIECE_TYPES, f'{char} is not a valid piece in FEN notation.'
                self.place_piece(
                    Square.from_file_rank(file, rank),
                    char.lower(),
                    WHITE if char.isupper() else BLACK,
                )
                file += 1

    def _attacks_from_square(self, square: Square, colour: Colour) -> Optional[Bitboard]:
        bb_sq = BB_SQUARES[square]

        if self.pawns[colour] & bb_sq:
            return None
        elif self.rooks[colour] & bb_sq:
            return None
        elif self.knights[colour] & bb_sq:
            return KNIGHT_MOVES[square]
        elif self.bishops[colour] & bb_sq:
            return None
        elif self.queens[colour] & bb_sq:
            return None
        elif self.kings[colour] & bb_sq:
            return KING_MOVES[square]

    def _pseudo_legal_moves(self, colour: Colour) -> Iterable[Move]:
        for from_square in bitboard_to_squares(
            self.occupied_colour[colour]
        ):
            attack_moves = self._attacks_from_square(from_square, colour)
            if attack_moves:
                attack_moves = attack_moves & ~self.occupied_colour[colour]
                for to_square in bitboard_to_squares(attack_moves):
                    yield Move(from_square, to_square)

    def place_piece(self, square: Square, piece_type: PieceType, colour: Colour):
        """Place a piece of a given colour on a square of the board."""
        mask = BB_SQUARES[square]

        if piece_type.lower() == PAWN:
            self.pawns[colour] |= mask
        elif piece_type.lower() == ROOK:
            self.rooks[colour] |= mask
        elif piece_type.lower() == KNIGHT:
            self.knights[colour] |= mask
        elif piece_type.lower() == BISHOP:
            self.bishops[colour] |= mask
        elif piece_type.lower() == QUEEN:
            self.queens[colour] |= mask
        elif piece_type.lower() == KING:
            self.kings[colour] |= mask

        self.occupied_colour[colour] |= mask
        self.occupied |= mask

    def piece_at(self, square: Square) -> Optional[Piece]:
        """Optionally returns the piece occupying the given square."""
        mask = BB_SQUARES[square]

        if not self.occupied & mask:
            return None

        for colour in [WHITE, BLACK]:
            if self.pawns[colour] & mask:
                return Piece(PAWN, colour)
            elif self.rooks[colour] & mask:
                return Piece(ROOK, colour)
            elif self.knights[colour] & mask:
                return Piece(KNIGHT, colour)
            elif self.bishops[colour] & mask:
                return Piece(BISHOP, colour)
            elif self.queens[colour] & mask:
                return Piece(QUEEN, colour)
            elif self.kings[colour] & mask:
                return Piece(KING, colour)

    def __str__(self):
        board_str = ''
        rank = 8
        for sq in SQUARES_VFLIP:
            if rank > sq.rank:
                board_str += f'\n{sq.rank + 1} '
                rank = sq.rank

            piece = self.piece_at(sq)
            if piece:
                board_str += f'[{piece.icon}]'
            else:
                board_str += '[ ]'
        board_str += '\n   A  B  C  D  E  F  G  H '
        return board_str
