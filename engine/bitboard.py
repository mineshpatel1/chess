from typing import Tuple

import log
from engine.position import index_to_rankfile, file_to_char
from engine.constants import (
    WHITE,
    BLACK,
    FILE_NAMES,
    RANK_NAMES,
)


class Square(int):
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


def mirror_square(square: Square) -> Square:
    """Returns a square position as if the board was mirrored vertically."""
    return Square(square ^ 56)


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

SQUARES_FLIPPED = [mirror_square(sq) for sq in SQUARES]

DIRECTIONS = {
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
EMPTY = 0
BOARD = (2 ** 64) - 1
WHITE_SQUARES = 6172840429334713770
BLACK_SQUARES = 12273903644374837845

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
    (BB_A1 | BB_A2 | BB_A3 | BB_A4 | BB_A5 | BB_A6 | BB_A7 | BB_A8) << i  # Bitwise shift the first file
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
    (BB_A1 | BB_B1 | BB_C1 | BB_D1 | BB_E1 | BB_F1 | BB_G1 | BB_H1) << (i * 8)  # Bitwise shift the first rank
    for i in range(8)
]


def bb_str(bb: int):
    s = '{0:b}'.format(bb)
    s = ('0' * (64 - len(s))) + s  # Pad with zeroes
    assert len(s) == 64
    return s


def bitboard_to_str(bb: int):
    board_str = ''
    rank = 8
    for sq in SQUARES_FLIPPED:
        if rank > sq.rank:
            board_str += f'\n{sq.rank + 1} '
            rank = sq.rank

        if bb_str(bb)[63 - sq] == '1':
            board_str += '[•]'
        else:
            board_str += '[ ]'
    board_str += '\n   A  B  C  D  E  F  G  H '
    return board_str

