from typing import List, Iterable

from game.square import *
from game.constants import (
    BLACK,
    WHITE,

    NORTH,
    NORTHEAST,
    EAST,
    SOUTHEAST,
    SOUTH,
    SOUTHWEST,
    WEST,
    NORTHWEST,
)


Bitboard = int


def msb(x: Bitboard) -> int:
    """Returns most significant bit."""
    return x.bit_length() - 1


def lsb(x: Bitboard) -> int:
    """Returns least significant bit."""
    return msb(x & -x)


def binary_str(i: int) -> str:
    """Returns a base 2 integer as a binary string of length 64."""
    s = '{0:b}'.format(i)
    s = ('0' * (64 - len(s))) + s  # Pad with zeroes
    assert len(s) == 64
    return s


def bit_count(i: Bitboard) -> int:
    """Number of bits set to 1 in the given integer."""
    count = 0
    while i:
        i &= i - 1
        count += 1
    return count


def bitboard_to_squares(bb: Bitboard) -> Iterable[Square]:
    """Returns squares populated in a given bitboard."""
    while bb:
        r = bb.bit_length() - 1
        yield Square(r)
        bb ^= BB_SQUARES[r]


def bitboard_to_str(bb: Bitboard) -> str:
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

BB_DIRECTIONS = {
    NORTH: 8,
    NORTHEAST: 9,
    EAST: 1,
    SOUTHEAST: -7,
    SOUTH: -8,
    SOUTHWEST: -9,
    WEST: -1,
    NORTHWEST: 7,
}

# 64-bit representations of game positions
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


def _gen_moves(intervals: Iterable[Tuple[int, int]]) -> List[Bitboard]:
    bbs = []
    for sq in SQUARES:
        moves = BB_EMPTY
        for i, j in intervals:
            _file = sq.file + i
            _rank = sq.rank + j
            if 0 <= _file < 8 and 0 <= _rank < 8:  # Checks within the game
                _sq = file_rank_to_index(sq.file + i, sq.rank + j)
                moves |= BB_SQUARES[_sq]
        bbs.append(moves)
    return bbs


def _gen_rays(file_adjust, rank_adjust) -> List[Bitboard]:
    def _in_board(_file, _rank):
        return 0 <= _file < 8 and 0 <= _rank < 8

    bbs = []
    for sq in SQUARES:
        moves = BB_EMPTY
        _file = sq.file
        _rank = sq.rank
        while _in_board(_file, _rank):  # Still in game
            _file = _file + file_adjust
            _rank = _rank + rank_adjust
            if _in_board(_file, _rank):
                _sq = file_rank_to_index(_file, _rank)
                moves |= BB_SQUARES[_sq]
        bbs.append(moves)
    return bbs


def _double_pawn_advances(pawn_moves):
    for _rank in (1, 6):
        for _file in range(8):
            i = file_rank_to_index(_file, _rank)
            if _rank == 1:  # White pawns
                single_move = pawn_moves[WHITE]['single'][i]
                pawn_moves[WHITE]['double'][i] = single_move << 8  # Double move
            else:
                single_move = pawn_moves[BLACK]['single'][i]
                pawn_moves[BLACK]['double'][i] = single_move >> 8  # Double move
    return pawn_moves


BB_ORIGINAL_ROOKS = {
    WHITE: (BB_A1 | BB_H1),
    BLACK: (BB_A8 | BB_H8),
}
BB_KNIGHT_MOVES = _gen_moves(((1, 2), (1, -2), (-1, 2), (-1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1)))
BB_KING_MOVES = _gen_moves(((1, 1), (0, 1), (1, 0), (-1, -1), (-1, 0), (0, -1), (1, -1), (-1, 1)))
BB_PAWN_ATTACKS = {
    WHITE: _gen_moves(((1, 1), (-1, 1))),
    BLACK: _gen_moves(((1, -1), (-1, -1))),
}
BB_PAWN_MOVES = {
    WHITE: {
        'single': _gen_moves(((0, 1),)),  # Single advance
        'double': [0] * 64,
    },
    BLACK: {
        'single': _gen_moves(((0, -1),)),
        'double': [0] * 64,
    },
}

BB_PAWN_MOVES = _double_pawn_advances(BB_PAWN_MOVES)

BB_RAYS = {
    NORTH: _gen_rays(0, 1),
    NORTHEAST: _gen_rays(1, 1),
    EAST: _gen_rays(1, 0),
    SOUTHEAST: _gen_rays(1, -1),
    SOUTH: _gen_rays(0, -1),
    SOUTHWEST: _gen_rays(-1, -1),
    WEST: _gen_rays(-1, 0),
    NORTHWEST: _gen_rays(-1, 1),
}

BB_CARDINALS = [
    BB_RAYS[NORTH][i] | BB_RAYS[EAST][i] | BB_RAYS[SOUTH][i] | BB_RAYS[WEST][i]
    for i in range(64)
]

BB_DIAGONALS = [
    BB_RAYS[NORTHEAST][i] | BB_RAYS[SOUTHEAST][i] | BB_RAYS[SOUTHWEST][i] | BB_RAYS[NORTHWEST][i]
    for i in range(64)
]

BB_BETWEEN = []  # type: List[List[int]]


def _calc_between(rays, _from_sq, _to_sq):
    possible = rays[_from_sq]
    blockers = possible & BB_SQUARES[_to_sq]
    blocked_paths = rays[lsb(blockers)] | rays[msb(blockers)]
    return (possible & ~blocked_paths) ^ BB_SQUARES[_to_sq]


for _from_sq in SQUARES:
    BB_BETWEEN.append([])
    for _to_sq in SQUARES:
        bb_to_sq = BB_SQUARES[_to_sq]

        between = None
        for _direction in BB_RAYS:
            if BB_RAYS[_direction][_from_sq] & bb_to_sq:
                between = _calc_between(BB_RAYS[_direction], _from_sq, _to_sq)
                continue

        if between:
            BB_BETWEEN[_from_sq].append(between)
        else:
            BB_BETWEEN[_from_sq].append(BB_EMPTY)

