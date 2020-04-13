import copy
import enum
import math
import re
import itertools
import typing

from typing import ClassVar, Callable, Counter, Dict, Generic, Hashable, Iterable, Iterator, List, Mapping, Optional, SupportsInt, Tuple, Type, TypeVar, Union


Color = bool
COLORS = [WHITE, BLACK] = [True, False]
COLOR_NAMES = ["black", "white"]

PieceType = int
PIECE_TYPES = [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING] = range(1, 7)
PIECE_SYMBOLS = [None, "p", "n", "b", "r", "q", "k"]
PIECE_NAMES = [None, "pawn", "knight", "bishop", "rook", "queen", "king"]

def piece_symbol(piece_type: PieceType, _PIECE_SYMBOLS: List[Optional[str]] = PIECE_SYMBOLS) -> str:
    return typing.cast(str, _PIECE_SYMBOLS[piece_type])

def piece_name(piece_type: PieceType) -> str:
    return typing.cast(str, PIECE_NAMES[piece_type])

UNICODE_PIECE_SYMBOLS = {
    "R": u"♖", "r": u"♜",
    "N": u"♘", "n": u"♞",
    "B": u"♗", "b": u"♝",
    "Q": u"♕", "q": u"♛",
    "K": u"♔", "k": u"♚",
    "P": u"♙", "p": u"♟",
}

FILE_NAMES = ["a", "b", "c", "d", "e", "f", "g", "h"]

RANK_NAMES = ["1", "2", "3", "4", "5", "6", "7", "8"]

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
"""The FEN for the standard chess starting position."""

STARTING_BOARD_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
"""The board part of the FEN for the standard chess starting position."""


class Status(enum.IntFlag):
    VALID = 0
    NO_WHITE_KING = 1
    NO_BLACK_KING = 2
    TOO_MANY_KINGS = 4
    TOO_MANY_WHITE_PAWNS = 8
    TOO_MANY_BLACK_PAWNS = 16
    PAWNS_ON_BACKRANK = 32
    TOO_MANY_WHITE_PIECES = 64
    TOO_MANY_BLACK_PIECES = 128
    BAD_CASTLING_RIGHTS = 256
    INVALID_EP_SQUARE = 512
    OPPOSITE_CHECK = 1024
    EMPTY = 2048
    RACE_CHECK = 4096
    RACE_OVER = 8192
    RACE_MATERIAL = 16384
    TOO_MANY_CHECKERS = 32768

STATUS_VALID = Status.VALID
STATUS_NO_WHITE_KING = Status.NO_WHITE_KING
STATUS_NO_BLACK_KING = Status.NO_BLACK_KING
STATUS_TOO_MANY_KINGS = Status.TOO_MANY_KINGS
STATUS_TOO_MANY_WHITE_PAWNS = Status.TOO_MANY_WHITE_PAWNS
STATUS_TOO_MANY_BLACK_PAWNS = Status.TOO_MANY_BLACK_PAWNS
STATUS_PAWNS_ON_BACKRANK = Status.PAWNS_ON_BACKRANK
STATUS_TOO_MANY_WHITE_PIECES = Status.TOO_MANY_WHITE_PIECES
STATUS_TOO_MANY_BLACK_PIECES = Status.TOO_MANY_BLACK_PIECES
STATUS_BAD_CASTLING_RIGHTS = Status.BAD_CASTLING_RIGHTS
STATUS_INVALID_EP_SQUARE = Status.INVALID_EP_SQUARE
STATUS_OPPOSITE_CHECK = Status.OPPOSITE_CHECK
STATUS_EMPTY = Status.EMPTY
STATUS_RACE_CHECK = Status.RACE_CHECK
STATUS_RACE_OVER = Status.RACE_OVER
STATUS_RACE_MATERIAL = Status.RACE_MATERIAL
STATUS_TOO_MANY_CHECKERS = Status.TOO_MANY_CHECKERS


Square = int
SQUARES = [
    A1, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8,
] = range(64)

SQUARE_NAMES = [f + r for r in RANK_NAMES for f in FILE_NAMES]

def square(file_index: int, rank_index: int) -> Square:
    """Gets a square number by file and rank index."""
    return rank_index * 8 + file_index

def square_file(square: Square) -> int:
    """Gets the file index of the square where ``0`` is the a-file."""
    return square & 7

def square_rank(square: Square) -> int:
    """Gets the rank index of the square where ``0`` is the first rank."""
    return square >> 3

def square_name(square: Square) -> str:
    """Gets the name of the square, like ``a3``."""
    return SQUARE_NAMES[square]

def square_distance(a: Square, b: Square) -> int:
    """
    Gets the distance (i.e., the number of king steps) from square *a* to *b*.
    """
    return max(abs(square_file(a) - square_file(b)), abs(square_rank(a) - square_rank(b)))

def square_mirror(square: Square) -> Square:
    """Mirrors the square vertically."""
    return square ^ 0x38

SQUARES_180 = [square_mirror(sq) for sq in SQUARES]


Bitboard = int
BB_EMPTY = 0
BB_ALL = 0xffff_ffff_ffff_ffff

BB_SQUARES = [
    BB_A1, BB_B1, BB_C1, BB_D1, BB_E1, BB_F1, BB_G1, BB_H1,
    BB_A2, BB_B2, BB_C2, BB_D2, BB_E2, BB_F2, BB_G2, BB_H2,
    BB_A3, BB_B3, BB_C3, BB_D3, BB_E3, BB_F3, BB_G3, BB_H3,
    BB_A4, BB_B4, BB_C4, BB_D4, BB_E4, BB_F4, BB_G4, BB_H4,
    BB_A5, BB_B5, BB_C5, BB_D5, BB_E5, BB_F5, BB_G5, BB_H5,
    BB_A6, BB_B6, BB_C6, BB_D6, BB_E6, BB_F6, BB_G6, BB_H6,
    BB_A7, BB_B7, BB_C7, BB_D7, BB_E7, BB_F7, BB_G7, BB_H7,
    BB_A8, BB_B8, BB_C8, BB_D8, BB_E8, BB_F8, BB_G8, BB_H8
] = [1 << sq for sq in SQUARES]

BB_CORNERS = BB_A1 | BB_H1 | BB_A8 | BB_H8
BB_CENTER = BB_D4 | BB_E4 | BB_D5 | BB_E5

BB_LIGHT_SQUARES = 0x55aa_55aa_55aa_55aa
BB_DARK_SQUARES = 0xaa55_aa55_aa55_aa55

BB_FILES = [
    BB_FILE_A,
    BB_FILE_B,
    BB_FILE_C,
    BB_FILE_D,
    BB_FILE_E,
    BB_FILE_F,
    BB_FILE_G,
    BB_FILE_H
] = [0x0101_0101_0101_0101 << i for i in range(8)]

BB_RANKS = [
    BB_RANK_1,
    BB_RANK_2,
    BB_RANK_3,
    BB_RANK_4,
    BB_RANK_5,
    BB_RANK_6,
    BB_RANK_7,
    BB_RANK_8
] = [0xff << (8 * i) for i in range(8)]

BB_BACKRANKS = BB_RANK_1 | BB_RANK_8


def lsb(bb: Bitboard) -> int:
    return (bb & -bb).bit_length() - 1

def scan_forward(bb: Bitboard) -> Iterator[Square]:
    while bb:
        r = bb & -bb
        yield r.bit_length() - 1
        bb ^= r

def msb(bb: Bitboard) -> int:
    return bb.bit_length() - 1

def scan_reversed(bb: Bitboard, *, _BB_SQUARES: List[Bitboard] = BB_SQUARES) -> Iterator[Square]:
    while bb:
        r = bb.bit_length() - 1
        yield r
        bb ^= _BB_SQUARES[r]

def popcount(bb: Bitboard, *, _bin: Callable[[int], str] = bin) -> int:
    return _bin(bb).count("1")

def flip_vertical(bb: Bitboard) -> Bitboard:
    # https://www.chessprogramming.org/Flipping_Mirroring_and_Rotating#FlipVertically
    bb = ((bb >> 8) & 0x00ff_00ff_00ff_00ff) | ((bb & 0x00ff_00ff_00ff_00ff) << 8)
    bb = ((bb >> 16) & 0x0000_ffff_0000_ffff) | ((bb & 0x0000_ffff_0000_ffff) << 16)
    bb = (bb >> 32) | ((bb & 0x0000_0000_ffff_ffff) << 32)
    return bb

def flip_horizontal(bb: Bitboard) -> Bitboard:
    # https://www.chessprogramming.org/Flipping_Mirroring_and_Rotating#MirrorHorizontally
    bb = ((bb >> 1) & 0x5555_5555_5555_5555) | ((bb & 0x5555_5555_5555_5555) << 1)
    bb = ((bb >> 2) & 0x3333_3333_3333_3333) | ((bb & 0x3333_3333_3333_3333) << 2)
    bb = ((bb >> 4) & 0x0f0f_0f0f_0f0f_0f0f) | ((bb & 0x0f0f_0f0f_0f0f_0f0f) << 4)
    return bb

def flip_diagonal(bb: Bitboard) -> Bitboard:
    # https://www.chessprogramming.org/Flipping_Mirroring_and_Rotating#FlipabouttheDiagonal
    t = (bb ^ (bb << 28)) & 0x0f0f_0f0f_0000_0000
    bb = bb ^ (t ^ (t >> 28))
    t = (bb ^ (bb << 14)) & 0x3333_0000_3333_0000
    bb = bb ^ (t ^ (t >> 14))
    t = (bb ^ (bb << 7)) & 0x5500_5500_5500_5500
    bb = bb ^ (t ^ (t >> 7))
    return bb

def flip_anti_diagonal(bb: Bitboard) -> Bitboard:
    # https://www.chessprogramming.org/Flipping_Mirroring_and_Rotating#FlipabouttheAntidiagonal
    t = bb ^ (bb << 36)
    bb = bb ^ ((t ^ (bb >> 36)) & 0xf0f0_f0f0_0f0f_0f0f)
    t = (bb ^ (bb << 18)) & 0xcccc_0000_cccc_0000
    bb = bb ^ (t ^ (t >> 18))
    t = (bb ^ (bb << 9)) & 0xaa00_aa00_aa00_aa00
    bb = bb ^ (t ^ (t >> 9))
    return bb


def shift_down(b: Bitboard) -> Bitboard:
    return b >> 8

def shift_2_down(b: Bitboard) -> Bitboard:
    return b >> 16

def shift_up(b: Bitboard) -> Bitboard:
    return (b << 8) & BB_ALL

def shift_2_up(b: Bitboard) -> Bitboard:
    return (b << 16) & BB_ALL

def shift_right(b: Bitboard) -> Bitboard:
    return (b << 1) & ~BB_FILE_A & BB_ALL

def shift_2_right(b: Bitboard) -> Bitboard:
    return (b << 2) & ~BB_FILE_A & ~BB_FILE_B & BB_ALL

def shift_left(b: Bitboard) -> Bitboard:
    return (b >> 1) & ~BB_FILE_H

def shift_2_left(b: Bitboard) -> Bitboard:
    return (b >> 2) & ~BB_FILE_G & ~BB_FILE_H

def shift_up_left(b: Bitboard) -> Bitboard:
    return (b << 7) & ~BB_FILE_H & BB_ALL

def shift_up_right(b: Bitboard) -> Bitboard:
    return (b << 9) & ~BB_FILE_A & BB_ALL

def shift_down_left(b: Bitboard) -> Bitboard:
    return (b >> 9) & ~BB_FILE_H

def shift_down_right(b: Bitboard) -> Bitboard:
    return (b >> 7) & ~BB_FILE_A


def _sliding_attacks(square: Square, occupied: Bitboard, deltas: Iterable[int]) -> Bitboard:
    attacks = BB_EMPTY

    for delta in deltas:
        sq = square

        while True:
            sq += delta
            if not (0 <= sq < 64) or square_distance(sq, sq - delta) > 2:
                break

            attacks |= BB_SQUARES[sq]

            if occupied & BB_SQUARES[sq]:
                break

    return attacks

def _step_attacks(square: Square, deltas: Iterable[int]) -> Bitboard:
    return _sliding_attacks(square, BB_ALL, deltas)

BB_KNIGHT_ATTACKS = [_step_attacks(sq, [17, 15, 10, 6, -17, -15, -10, -6]) for sq in SQUARES]
BB_KING_ATTACKS = [_step_attacks(sq, [9, 8, 7, 1, -9, -8, -7, -1]) for sq in SQUARES]
BB_PAWN_ATTACKS = [[_step_attacks(sq, deltas) for sq in SQUARES] for deltas in [[-7, -9], [7, 9]]]


def _edges(square: Square) -> Bitboard:
    return (((BB_RANK_1 | BB_RANK_8) & ~BB_RANKS[square_rank(square)]) |
            ((BB_FILE_A | BB_FILE_H) & ~BB_FILES[square_file(square)]))

def _carry_rippler(mask: Bitboard) -> Iterator[Bitboard]:
    # Carry-Rippler trick to iterate subsets of mask.
    subset = BB_EMPTY
    while True:
        yield subset
        subset = (subset - mask) & mask
        if not subset:
            break

def _attack_table(deltas: List[int]) -> Tuple[List[Bitboard], List[Dict[Bitboard, Bitboard]]]:
    mask_table = []
    attack_table = []

    for square in SQUARES:
        attacks = {}

        mask = _sliding_attacks(square, 0, deltas) & ~_edges(square)
        for subset in _carry_rippler(mask):
            attacks[subset] = _sliding_attacks(square, subset, deltas)

        attack_table.append(attacks)
        mask_table.append(mask)

    return mask_table, attack_table

BB_DIAG_MASKS, BB_DIAG_ATTACKS = _attack_table([-9, -7, 7, 9])
BB_FILE_MASKS, BB_FILE_ATTACKS = _attack_table([-8, 8])
BB_RANK_MASKS, BB_RANK_ATTACKS = _attack_table([-1, 1])