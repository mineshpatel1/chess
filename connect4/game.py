import random
from typing import Iterable, Optional

import log
from game.bitboard import lsb, bit_count
from game.exceptions import IllegalMove

RED = True
YELLOW = False

GRID_SIZE = 7 * 6

WINS = (
    [0, 1, 2, 3],
    [1, 2, 3, 4],
    [2, 3, 4, 5],
    [3, 4, 5, 6],
    [7, 8, 9, 10],
    [8, 9, 10, 11],
    [9, 10, 11, 12],
    [10, 11, 12, 13],
    [14, 15, 16, 17],
    [15, 16, 17, 18],
    [16, 17, 18, 19],
    [17, 18, 19, 20],
    [21, 22, 23, 24],
    [22, 23, 24, 25],
    [23, 24, 25, 26],
    [24, 25, 26, 27],
    [28, 29, 30, 31],
    [29, 30, 31, 32],
    [30, 31, 32, 33],
    [31, 32, 33, 34],
    [35, 36, 37, 38],
    [36, 37, 38, 39],
    [37, 38, 39, 40],
    [38, 39, 40, 41],

    [0, 7, 14, 21],
    [7, 14, 21, 28],
    [14, 21, 28, 35],
    [1, 8, 15, 22],
    [8, 15, 22, 29],
    [15, 22, 29, 36],
    [2, 9, 16, 23],
    [9, 16, 23, 30],
    [16, 23, 30, 37],
    [3, 10, 17, 24],
    [10, 17, 24, 31],
    [17, 24, 31, 38],
    [4, 11, 18, 25],
    [11, 18, 25, 32],
    [18, 25, 32, 39],
    [5, 12, 19, 26],
    [12, 19, 26, 33],
    [19, 26, 33, 40],
    [6, 13, 20, 27],
    [13, 20, 27, 34],
    [20, 27, 34, 41],

    [3, 9, 15, 21],
    [4, 10, 16, 22],
    [10, 16, 22, 28],
    [5, 11, 17, 23],
    [11, 17, 23, 29],
    [17, 23, 29, 35],
    [6, 12, 18, 24],
    [12, 18, 24, 30],
    [18, 24, 30, 36],
    [13, 19, 25, 31],
    [19, 25, 31, 37],
    [20, 26, 32, 38],

    [3, 11, 19, 27],
    [2, 10, 18, 26],
    [10, 18, 26, 34],
    [1, 9, 17, 25],
    [9, 17, 25, 33],
    [17, 25, 33, 41],
    [0, 8, 16, 24],
    [8, 16, 24, 32],
    [16, 24, 32, 40],
    [7, 15, 23, 31],
    [15, 23, 31, 39],
    [14, 22, 30, 38],
)

Slot = int


def mirror_square(slot: Slot) -> Slot:
    rank = int(slot / 7)
    return slot + (35 - (rank * 14))


SLOTS = [
    A1, B1, C1, D1, E1, F1, G1,
    A2, B2, C2, D2, E2, F2, G2,
    A3, B3, C3, D3, E3, F3, G3,
    A4, B4, C4, D4, E4, F4, G4,
    A5, B5, C5, D5, E5, F5, G5,
    A6, B6, C6, D6, E6, F6, G6,
] = [i for i in range(42)]

SLOTS_VFLIP = [mirror_square(slot) for slot in SLOTS]

Bitboard = int

BB_EMPTY = 0

BB_SLOTS = [
    BB_A1, BB_B1, BB_C1, BB_D1, BB_E1, BB_F1, BB_G1,
    BB_A2, BB_B2, BB_C2, BB_D2, BB_E2, BB_F2, BB_G2,
    BB_A3, BB_B3, BB_C3, BB_D3, BB_E3, BB_F3, BB_G3,
    BB_A4, BB_B4, BB_C4, BB_D4, BB_E4, BB_F4, BB_G4,
    BB_A5, BB_B5, BB_C5, BB_D5, BB_E5, BB_F5, BB_G5,
    BB_A6, BB_B6, BB_C6, BB_D6, BB_E6, BB_F6, BB_G6,
] = [1 << sq for sq in SLOTS]

BB_FILES = [
    (BB_A1 | BB_A2 | BB_A3 | BB_A4 | BB_A5 | BB_A6),
    (BB_B1 | BB_B2 | BB_B3 | BB_B4 | BB_B5 | BB_B6),
    (BB_C1 | BB_C2 | BB_C3 | BB_C4 | BB_C5 | BB_C6),
    (BB_D1 | BB_D2 | BB_D3 | BB_D4 | BB_D5 | BB_D6),
    (BB_E1 | BB_E2 | BB_E3 | BB_E4 | BB_E5 | BB_E6),
    (BB_F1 | BB_F2 | BB_F3 | BB_F4 | BB_F5 | BB_F6),
    (BB_G1 | BB_G2 | BB_G3 | BB_G4 | BB_G5 | BB_G6),
]


def _bb_win(_win):
    out = BB_EMPTY
    for w in _win:
        out |= BB_SLOTS[w]
    return out


BB_WINS = [_bb_win(w) for w in WINS]


def bitboard_to_slots(bb: Bitboard) -> Iterable[Slot]:
    """Returns slots populated in a given bitboard."""
    while bb:
        r = bb.bit_length() - 1
        yield Slot(r)
        bb ^= BB_SLOTS[r]


def binary_str(i: int) -> str:
    """Returns a base 2 integer as a binary string of length 64."""
    s = '{0:b}'.format(i)
    s = ('0' * (42 - len(s))) + s  # Pad with zeroes
    assert len(s) == 42
    return s


def bitboard_to_str(bb: Bitboard) -> str:
    """Prints a visual representation of the occupation represented by the input bitboard integer."""
    board_str = ''
    rank = 7
    for sq in SLOTS_VFLIP:
        sq_rank = int(sq / 6)
        if rank > sq_rank:
            board_str += f'\n{sq_rank + 1} '
            rank = sq_rank

        if binary_str(bb)[41 - sq] == '1':
            board_str += '[•]'
        else:
            board_str += '[ ]'
    board_str += '\n   A  B  C  D  E  F  G '
    return board_str


class Piece:
    def __init__(self, colour):
        self.colour = colour

    def __str__(self):
        return 'O' if self.colour else 'X'


class Connect4:
    def __init__(self):
        self.player = None
        self.state = None
        self.occupied = None
        self.occupied_colour = None

        self.reset()

    def reset(self):
        self.player = RED
        self.state = [0] * GRID_SIZE
        self.state = BB_EMPTY
        self.occupied = BB_EMPTY
        self.occupied_colour = {
            RED: BB_EMPTY,
            YELLOW: BB_EMPTY,
        }

    def make_move(self, move: int):
        if move not in list(self.legal_moves):
            raise IllegalMove

        if self.is_game_over:
            raise IllegalMove("Game is over.")

        bb_move = BB_SLOTS[move]
        self.occupied |= bb_move
        self.occupied_colour[self.player] |= bb_move
        self.player = not self.player

    def make_random_move(self):
        self.make_move(random.choice(list(self.legal_moves)))

    def piece_at(self, idx: int) -> Optional[Piece]:
        bb_slot = BB_SLOTS[idx]
        if not bool(self.occupied & bb_slot):
            return None
        elif self.occupied_colour[RED] & bb_slot:
            return Piece(RED)
        elif self.occupied_colour[YELLOW] & bb_slot:
            return Piece(YELLOW)

    @property
    def legal_moves(self):
        free_files = []
        for file in BB_FILES:
            if not file & self.occupied == file:
                free_files.append(file)

        for file in free_files:
            yield lsb(file & ~self.occupied)

    @property
    def is_game_over(self) -> bool:
        for win in BB_WINS:
            for colour in [RED, YELLOW]:
                if bit_count(win & self.occupied_colour[colour]) == 4:  # Connect 4!
                    return True

        # No more moves left
        if bit_count(self.occupied) == 42:
            return True
        return False

    @property
    def end_result(self) -> Optional[int]:
        """
        Returns: 1 if RED has won, -1 if YELLOW has won, 0 if a draw, None if not over.
        """
        for win in BB_WINS:
            for colour in [RED, YELLOW]:
                if bit_count(win & self.occupied_colour[colour]) == 4:  # Connect 4!
                    return 1 if colour == RED else -1

        # No more moves left
        if bit_count(self.occupied) == 42:
            return 0
        return None

    def player_has_won(self, colour):
        for win in BB_WINS:
            if bit_count(win & self.occupied_colour[colour]) == 4:  # Connect 4!
                return True
        return False

    def __str__(self):
        board_str = ''
        rank = 7
        for sq in SLOTS_VFLIP:
            sq_rank = int(sq / 6)
            if rank > sq_rank:
                board_str += f'\n{sq_rank + 1} '
                rank = sq_rank

            piece = self.piece_at(sq)
            if piece:
                board_str += f'[{piece}]'
            else:
                board_str += '[ ]'
        board_str += '\n   A  B  C  D  E  F  G '
        return board_str
