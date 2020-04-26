import random
from typing import List, Iterable

import log
from game.bitboard import lsb, bit_count
from game.exceptions import IllegalMove

CROSSES = True
NOUGHTS = False

LOW_BOUND = -9999999
HIGH_BOUND = 9999999

GRID_SIZE = 9

WINS = (
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],

    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],

    [0, 4, 8],
    [1, 4, 7],
)

Square = int

SQUARES = [
    A, B, C,
    D, E, F,
    G, H, I,
] = [i for i in range(9)]

Bitboard = int
BB_EMPTY = 0
BB_ALL = 511

BB_SQUARES = [
    BB_A, BB_B, BB_C,
    BB_D, BB_E, BB_F,
    BB_G, BB_H, BB_I,
] = [1 << sq for sq in SQUARES]

def _bb_win(_win):
    out = BB_EMPTY
    for w in _win:
        out |= BB_SQUARES[w]
    return out

BB_WINS = [_bb_win(w) for w in WINS]


def bitboard_to_squares(bb: Bitboard) -> Iterable[Square]:
    """Returns squares populated in a given bitboard."""
    while bb:
        r = bb.bit_length() - 1
        yield Square(r)
        bb ^= BB_SQUARES[r]


def binary_str(i: int) -> str:
    """Returns a base 2 integer as a binary string of length 64."""
    s = '{0:b}'.format(i)
    s = ('0' * (9 - len(s))) + s  # Pad with zeroes
    assert len(s) == 9
    return s


class Game:
    def __init__(self, state: List[int] = None):
        self.state = None
        self.occupied = None
        self.turn = None
        self.occupied_player = None
        self.reset()

        if state:
            raise NotImplementedError

    def reset(self):
        self.state = BB_EMPTY
        self.occupied = BB_EMPTY
        self.turn = CROSSES
        self.occupied_player = {
            CROSSES: BB_EMPTY,
            NOUGHTS: BB_EMPTY,
        }

    def mark_at(self, square: int):
        bb_slot = BB_SQUARES[square]
        if not bool(self.occupied & bb_slot):
            return None
        elif self.occupied_player[CROSSES] & bb_slot:
            return 1
        elif self.occupied_player[NOUGHTS] & bb_slot:
            return -1

    def make_move(self, move: int):
        if move not in list(self.legal_moves):
            raise IllegalMove

        bb_sq = BB_SQUARES[move]
        self.occupied_player[self.turn] |= bb_sq
        self.occupied |= bb_sq
        self.turn = not self.turn

    def make_random_move(self) -> int:
        random_move = random.choice(list(self.legal_moves))
        self.make_move(random_move)
        return random_move

    @property
    def is_game_over(self):
        return self.end_result is not None

    @property
    def end_result(self):
        for win in BB_WINS:
            for player in [CROSSES, NOUGHTS]:
                if bit_count(win & self.occupied_player[player]) == 3:
                    return 1 if player == CROSSES else -1

        if bit_count(self.occupied) == 9:
            return 0

        return None

    @property
    def legal_moves(self) -> Iterable[Square]:
        for move in bitboard_to_squares(BB_ALL & ~self.occupied):
            yield move

    def __str__(self) -> str:
        board_str = '\n|'
        rank = 0
        for sq in SQUARES:
            sq_rank = int(sq / 3)
            if sq_rank > rank:
                board_str += '\n-------------\n|'
                rank += 1

            mark = self.mark_at(sq)
            if mark is not None:
                board_str += ' X ' if mark == 1 else ' O '
            else:
                board_str += '   '
            board_str += '|'
        return board_str
