from __future__ import annotations

import random
import numpy as np
from typing import List, Iterable

import log
from game.bitboard import bit_count
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
    [2, 4, 6],
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
    def __init__(self, from_array: List[int] = None, mhn: str = None):
        self.state = None
        self.occupied = None
        self.turn = None
        self.occupied_player = None
        self.move_history = None
        self._history = None
        self.reset()

        if from_array:
            self.set_from_array(from_array)

        if mhn:
            self.set_from_mhn(mhn)

    def _save(self):
        self._history.append(GameState(self))

    def reset(self):
        self.state = BB_EMPTY
        self.occupied = BB_EMPTY
        self.turn = CROSSES
        self.move_history = []
        self._history = []
        self.occupied_player = {
            CROSSES: BB_EMPTY,
            NOUGHTS: BB_EMPTY,
        }

    def set_from_array(self, array: List[int]):
        assert len(array) == 9
        for i, sq in enumerate(array):
            if sq == 1:
                self.occupied_player[CROSSES] |= BB_SQUARES[i]
            elif sq == -1:
                self.occupied_player[NOUGHTS] |= BB_SQUARES[i]
        self.occupied = self.occupied_player[CROSSES] | self.occupied_player[NOUGHTS]

    def set_from_mhn(self, mhn: str):
        for move in mhn:
            self.make_move(int(move))

    def mark_at(self, square: int):
        bb_slot = BB_SQUARES[square]
        if not bool(self.occupied & bb_slot):
            return None
        elif self.occupied_player[CROSSES] & bb_slot:
            return 1
        elif self.occupied_player[NOUGHTS] & bb_slot:
            return -1

    def make_move(self, move: int):
        self._save()
        if move not in list(self.legal_moves):
            raise IllegalMove

        bb_sq = BB_SQUARES[move]
        self.occupied_player[self.turn] |= bb_sq
        self.occupied |= bb_sq
        self.turn = not self.turn
        self.move_history.append(move)

    def unmake_move(self):
        self.move_history = self.move_history[:-1]
        state = self._history.pop()
        state.load(self)

    def make_random_move(self) -> int:
        random_move = random.choice(list(self.legal_moves))
        self.make_move(random_move)
        return random_move

    def copy(self) -> Game:
        _game = Game(from_array=self.array)
        _game.turn = self.turn
        return _game

    @property
    def is_game_over(self):
        return self.end_result is not None

    @property
    def end_result(self):
        for win in BB_WINS:
            for player in [CROSSES, NOUGHTS]:
                if bit_count(win & self.occupied_player[player]) == 3:
                    return 1 if player else -1

        if bit_count(self.occupied) == 9:
            return 0

        return None

    @property
    def legal_moves(self) -> Iterable[Square]:
        if not self.is_game_over:
            for move in bitboard_to_squares(BB_ALL & ~self.occupied):
                yield move

    @property
    def array(self) -> List[int]:
        out = []
        for sq in SQUARES:
            mark = self.mark_at(sq)
            if mark is None:
                out.append(0)
            else:
                out.append(mark)
        return out

    @property
    def value(self) -> int:
        """Evaluation based on number of possible victories left remaining."""
        result = self.end_result
        if result is not None:
            if result == 0:
                return 0
            elif result == 1:
                return HIGH_BOUND
            elif result == -1:
                return LOW_BOUND
            else:
                raise Exception

        noughts_possible = 0
        crosses_possible = 0
        for win in BB_WINS:
            if not bool(win & self.occupied_player[NOUGHTS]):
                noughts_possible += 1

            if not bool(win & self.occupied_player[CROSSES]):
                crosses_possible += 1
        return noughts_possible - crosses_possible

    @property
    def id(self) -> int:
        return hash((self.occupied_player[CROSSES], self.occupied_player[NOUGHTS]))

    @property
    def mhn(self) -> str:
        out = ''
        for move in self.move_history:
            out += str(move)
        return out

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

    @property
    def model_input(self) -> np.ndarray:
        """
        Returns board representation as a NumPy array that can be recognised as input for model training.
        Is relative to the player who has to play next as opposed to noughts and crosses objectively.
        """
        x = []
        our_pieces = list(bitboard_to_squares(self.occupied_player[self.turn]))
        for i in range(9):
            if i in our_pieces:
                x.append(1)
            else:
                x.append(0)

        y = []
        enemy_pieces = list(bitboard_to_squares(self.occupied_player[not self.turn]))
        for i in range(9):
            if i in enemy_pieces:
                y.append(-1)
            else:
                y.append(0)

        position = np.append(x, y)
        return np.reshape(position, (2, 3, 3))


class GameState:
    def __init__(self, game: Game):
        self.occupied_crosses = game.occupied_player[CROSSES]
        self.occupied_noughts = game.occupied_player[NOUGHTS]
        self.turn = game.turn

    def load(self, game: Game):
        game.occupied_player[CROSSES] = self.occupied_crosses
        game.occupied_player[NOUGHTS] = self.occupied_noughts
        game.occupied = self.occupied_crosses | self.occupied_noughts
        game.turn = self.turn
