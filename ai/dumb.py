import random
from typing import Tuple

from engine.game import Game
from engine.position import Position

def first_possible_move(board: Game) -> Tuple[Position, Position]:
    """Ultra fast, ultra terrible and predictable."""
    for move in board.possible_moves(board.turn):
        return move

def random_move(board: Game) -> Tuple[Position, Position]:
    """Ultra terrible, but less predictable."""
    return random.choice(list(board.possible_moves(board.turn)))
