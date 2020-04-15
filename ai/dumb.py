import random

from engine.board import Board, Move


def first_possible_move(board: Board) -> Move:
    """Ultra fast, ultra terrible and predictable."""
    for move in board.legal_moves:
        return move


def random_move(board: Board) -> Move:
    """Ultra terrible, but less predictable."""
    return random.choice(list(board.legal_moves))
