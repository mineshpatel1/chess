from typing import Tuple

import log
from engine.board import Board
from engine.constants import WHITE

LOW_BOUND = -9999


def _negamax(board: Board, depth: int, counter: int):
    if depth == 0:
        counter += 1
        return board.relative_value, counter

    score = LOW_BOUND
    for move in board.legal_moves:
        board.make_move(move)
        value, counter = _negamax(board, depth - 1, counter)
        value *= -1
        board.unmake_move()
        if value > score:
            score = value
    return score, counter


def negamax(board: Board, depth: int, print_count: bool = False):
    """
    Implementation of MiniMax algorithm using the negamax formulation. This is a search tree that searches all possible
    moves making optimal choices for each player in accordance to optimising the cost function (in this case board
    value). Then the original move that could lead the best score is chosen.

    https://www.chessprogramming.org/Minimax
    https://www.chessprogramming.org/Negamax
    """
    score = LOW_BOUND
    best_move = None
    counter = 0
    assert depth > 0

    for move in board.legal_moves:
        board.make_move(move)
        value, counter = _negamax(board, depth - 1, counter)
        value = value * -1
        board.unmake_move()

        if value > score:
            score = value
            best_move = move

    if print_count:
        log.info(f"Evaluations: {counter}")
    return best_move


