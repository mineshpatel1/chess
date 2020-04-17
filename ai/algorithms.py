import time
import random

import log
from engine.board import Board, Move

LOW_BOUND = -9999
HIGH_BOUND = 9999


def first_possible_move(board: Board) -> Move:
    """Ultra fast, ultra terrible and predictable."""
    for move in board.legal_moves:
        return move


def random_move(board: Board) -> Move:
    """Ultra terrible, but less predictable."""
    return random.choice(list(board.legal_moves))


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
    start_time = time.time()
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
        elapsed = time.time() - start_time
        log.info(
            f"Evaluations: {counter} in {elapsed}s at {counter/elapsed} evals/s."
        )
    return best_move


def _alpha_beta_min(board: Board, depth: int, alpha: int, beta: int, counter: int):
    if depth == 0:
        counter += 1
        return board.value, counter

    best = HIGH_BOUND
    for move in board.legal_moves:
        board.make_move(move)
        score, counter = _alpha_beta_max(board, depth - 1, alpha, beta, counter)
        board.unmake_move()

        best = min([score, best])
        beta = min([beta, best])
        if beta <= alpha:
            return score, counter
    return beta, counter


def _alpha_beta_max(board: Board, depth: int, alpha: int, beta: int, counter: int):
    if depth == 0:
        counter += 1
        return board.value, counter

    best = LOW_BOUND
    for move in board.legal_moves:
        board.make_move(move)
        score, counter = _alpha_beta_min(board, depth - 1, alpha, beta, counter)
        board.unmake_move()

        best = max([score, best])
        alpha = max([alpha, best])
        if beta <= alpha:
            return best, counter
    return alpha, counter


def alpha_beta(board: Board, depth: int, print_count: bool = False):
    """
    Implementation of Alpha-Beta pruning to optimise the MiniMax algorithm. This stops evaluating a move when at least
    one possibility has been found that proves the move to be worse than a previously examined move.

    https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning
    """

    alpha = LOW_BOUND
    beta = HIGH_BOUND
    score = LOW_BOUND
    best_move = None
    counter = 0
    start_time = time.time()
    assert depth > 0

    for move in board.legal_moves:
        board.make_move(move)
        value, counter = _alpha_beta_max(board, depth - 1, alpha, beta, counter)
        value *= -1
        board.unmake_move()

        if value > score:
            score = value
            best_move = move
    if print_count:
        elapsed = time.time() - start_time
        log.info(
            f"Evaluations: {counter} in {elapsed}s at {counter / elapsed} evals/s."
        )
    return best_move
