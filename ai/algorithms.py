import time
import random
import multiprocessing
from typing import Callable

import log
from game.board import Board, Move

LOW_BOUND = -9999999
HIGH_BOUND = 9999999


def simple_eval(board: Board) -> int:
    return board.value


def weighted_eval(board: Board) -> int:
    return board.weighted_value


def first_possible_move(board: Board) -> Move:
    """Ultra fast, ultra terrible and predictable."""
    for move in board.legal_moves:
        return move


def random_move(board: Board) -> Move:
    """Ultra terrible, but less predictable."""
    return random.choice(list(board.legal_moves))


def _minimax(board: Board, depth: int, is_maximising_player: bool, player: bool):
    if depth == 0:
        return board.value if player else board.value * -1  # Opposing player has the current turn

    if is_maximising_player:
        score = LOW_BOUND
        for move in board.legal_moves:
            board.make_move(move)
            score = max([_minimax(board, depth - 1, not is_maximising_player, player), score])
            board.unmake_move()
    else:
        score = HIGH_BOUND
        for move in board.legal_moves:
            board.make_move(move)
            score = min([_minimax(board, depth - 1, not is_maximising_player, player), score])
            board.unmake_move()
    return score


def minimax(board: Board, depth: int):
    """
    Implementation of MiniMax algorithm using the standard formulation. This should play identically to negamax,
    but is a bit easier to understand how the algorithm works.

    https://www.chessprogramming.org/Minimax
    """
    score = LOW_BOUND
    best_move = None
    player = board.turn  # Current player is the AI player

    for move in board.legal_moves:
        board.make_move(move)
        value = _minimax(board, depth - 1, False, player)
        board.unmake_move()

        if value > score:
            score = value
            best_move = move
    return best_move


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
    moves making optimal choices for each player in accordance to optimising the cost function (in this case game
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


def _alpha_beta_min(board: Board, depth: int, alpha: int, beta: int, player: bool, board_eval: Callable, counter: int):
    if depth == 0:
        counter += 1
        value = board_eval(board) if not player else board_eval(board) * -1
        return value, counter

    best = HIGH_BOUND
    i = 0
    for move in board.legal_moves:
        board.make_move(move)
        score, counter = _alpha_beta_max(board, depth - 1, alpha, beta, player, board_eval, counter)
        board.unmake_move()

        best = min([score, best])
        beta = min([beta, best])
        if beta <= alpha:
            return best, counter
        i += 1

    if i == 0:  # Game is over
        if board.is_checkmate and board.turn != player:
            return HIGH_BOUND + 1, counter  # Value checkmate above all else
        else:
            return LOW_BOUND - 1, counter   # Any other end game state is the worst case scenario
    return beta, counter


def _alpha_beta_max(board: Board, depth: int, alpha: int, beta: int, player: bool, board_eval: Callable, counter: int):
    if depth == 0:
        counter += 1
        value = board_eval(board) if not player else board_eval(board) * -1
        return value, counter

    best = LOW_BOUND
    i = 0
    for move in board.legal_moves:
        board.make_move(move)
        score, counter = _alpha_beta_min(board, depth - 1, alpha, beta, player, board_eval, counter)
        board.unmake_move()

        best = max([score, best])
        alpha = max([alpha, best])
        if beta <= alpha:
            return best, counter
        i += 1

    if i == 0:  # Game is over
        if board.is_checkmate and board.turn != player:
            return LOW_BOUND - 1, counter  # Value checkmate above all else
        else:
            return HIGH_BOUND + 1, counter   # Any other end game state is the worst case scenario

    return alpha, counter


def _get_best_move(_board: Board, _depth: int, _base_move: Move, _board_eval: Callable):
    """Helper function for running alpha_beta on multiple threads."""

    alpha = LOW_BOUND
    beta = HIGH_BOUND
    _counter = 0
    _player = _board.turn

    _board.make_move(_base_move)

    _value, _counter = _alpha_beta_max(_board, _depth - 1, alpha, beta, _player, _board_eval, _counter)
    _value *= -1

    _board.unmake_move()

    return _base_move, _value, _counter


def alpha_beta(board: Board, depth: int = 3, board_eval: Callable = weighted_eval, print_count: bool = False):
    """
    Implementation of Alpha-Beta pruning to optimise the MiniMax algorithm. This stops evaluating a move when at least
    one possibility has been found that proves the move to be worse than a previously examined move. Should play
    identically to negamax for the same search depth.

    https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning
    """

    best_move = None
    start_time = time.time()
    assert depth > 0

    legal_moves = list(board.legal_moves)

    jobs = []
    for move in legal_moves:
        new_board = Board(board.fen)
        jobs.append((new_board, depth, move, board_eval))

    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    result = pool.starmap(_get_best_move, jobs)

    counter = 0
    score = LOW_BOUND
    for move, value, sub_counter in result:
        counter += sub_counter
        if value > score:
            score = value
            best_move = move

    if print_count:
        elapsed = time.time() - start_time
        log.info(
            f"Evaluations: {counter} in {elapsed}s at {counter / elapsed} evals/s."
        )
    return best_move
