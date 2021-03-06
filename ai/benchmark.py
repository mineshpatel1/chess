import time
import inspect
from typing import Tuple, Union, Callable

import log
import chess
from game.board import Board


def _traverse_moves(board: Union[chess.Board, Board], depth: int, counter: int):
    if depth == 0:
        counter += 1
        return counter

    for move in board.legal_moves:
        board.push(move)
        counter = _traverse_moves(board, depth - 1, counter)
        board.pop()
    return counter


def traverse_moves(board: Union[chess.Board, Board], depth: int, print_summary: bool = True):
    counter = 0
    start_time = time.time()

    for move in board.legal_moves:
        board.push(move)
        counter = _traverse_moves(board, depth - 1, counter)
        board.pop()

    duration = time.time() - start_time
    if print_summary:
        log.info(f"Evaluations: {counter} in {duration} at {counter / duration} evaluations/s.")
    return counter


def _process_results(board: Board, print_summary: bool = True) -> Tuple[int, str]:
    if print_summary:
        log.newline()
        log.info(board)
        log.info(f'FEN: {board.fen}')
    if board.is_checkmate:
        winner = 'Black' if board.turn else 'White'
        log.info(f'{winner} is the winner!')
        result = 1 if winner == 'White' else -1
    else:
        log.info('Match drawn.')
        result = 0

    pgn = board.pgn_uci
    if print_summary:
        log.newline()
        log.info(pgn)
    return result, pgn


def simulate_game(
    white: Callable, black: Callable,
    print_moves: bool = False,
    print_summary: bool = True,
):
    """Simulates a game between two AIs by specifying a best move function for White and one for Black."""
    b = Board()
    log.info('Simulating game...')
    while not b.is_game_over:
        start_time = time.time()
        if b.turn:
            move = white(b)  # White player
        else:
            move = black(b)  # Black player

        if print_moves:
            log.info(f'{b.fullmoves}. {move} ({time.time() - start_time}s)')
        b.make_move(move)
    return _process_results(b, print_summary)


async def simulate_game_async(white, black, print_moves=True, print_summary=True):
    """
    Same as simulate_game, but allows white or black to be specified as async functions. Useful when using engines.
    """
    b = Board()
    while not b.is_game_over:
        start_time = time.time()
        if b.turn:
            if inspect.iscoroutinefunction(white):
                move = await white(b)  # White player
            else:
                move = white(b)
        else:
            if inspect.iscoroutinefunction(black):
                move = await black(b)
            else:
                move = black(b)  # Black player

        if print_moves:
            log.info(f'{b.fullmoves}. {move} ({time.time() - start_time}s)')
        b.make_move(move)

    return _process_results(b, print_summary)

