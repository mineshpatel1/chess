import time
from typing import Union, Callable

import log
import chess
from engine.board import Board


def _traverse_moves(board: Union[chess.Board, Board], depth: int, counter: int):
    if depth == 0:
        counter += 1
        return counter

    for move in board.legal_moves:
        board.push(move)
        counter = _traverse_moves(board, depth - 1, counter)
        board.pop()
    return counter


def traverse_moves(board: Union[chess.Board, Board], depth: int):
    counter = 0
    start_time = time.time()

    for move in board.legal_moves:
        board.push(move)
        counter = _traverse_moves(board, depth - 1, counter)
        board.pop()

    duration = time.time() - start_time
    log.info(f"Evaluations: {counter} in {duration} at {counter / duration} evaluations/s.")
    return counter


def simulate_game(white: Callable, black: Callable, print_moves: bool = True):
    b = Board()
    while not b.is_game_over:
        if b.turn:
            move = white(b)  # White player
        else:
            move = black(b)  # Black player

        if print_moves:
            log.info(move)
        b.make_move(move)

    log.newline()
    log.info(b)
    log.info(f'FEN: {b.fen}')
    if b.is_checkmate:
        winner = 'Black' if b.turn else 'White'
        log.info(f'{winner} is the winner!')
        result = 1 if winner == 'White' else -1
    else:
        log.info('Match drawn.')
        result = 0
    log.newline()
    log.info(b.pgn_uci)
    return result
