import time
from typing import Union

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
