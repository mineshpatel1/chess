import time
import queue
import threading
from typing import Union, Callable

import log
import chess
from game.board import Board


def batch(_func):
    """
    Decorator to wrap a function so that it can run in multiple threads.
    Takes a list of tuples with the inputs of the child function.
    """
    def batch_wrap(
        _lst, num_threads=25,
    ):
        def worker():
            while True:
                item = q.get()
                _func(**item)
                q.task_done()

        q = queue.Queue()

        for _i in range(num_threads):
            t = threading.Thread(target=worker)
            t.daemon = True
            t.start()

        for _item in _lst:
            q.put(_item)
        q.join()  # Wait for all operations to complete

    return batch_wrap


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


def simulate_game(
    white: Callable, black: Callable,
    print_moves: bool = False,
    print_summary: bool = True,
):
    b = Board()
    log.info('Simulating game...')
    while not b.is_game_over:
        if b.turn:
            move = white(b)  # White player
        else:
            move = black(b)  # Black player

        if print_moves:
            log.info(move)
        b.make_move(move)

    if print_summary:
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

    pgn = b.pgn_uci
    if print_summary:
        log.newline()
        log.info(pgn)
    return result, pgn


def batch_simulate(white: Callable, black: Callable, n=2, num_threads=5):
    @batch
    def _simulate(_results):
        result = simulate_game(white, black, False, False)
        _results.append(result)

    results = []
    jobs = [{'_results': results}] * n
    _simulate(jobs, num_threads)
    return results
