"""
Playing a game out between two move-choosing functions.

Useful for pitting evaluations or depths against each other, and for driving a game against an
outside engine. The players are just callables taking a state and returning a move, so an AI,
a random mover and a subprocess speaking UCI are all the same thing from here.
"""

import time
import inspect
from typing import Callable

import log
from games.base import GameState, Outcome


def _report(state: GameState, outcome: Outcome, print_summary: bool) -> Outcome:
    if print_summary:
        log.newline()
        log.info(state)
        log.info(str(outcome))
    return outcome


def simulate_game(
    state: GameState,
    first: Callable,
    second: Callable,
    print_moves: bool = False,
    print_summary: bool = True,
) -> Outcome:
    """
    Plays `state` out to a finish, `first` choosing for the player who moves first.

    The state is mutated rather than copied, so the caller keeps the finished game and can ask
    it for whatever record it keeps - a PGN, a move list - without this needing to know that
    such a thing exists.
    """
    log.info('Simulating game...')
    while not state.is_game_over:
        start_time = time.time()
        move = first(state) if state.turn else second(state)

        if print_moves:
            log.info(f'{move} ({time.time() - start_time:.3f}s)')
        state.make_move(move)

    return _report(state, state.result, print_summary)


async def simulate_game_async(
    state: GameState,
    first: Callable,
    second: Callable,
    print_moves: bool = True,
    print_summary: bool = True,
) -> Outcome:
    """As simulate_game, but either player may be a coroutine - an engine over a pipe, say."""
    while not state.is_game_over:
        start_time = time.time()
        chooser = first if state.turn else second

        move = chooser(state)
        if inspect.isawaitable(move):
            move = await move

        if print_moves:
            log.info(f'{move} ({time.time() - start_time:.3f}s)')
        state.make_move(move)

    return _report(state, state.result, print_summary)
