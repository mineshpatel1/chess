"""
Move enumeration, for checking that a game generates exactly the moves it should.

Counting leaves at a fixed depth is a blunt instrument that catches almost everything: a rule
implemented too loosely shows up as extra nodes, one implemented too tightly as missing ones,
and both compound with depth. Chess has published reference counts to check against
(https://www.chessprogramming.org/Perft_Results), which is what tests/test_permutations.py
uses; a new game has no such table, but the counts are still worth pinning once they are
believed, so that they cannot drift unnoticed.
"""

import time
from typing import Optional

import log
from games.base import GameState


def _count(state: GameState, depth: int) -> int:
    if depth == 0:
        return 1

    total = 0
    for move in state.legal_moves:
        state.make_move(move)
        total += _count(state, depth - 1)
        state.unmake_move()
    return total


def traverse_moves(state: GameState, depth: int, print_summary: bool = True) -> int:
    """The number of distinct move sequences of exactly `depth` plies from this position."""
    start_time = time.perf_counter()
    total = _count(state, depth)
    duration = time.perf_counter() - start_time

    if print_summary:
        rate = total / duration if duration else float('inf')
        log.info(f'{total:,} nodes in {duration:.3f}s at {rate:,.0f} nodes/s.')
    return total


def divide(state: GameState, depth: int) -> Optional[dict]:
    """
    Node counts broken down by first move.

    When a total disagrees with a reference, this is how the disagreement gets localised:
    compare per-move counts against another implementation and the wrong move names itself,
    rather than having to bisect the position by hand.
    """
    if depth < 1:
        return None

    counts = {}
    for move in state.legal_moves:
        state.make_move(move)
        counts[str(move)] = _count(state, depth - 1)
        state.unmake_move()
    return counts
