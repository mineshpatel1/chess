"""
Move enumeration, for checking that a game generates exactly the moves it should.

Counting leaves at a fixed depth is a blunt instrument that catches almost everything: a rule
implemented too loosely shows up as extra nodes, one implemented too tightly as missing ones,
and both compound with depth. Chess has published reference counts to check against
(https://www.chessprogramming.org/Perft_Results), which is what tests/test_permutations.py
uses; a new game has no such table, but the counts are still worth pinning once they are
believed, so that they cannot drift unnoticed.

A finished game is counted as having no continuations, which for chess is what generating no
moves already achieves and for a game won while moves remain has to be said out loud. See
`_count`.
"""

import time
from typing import Optional

import log
from games.base import GameState


def _count(state: GameState, depth: int) -> int:
    """
    Leaves below this position, counting a finished game as having none.

    The decided check comes *after* the horizon, and the order is the whole of its correctness.
    A game that finishes at exactly `depth` plies is a leaf and counts as one, which is what
    chess already does for free: a checkmate at the horizon returns 1 here because the loop
    never runs. Checking first would make a Connect 4 win at the horizon count as zero and put
    the two games on different definitions.

    Below the horizon a finished game has no continuations, again matching chess, where a mated
    position generates no moves and contributes nothing. Chess reaches that by generating zero
    moves; Connect 4 cannot, because it is won with six columns still playable, so it needs to
    be told. Without this, perft counts discs dropped after somebody has already won.

    Chess never overrides `outcome`, so for chess this is one attribute load per node and the
    five pinned reference counts are unchanged.
    """
    if depth == 0:
        return 1

    if state.outcome is not None:
        return 0

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

    if state.outcome is not None:  # Decided, so there is nothing to divide
        return {}

    counts = {}
    for move in state.legal_moves:
        state.make_move(move)
        counts[str(move)] = _count(state, depth - 1)
        state.unmake_move()
    return counts
