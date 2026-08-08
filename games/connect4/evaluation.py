"""
How good a Connect 4 position is, in the terms the search wants it.

`value` is absolute - positive for Yellow, negative for Red - because that is the natural way to
read a position. The search is negamax, where every node speaks for whoever is to move in it, so
`weighted_eval` flips the sign for Red. Getting that flip wrong gives an engine that plays well
at even depths and badly at odd ones, which is a hard thing to notice and an easy thing to
avoid, so the flip lives in exactly one place.

The board is imported for type checking only: board.py names `weighted_eval` as its default
evaluation, so a runtime import back would close the circle. That is why the shift primitive the
threat terms need lives in bitboard.py, which neither module owns.

What is counted is **open threes, and nothing else**. A run of three with an empty cell on the
end is one move from winning; `runs` finds the runs and a shift finds the cell that would
complete them, which is the same primitive win detection uses one length up. Each such cell is
weighted twice over:

*By playability.* A completion cell sitting on the current landing square can be taken now. One
floating four rows above the stack cannot be taken until four other discs are played underneath
it, and counting the two alike overvalues castles in the air.

*By direction.* A vertical threat is answered by playing on top of it, which costs the blocker
nothing they were not going to spend anyway. A horizontal or diagonal threat has to be answered
in one particular column, and those are the threats that win games.

## What was tried and thrown away

Three further terms were built, measured and removed. Each was tested the way stage 7 says -
against the version before it - and each won that match:

    open twos on top of threes    0.527 +/- 0.018 over 600 games
    playability weighting          0.515 +/- 0.018 over 600 games
    direction weighting            0.580 +/- 0.029 over 200 games

and every one of them made the engine *worse*. Scored against a fixed opponent rather than
against its own predecessor, the ladder runs the other way:

    threes only (this)                        0.700 +/- 0.019
    threes + a centre-column bonus            0.663 +/- 0.020
    + open twos, playability, direction       0.388 +/- 0.025
    a centre-column bonus alone               0.425 +/- 0.013

all at depth 4 over 300 games against the same opponent: this same search with an evaluation
that returns zero. On a seed it was not tuned against, this one scores 0.649 +/- 0.018 over 400
games, and 0.705 +/- 0.029 at depth 5, so it is not an artefact of the depth it was tuned at.
Against `ai.search.random_move` it wins 200 games out of 200.

Two lessons, both of which cost several hundred games to learn.

**Compare against a fixed opponent, not against your last version.** A chain of pairwise wins
is not a chain of improvements. Every step above beat the step before it and the endpoint was
worse than the start.

**Returning zero is a real answer.** The opponent that beats most of these is not passive: with
nothing to choose between moves, the search takes the first one generated, and `legal_moves`
generates the centre column first, so an evaluation of zero *is* the policy "take the middle
unless there is a tactic". That is a strong Connect 4 heuristic. An evaluation with an opinion
about every quiet position overrides it everywhere, including in the many positions where it
has nothing useful to say - which is exactly why the centre-column bonus, the most obviously
correct term of the lot, lost outright. Saying nothing in a quiet position keeps the tie-break,
and keeping the tie-break is worth more than any of the terms above.

What is still not counted is parity, and it is the thing that matters most. Across 42 cells the
first player takes the odd squares and the second the even ones, so a threat on an odd row
favours Yellow and one on an even row favours Red, whoever built it. One well placed odd threat
beats four badly placed ones. Counting threats cannot see that, which is why this evaluation
plateaus and why depth buys more than terms past a point.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

from games.connect4.bitboard import Bitboard, bit_count, drops, runs
from games.connect4.constants import (
    CONNECT,
    DIRECTIONS,
    FULL_BOARD,
    RED,
    VERTICAL,
    YELLOW,
)

if TYPE_CHECKING:
    from games.connect4.board import Connect4

# Every score has to stay well inside ai.search.MATE, or a merely good position becomes
# indistinguishable from a forced win and the engine stops trying to actually finish games.
MAX_EVAL = 10_000

# Per cell that would complete a run of three into a win.
THREE = 5

# Per cell, by whether it can be played this move or is floating above the stack.
PLAYABLE = 3
DISTANT = 1

# Per cell, by direction. A vertical threat is blocked by playing on top of it, for free.
UPRIGHT = 1
ACROSS = 3

# The weights above multiplied out once at import, so the leaf does one lookup and two multiplies
# per direction rather than rebuilding the product every time it is asked.
#
#   (direction, value of a playable completion cell, value of a distant one)
#
THREAT_TERMS: Tuple[Tuple[int, int, int], ...] = tuple(
    (
        direction,
        THREE * (UPRIGHT if direction == VERTICAL else ACROSS) * PLAYABLE,
        THREE * (UPRIGHT if direction == VERTICAL else ACROSS) * DISTANT,
    )
    for direction in DIRECTIONS
)

# Runs one short of a win. Named rather than inlined because `threat_cells` is written for any
# length, and the choice to only ever ask it for this one is a result rather than a detail.
THREAT_LENGTH = CONNECT - 1


def threat_cells(position: Bitboard, direction: int, length: int, empty: Bitboard) -> Bitboard:
    """
    The empty cells that would extend a run of `length` in `position` by one.

    `runs` marks the lowest cell of every run, so the two cells that would extend it are one
    step below that mark and `length` steps above it. Intersecting with the empty cells does
    three jobs at once: it drops the ends that are already occupied, it drops the ones that
    walked off the board, and it drops the ones that landed on a sentinel - because `empty` is
    carved out of FULL_BOARD, which contains neither.
    """
    anchors = runs(position, direction, length)
    if not anchors:
        return 0
    return ((anchors << (length * direction)) | (anchors >> direction)) & empty


def threat_value(position: Bitboard, empty: Bitboard, playable: Bitboard) -> int:
    """One player's open threes, weighted by direction and by whether they are live."""
    total = 0
    for direction, playable_weight, distant_weight in THREAT_TERMS:
        cells = threat_cells(position, direction, THREAT_LENGTH, empty)
        if not cells:
            continue
        total += playable_weight * bit_count(cells & playable)
        total += distant_weight * bit_count(cells & ~playable)
    return total


def value(board: 'Connect4') -> int:
    """
    How much better Yellow's position is than Red's. Positive favours Yellow.

    Exactly zero in a position where neither side has an open three, which is most of them, and
    deliberately so - see the module docstring. Says nothing about a position that is already
    won either; that is the search's business, and `outcome` is checked before this is reached.
    """
    yellow, red = board.discs[YELLOW], board.discs[RED]
    occupied = yellow | red
    empty = FULL_BOARD & ~occupied
    playable = drops(occupied)

    return threat_value(yellow, empty, playable) - threat_value(red, empty, playable)


def weighted_eval(board: 'Connect4') -> int:
    """The evaluation the search uses, read from the point of view of the player to move."""
    return value(board) if board.turn else -value(board)
