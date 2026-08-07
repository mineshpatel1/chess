"""
How good a Connect 4 position is, in the terms the search wants it.

`Connect4.value` is absolute - positive for Yellow, negative for Red - because that is the
natural way to read a position. The search is negamax, where every node speaks for whoever is
to move in it, so the functions here flip the sign for Red. Getting that flip wrong is the
classic way to end up with an engine that plays well at even depths and badly at odd ones, and
it is the reason the flip lives in one place rather than in the board.

Connect4 is imported for type checking only: board.py names an evaluation here as its default,
and a runtime import back would close the circle. The shift primitives the threat terms need
therefore come from bitboard.py, which neither of them owns.

Terms are added one at a time and each is kept only if it wins a match against the version
before it - see ai/match.py. The weights below record those results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from games.connect4.board import Connect4

# Every score must stay far inside ai.search.MATE, or a merely good position becomes
# indistinguishable from a forced win and the search stops trying to actually win.
MAX_EVAL = 10_000


def centre_eval(board: 'Connect4') -> int:
    """Placeholder. Stage 7 replaces this with the real terms."""
    return 0


def weighted_eval(board: 'Connect4') -> int:
    """The evaluation the search uses, read from the point of view of the player to move."""
    return centre_eval(board) if board.turn else -centre_eval(board)
