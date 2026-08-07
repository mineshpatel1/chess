"""
How good a chess position is, in the terms the search wants it.

ChessBoard.value and ChessBoard.weighted_value are absolute - positive for White, negative for
Black - because that is the natural way to read a position. The search is negamax, where every
node speaks for whoever is to move in it, so these flip the sign for Black.

ChessBoard is imported for type checking only: board.py names weighted_eval as its default
evaluation, and a runtime import here would close the circle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from games.chess.board import ChessBoard


def simple_eval(board: 'ChessBoard') -> int:
    """Material only. Cheap, and enough to play legally rather than well."""
    return board.value if board.turn else -board.value


def weighted_eval(board: 'ChessBoard') -> int:
    """Material plus where the pieces stand. More expensive, and much stronger."""
    return board.weighted_value if board.turn else -board.weighted_value
