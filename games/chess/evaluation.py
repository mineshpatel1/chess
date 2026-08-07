"""
How good a chess position is, in the terms the search wants it.

Board.value and Board.weighted_value are absolute - positive for White, negative for Black -
because that is the natural way to read a position. The search is negamax, where every node
speaks for whoever is to move in it, so these flip the sign for Black.

Board is imported for type checking only: board.py names weighted_eval as its default
evaluation, and a runtime import here would close the circle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from games.chess.board import Board


def simple_eval(board: 'Board') -> int:
    """Material only. Cheap, and enough to play legally rather than well."""
    return board.value if board.turn else -board.value


def weighted_eval(board: 'Board') -> int:
    """Material plus where the pieces stand. More expensive, and much stronger."""
    return board.weighted_value if board.turn else -board.weighted_value
