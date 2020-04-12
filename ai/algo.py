from typing import Tuple

from engine.game import Game
from engine.position import Position
from engine.constants import (
    PAWN,
    ROOK,
    KNIGHT,
    BISHOP,
    QUEEN,
    KING,
)

POINTS = {
    PAWN: 10,
    KNIGHT: 30,
    BISHOP: 30,
    ROOK: 50,
    QUEEN: 90,
    KING: 900,
}

def evaluate_piece(board: Game) -> Tuple[Position, Position]:
    possible_moves = list(board.possible_moves(board.turn))
    default_move = possible_moves[0]
    best_move = (None, -1)  # type: Tuple[Tuple[Position, Position], int]

    for start, target in possible_moves[1:]:
        piece = board.is_occupied(target)  # Can we take a piece?
        if piece and POINTS[piece.type] > best_move[1]:  # Get the highest value target
            best_move = ((start, target), POINTS[piece.type])

    if best_move[0]:
        return best_move[0]
    else:
        return default_move

