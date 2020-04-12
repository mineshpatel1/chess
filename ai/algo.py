from typing import Tuple

from engine.game import Game
from engine.position import Position

def evaluate_piece(board: Game) -> Tuple[Position, Position]:
    """Really basic algorithm that just looks at the possible moves on show and takes the best piece."""
    possible_moves = list(board.possible_moves(board.turn))
    score = -1
    best_move = possible_moves[0]  # type: Tuple[Position, Position]

    for start, target in possible_moves[1:]:
        piece = board.is_occupied(target)  # Can we take a piece?
        if piece and piece.value > score:  # Get the highest value target
            best_move = (start, target)
            score = piece.value
    return best_move

def brute_force_minimax(board: Game, depth: int) -> Tuple[Position, Position]:
    possible_moves = list(board.possible_moves(board.turn))
