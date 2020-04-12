from typing import Tuple

import log
from engine.game import Game
from engine.position import Position
from engine.constants import WHITE, BLACK

LOW_BOUND = -9999
COUNTER = 0

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


def _negamax(board: Game, depth: int) -> int:
    if depth == 0:
        board.counter += 1
        log.info(board.counter)
        return board.relative_value

    score = LOW_BOUND
    for move in board.possible_moves(board.turn):
        board.player_move(*move)
        value = -1 * _negamax(board, depth - 1)
        board.undo_move()
        if value > score:
            score = value
    return score


def negamax(board: Game, depth: int = 1) -> Tuple[Position, Position]:
    """
    Implementation of MiniMax algorithm using the negamax formulation. This is a search tree that searches all possible
    moves making optimal choices for each player in accordance to optimising the cost function (in this case board
    value). Then the original move that could lead the best score is chosen.

    https://www.chessprogramming.org/Minimax
    https://www.chessprogramming.org/Negamax
    """

    assert 0 < depth < 3, "Depth must be more than zero but less than 3 (this is very slow at the moment)."

    possible_moves = list(board.possible_moves(board.turn))  # Test with just the first one

    score = LOW_BOUND
    best_move = None

    for move in possible_moves:
        board.player_move(*move)
        value = _negamax(board, depth - 1)
        board.undo_move()

        if value > score:
            score = value
            best_move = move

    return best_move
