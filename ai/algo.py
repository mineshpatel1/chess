from typing import Tuple

import log
from engine.game import Game
from engine.position import Position
from engine.constants import WHITE, BLACK, PIECE_VALUES

LOW_BOUND = -9999


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


def _negamax(board: Game, depth: int, counter: int) -> Tuple[int, int]:
    if depth == 0:
        counter += 1
        return board.relative_value, counter

    score = LOW_BOUND
    for move in board.possible_moves(board.turn):
        board.player_move(*move)
        value, counter = _negamax(board, depth - 1, counter)
        value *= -1
        board.undo_move()
        if value > score:
            score = value
    return score, counter


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
    counter = 0

    for move in possible_moves:
        board.player_move(*move)
        value, counter = _negamax(board, depth - 1, counter)
        board.undo_move()

        if value > score:
            score = value
            best_move = move

    log.info(f"Evaluations: {counter}")
    return best_move


def evaluation_superfast(board):
    total = 0
    for square, piece in board.piece_map().items():
        if piece.color:
            total += PIECE_VALUES[piece.symbol().lower()]
        else:
            total -= PIECE_VALUES[piece.symbol().lower()]
    modifier = 1 if board.turn else -1
    return total * modifier


def _negamax_superfast(board, depth, counter):
    if depth == 0:
        counter += 1
        return evaluation_superfast(board), counter

    score = LOW_BOUND
    for move in board.legal_moves:
        board.push_uci(move.uci())
        value, counter = _negamax_superfast(board, depth - 1, counter)
        value *= -1
        board.pop()
        if value > score:
            score = value
    return score, counter


def negamax_superfast(board, depth):
    score = LOW_BOUND
    best_move = None
    counter = 0

    for move in board.legal_moves:
        board.push_uci(move.uci())
        value, counter = _negamax_superfast(board, depth - 1, counter)
        board.pop()

        if value > score:
            score = value
            best_move = move

    log.info(f"Evaluations: {counter}")
    return best_move


def _negamax_bitboard(board, depth, counter):
    if depth == 0:
        counter += 1
        return board.relative_value, counter

    score = LOW_BOUND
    for move in board.legal_moves:
        board.make_move(move)
        value, counter = _negamax_bitboard(board, depth - 1, counter)
        value *= -1
        board.unmake_move()
        if value > score:
            score = value
    return score, counter


def negamax_bitboard(board, depth):
    score = LOW_BOUND
    best_move = None
    counter = 0

    for move in board.legal_moves:
        board.make_move(move)
        value, counter = _negamax_bitboard(board, depth - 1, counter)
        board.unmake_move()

        if value > score:
            score = value
            best_move = move

    log.info(f"Evaluations: {counter}")
    return best_move
