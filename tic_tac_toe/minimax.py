import log
from tic_tac_toe.game import Game, HIGH_BOUND, LOW_BOUND


def _alpha_beta(board: Game, depth: int, alpha: int, beta: int, is_maximising_player: bool):
    if depth == 0:
        return board.value  # Opposing player has the current turn

    if board.is_game_over:
        val = board.value
        if val > 0:
            return val + depth
        else:
            return val - depth

    if is_maximising_player:
        score = LOW_BOUND
        for move in board.legal_moves:
            board.make_move(move)
            score = max(_alpha_beta(board, depth - 1, alpha, beta, not is_maximising_player), score)
            alpha = max(alpha, score)
            board.unmake_move()
            if alpha >= beta:
                return score
    else:
        score = HIGH_BOUND
        for move in board.legal_moves:
            board.make_move(move)
            score = min(_alpha_beta(board, depth - 1, alpha, beta, not is_maximising_player), score)
            beta = min(beta, score)
            board.unmake_move()
            if alpha >= beta:
                return score
    return score


def alpha_beta(board: Game, depth: int):
    score = LOW_BOUND * 2
    best_move = None
    player = board.turn  # Current player is the AI player

    for move in board.legal_moves:
        board.make_move(move)
        value = _alpha_beta(board, depth - 1, LOW_BOUND * 2, HIGH_BOUND * 2, not player)
        board.unmake_move()

        if not player:
            value *= -1

        if value >= score:
            score = value
            best_move = move
    return best_move
