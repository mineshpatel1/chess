import log
from connect_4.game import Connect4, HIGH_BOUND, LOW_BOUND, IllegalMove


def _minimax(board: Connect4, depth: int, is_maximising_player: bool, player: bool):
    if depth == 0:
        return board.value if player else board.value * -1  # Opposing player has the current turn

    if board.is_game_over:
        if not player:
            return (-1 * board.value) + depth
        else:
            return board.value - depth

    if is_maximising_player:
        score = LOW_BOUND
        for move in board.legal_moves:
            board.make_move(move)
            a = _minimax(board, depth - 1, not is_maximising_player, player)
            score = max([a, score])
            board.unmake_move()
    else:
        score = HIGH_BOUND
        for move in board.legal_moves:
            board.make_move(move)
            a = _minimax(board, depth - 1, not is_maximising_player, player)
            score = min([a, score])
            board.unmake_move()
    return score


def minimax(board: Connect4, depth: int):
    score = LOW_BOUND
    best_move = None
    player = board.turn  # Current player is the AI player

    for move in board.legal_moves:
        board.make_move(move)
        value = _minimax(board, depth - 1, False, player)
        board.unmake_move()

        if value >= score:
            score = value
            best_move = move

    return best_move
