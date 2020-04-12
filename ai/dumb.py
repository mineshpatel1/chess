import random

def first_possible_move(board):
    """Ultra fast, ultra terrible and predictable."""
    for move in board.possible_moves(board.turn):
        return move

def random_move(board):
    """Ultra terrible, but less predictable."""
    return random.choice(list(board.possible_moves(board.turn)))
