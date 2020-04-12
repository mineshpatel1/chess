import random
from flask import request

import log
from engine.constants import WHITE, BLACK
from engine.game import Game
from engine.exceptions import IllegalMove, Checkmate, Draw
from engine import position
from web.server import app

cache = {
    'board': None,
}

def first_possible_move(board):
    """Ultra fast, ultra terrible and predictable."""
    for move in board.possible_moves(board.turn):
        return move

def random_move(board):
    """Ultra terrible, but less predictable."""
    return random.choice(list(board.possible_moves(board.turn)))

def json_board(board, params=None):
    _by_rank = {}
    for square in board.squares:
        rank = square.pos.rank
        _by_rank[rank] = _by_rank.get(rank, [])

        _square = {
            'rank': square.pos.rank,
            'file': square.pos.file,
            'index': square.pos.index,
        }

        if square.piece:
            _square['piece'] = square.piece.type
            _square['piece_colour'] = square.piece.colour

        _by_rank[rank].append(_square)

    by_rank = {}
    for i, rank in enumerate(sorted(_by_rank.keys(), reverse=True)):
        by_rank[i] = _by_rank[rank]

    payload = {'board': by_rank, 'turn': board.turn, 'fen': board.fen}
    if params:
        payload.update(params)
    return payload

@app.route('/newGame')
def new_game():
    board = Game()
    cache['board'] = board
    return json_board(board)

@app.route('/loadGame', methods=['POST'])
def load_game():
    data = request.get_json()
    board = Game(state=data['state'])
    cache['board'] = board
    return json_board(board)

@app.route('/makeMove', methods=['POST'])
def make_move():
    """Both white and black players are human."""
    data = request.get_json()
    board = cache['board']

    try:
        board.player_move(
            position.from_index(data['start_pos']),
            position.from_index(data['end_pos']),
        )
        board.raise_if_game_over()
        return json_board(board)
    except IllegalMove as err:
        return {'error': str(err)}
    except Checkmate:
        winner = BLACK if board.turn == WHITE else WHITE
        return json_board(board, {'end': f"Checkmate: {winner} wins!"})
    except Draw as err:
        return json_board(board, {'end': str(err)})


@app.route('/makeMoveAi')
def make_move_ai():
    """Randomly choose a possible move."""
    board = cache['board']
    try:
        board.raise_if_game_over()
        # from_pos, to_pos = first_possible_move(board)
        from_pos, to_pos = random_move(board)
        board.player_move(from_pos, to_pos)
        return json_board(board)
    except IllegalMove as err:
        return {'error': str(err)}
    except Checkmate:
        winner = BLACK if board.turn == WHITE else WHITE
        return json_board(board, {'end': f"Checkmate: {winner} wins!"})
    except Draw as err:
        return json_board(board, {'end': err.MESSAGE})
