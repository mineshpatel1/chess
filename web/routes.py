import time
from flask import request

import log
from engine.board import Board
from engine import position
from web.app import app

cache = {
    'board': None,
}

def json_board(board):
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

    return {'board': by_rank}

@app.route('/')
@app.route('/index')
def index():
    return "Hello, World, Shirley"

@app.route('/time')
def get_current_time():
    return {'time': time.time()}

@app.route('/newBoard')
def new_board():
    board = Board()
    cache['board'] = board
    return json_board(board)


@app.route('/makeMove', methods=['POST'])
def make_move():
    data = request.get_json()
    board = cache['board']
    board.player_move(
        position.from_index(data['start_pos']),
        position.from_index(data['end_pos']),
    )
    return json_board(board)
