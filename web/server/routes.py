from typing import Optional, Dict, Any
from flask import request

import log
from ai import dumb
from engine.constants import WHITE, BLACK
from engine.board import Board, Move, SQUARES, SQUARES_VFLIP
from engine.exceptions import IllegalMove, Checkmate, Draw
from web.server import app

cache = {
    'board': Board(),
}


def json_board(board: Board, params: Optional[Dict[str, Any]] = None):
    by_rank = {}
    for sq in SQUARES_VFLIP:
        rank = sq.rank
        by_rank[rank] = by_rank.get(rank, [])

        _square = {
            'rank': sq.rank,
            'file': sq.file,
            'index': int(sq),
        }

        piece = board.piece_at(sq)
        if piece:
            _square['piece'] = piece.type
            _square['piece_colour'] = 'white' if piece.colour else 'black'

        by_rank[rank].append(_square)

    by_rank_reverse = {}
    for i, rank in enumerate(sorted(by_rank.keys(), reverse=True)):
        by_rank_reverse[i] = by_rank[rank]

    payload = {'board': by_rank_reverse, 'turn': board.turn_name, 'fen': board.fen}
    if params:
        payload.update(params)
    return payload


@app.route('/newGame')
def new_game():
    board = Board()
    cache['board'] = board
    return json_board(board)


@app.route('/loadGame', methods=['POST'])
def load_game():
    data = request.get_json()
    board = Board(fen=data['state'])
    cache['board'] = board
    return json_board(board)


@app.route('/makeMove', methods=['POST'])
def make_move():
    """Both white and black players are human."""
    data = request.get_json()
    board = cache['board']

    try:
        board.make_safe_move(Move(data['start_pos'], data['end_pos']))
        board.raise_if_game_over()
        return json_board(board)
    except IllegalMove as err:
        return {'error': str(err)}
    except Checkmate:
        winner = BLACK if board.turn == WHITE else WHITE
        return json_board(board, {'end': f"Checkmate: {winner} wins!"})
    except Draw as err:
        return json_board(board, {'end': str(err)})
    except Exception as e:
        for move in board.move_history:
            log.info(move)
        raise Exception(e)


@app.route('/makeMoveAi')
def make_move_ai():
    """Randomly choose a possible move."""
    board = cache['board']
    try:
        board.raise_if_game_over()
        move = dumb.random_move(board)
        board.make_move(move)
        return json_board(board)
    except IllegalMove as err:
        return {'error': str(err)}
    except Checkmate:
        winner = board.turn_name
        return json_board(board, {'end': f"Checkmate: {winner} wins!"})
    except Draw as err:
        return json_board(board, {'end': err.MESSAGE})
