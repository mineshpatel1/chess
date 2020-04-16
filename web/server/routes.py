from typing import Optional, Dict, Any
from flask import request

import log
from ai import dumb, algo
from engine.constants import WHITE, BLACK
from engine.board import Board, Move, SQUARES_VFLIP
from engine.exceptions import IllegalMove, Checkmate, Draw
from web.server import app

cache = {
    'board': Board(),
}


def log_exception(board: Board, e: Exception):
    log.error(str(e))
    log.info(f"Board state: {board.fen}")
    log.info('Move history:')
    for move in board.move_history:
        log.info(f"    {move.uci}")
    raise Exception(e)


def json_board(board: Board, params: Optional[Dict] = None):
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
        board.raise_if_game_over()
        board.make_safe_move(Move(data['start_pos'], data['end_pos']))
        return json_board(board)
    except IllegalMove as err:
        return {'error': str(err)}
    except Checkmate:
        winner = 'Black' if board.turn == WHITE else 'White'
        return json_board(board, {'end': f"Checkmate: {winner} wins!"})
    except Draw as err:
        return json_board(board, {'end': err.MESSAGE})
    except Exception as e:
        log_exception(board, e)


@app.route('/makeMoveAi')
def make_move_ai():
    """Randomly choose a possible move."""
    board = cache['board']
    try:
        board.raise_if_game_over()

        # move = dumb.random_move(board)
        move = algo.negamax(board, depth=3)
        
        board.make_move(move)
        board.raise_if_game_over()
        return json_board(board)
    except IllegalMove as err:
        return {'error': str(err)}
    except Checkmate:
        winner = 'Black' if board.turn == WHITE else 'White'
        return json_board(board, {'end': f"Checkmate: {winner} wins!"})
    except Draw as err:
        return json_board(board, {'end': err.MESSAGE})
    except Exception as e:
        log_exception(board, e)
