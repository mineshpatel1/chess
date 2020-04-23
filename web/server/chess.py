from typing import Optional, Dict
from flask import request

import log
from ai import algorithms
from game.constants import WHITE
from game.board import Board, Move, SQUARES_VFLIP
from game.exceptions import IllegalMove, Checkmate, Draw
from web.server import app

cache = {
    'game': Board(),
}


def _log_exception(board: Board, e: Exception):
    log.error(str(e))
    log.info(f"Board state: {board.fen}")
    log.info('Move history:')
    for move in board.move_history:
        log.info(f"    {move.uci}")
    raise Exception(e)


def _json_board(board: Board, params: Optional[Dict] = None):
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


@app.route('/chess/newGame', methods=['POST'])
def chess_new_game():
    data = request.get_json()
    board = Board()
    cache['game'] = board

    if not data['player']:
        return chess_make_move_ai()  # Make first move
    return _json_board(board)


@app.route('/chess/loadGame', methods=['POST'])
def chess_load_game():
    data = request.get_json()
    if 'state' in data:
        board = Board(fen=data['state'])
        cache['game'] = board
    else:
        board = cache['game']
    return _json_board(board)


@app.route('/chess/makeMove', methods=['POST'])
def chess_make_move():
    """Both white and black players are human."""
    data = request.get_json()
    board = cache['game']

    try:
        board.raise_if_game_over()
        board.make_safe_move(Move(data['start_pos'], data['end_pos']))
        return _json_board(board)
    except IllegalMove as err:
        return {'error': str(err)}
    except Checkmate:
        winner = 'Black' if board.turn == WHITE else 'White'
        return _json_board(board, {'end': f"Checkmate: {winner} wins!"})
    except Draw as err:
        return _json_board(board, {'end': err.MESSAGE})
    except Exception as e:
        _log_exception(board, e)


@app.route('/chess/makeMoveAi')
def chess_make_move_ai():
    """Randomly choose a possible move."""
    board = cache['game']
    try:
        board.raise_if_game_over()

        # move = algorithms.random_move(game)
        move = algorithms.alpha_beta(board, depth=3)

        board.make_move(move)
        board.raise_if_game_over()
        return _json_board(board)
    except IllegalMove as err:
        return {'error': str(err)}
    except Checkmate:
        winner = 'Black' if board.turn == WHITE else 'White'
        return _json_board(board, {'end': f"Checkmate: {winner} wins!"})
    except Draw as err:
        return _json_board(board, {'end': err.MESSAGE})
    except Exception as e:
        _log_exception(board, e)
