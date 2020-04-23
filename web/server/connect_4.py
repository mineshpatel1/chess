from flask import request
from typing import Optional, Dict

from connect_4.game import Connect4, IllegalMove, SLOTS_VFLIP
from connect_4.ai import minimax
from web.server import app

cache = {
    'game': Connect4(),
}


def _end_message(result: int) -> str:
    if result == 1:
        return 'Game over: Red wins!'
    elif result == -1:
        return 'Game over: Yellow wins!'
    elif result == 0:
        return 'Game over: Draw'


def _json_board(board: Connect4, params: Optional[Dict] = None):
    by_rank = {}
    for slot in SLOTS_VFLIP:
        rank = int(slot / 7)
        file = slot % 7

        by_rank[rank] = by_rank.get(rank, [])

        _square = {
            'rank': rank,
            'file': file,
            'index': int(slot),
        }

        piece = board.piece_at(slot)
        if piece:
            _square['piece'] = 'red' if piece.colour else 'yellow'

        by_rank[rank].append(_square)

    by_rank_reverse = {}
    for i, rank in enumerate(sorted(by_rank.keys(), reverse=True)):
        by_rank_reverse[i] = by_rank[rank]

    payload = {'game': by_rank_reverse, 'turn': board.turn_name, 'mhn': board.mhn}
    if params:
        payload.update(params)
    return payload


@app.route('/connect4/newGame', methods=['POST'])
def c4_new_game():
    data = request.get_json()
    c4 = Connect4()
    cache['game'] = c4

    if not data['player']:
        return c4_make_move_ai()  # Make first move

    return _json_board(c4)


@app.route('/connect4/makeMove', methods=['POST'])
def c4_make_move():
    """Both white and black players are human."""
    data = request.get_json()
    c4 = cache['game']

    try:
        c4.make_move(c4.file_to_move(data['move']))
        end_result = c4.end_result
        if end_result is not None:
            return _json_board(c4, {'end': _end_message(end_result)})
        return _json_board(c4)
    except IllegalMove:
        return {'error': "Illegal move."}


@app.route('/connect4/makeMoveAi')
def c4_make_move_ai():
    """Randomly choose a possible move."""
    c4 = cache['game']
    # c4.make_random_move()
    move = minimax(c4, 4)
    c4.make_move(move)

    end_result = c4.end_result
    if end_result is not None:
        return _json_board(c4, {'end': _end_message(end_result)})
    return _json_board(c4)
