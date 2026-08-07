import time
import random
import multiprocessing
from typing import Callable

import log
from games.base import GameState, Outcome
from games.chess.board import Board, Move

LOW_BOUND = -9999999
HIGH_BOUND = 9999999

# Score of a forced mate, kept well inside the search window so it never collides with the
# alpha/beta bounds themselves.
MATE = 1000000


def _terminal_value(board: Board, depth: int, player: bool) -> int:
    """
    Scores a position that has no legal moves, in the same terms as the rest of the tree.

    Values inside the search are positive for the opponent: _get_best_move plays the root
    move before descending, so _alpha_beta_max nodes belong to the opponent and
    _alpha_beta_min nodes belong to us, and the sign is flipped once at the root.

    Checkmate is therefore scored against whoever is to move, which is what makes one
    function correct at both node types. Scoring it against `player` instead is only ever
    right at one of them, and leaves the engine unable to see mate against itself.

    Deeper mates score lower, so a mate in one is preferred over a mate in five and the
    engine does not shuffle in a won position. Stalemate is a draw and scores zero.
    """
    if not board.is_in_check:  # No moves and no check: stalemate, which is a draw
        return 0

    mate_score = MATE + depth  # Larger remaining depth means a mate closer to the root
    return mate_score if board.turn == player else -mate_score


def simple_eval(board: Board) -> int:
    return board.value


def weighted_eval(board: Board) -> int:
    return board.weighted_value


# Negamax wants every score from the point of view of the player to move, so that a node can
# negate what its children hand back without knowing who anybody is. Board.value and
# Board.weighted_value are absolute - positive for White - so these flip them for Black.
def relative_simple_eval(board: Board) -> int:
    return board.value if board.turn else -board.value


def relative_weighted_eval(board: Board) -> int:
    return board.weighted_value if board.turn else -board.weighted_value


def first_possible_move(board: Board) -> Move:
    """Ultra fast, ultra terrible and predictable."""
    for move in board.legal_moves:
        return move


def random_move(board: Board) -> Move:
    """Ultra terrible, but less predictable."""
    return random.choice(list(board.legal_moves))


def _minimax(board: Board, depth: int, is_maximising_player: bool, player: bool):
    if depth == 0:
        return board.value if player else board.value * -1  # Opposing player has the current turn

    if is_maximising_player:
        score = LOW_BOUND
        for move in board.legal_moves:
            board.make_move(move)
            score = max([_minimax(board, depth - 1, not is_maximising_player, player), score])
            board.unmake_move()
    else:
        score = HIGH_BOUND
        for move in board.legal_moves:
            board.make_move(move)
            score = min([_minimax(board, depth - 1, not is_maximising_player, player), score])
            board.unmake_move()
    return score


def minimax(board: Board, depth: int):
    """
    Implementation of MiniMax algorithm using the standard formulation. This should play identically to negamax,
    but is a bit easier to understand how the algorithm works.

    https://www.chessprogramming.org/Minimax
    """
    score = LOW_BOUND
    best_move = None
    player = board.turn  # Current player is the AI player

    for move in board.legal_moves:
        board.make_move(move)
        value = _minimax(board, depth - 1, False, player)
        board.unmake_move()

        if value > score:
            score = value
            best_move = move
    return best_move


def _negamax(board: Board, depth: int, counter: int):
    if depth == 0:
        counter += 1
        return board.relative_value, counter

    score = LOW_BOUND
    for move in board.legal_moves:
        board.make_move(move)
        value, counter = _negamax(board, depth - 1, counter)
        value *= -1
        board.unmake_move()
        if value > score:
            score = value
    return score, counter


def negamax(board: Board, depth: int, print_count: bool = False):
    """
    Implementation of MiniMax algorithm using the negamax formulation. This is a search tree that searches all possible
    moves making optimal choices for each player in accordance to optimising the cost function (in this case game
    value). Then the original move that could lead the best score is chosen.

    https://www.chessprogramming.org/Minimax
    https://www.chessprogramming.org/Negamax
    """
    score = LOW_BOUND
    best_move = None
    counter = 0
    start_time = time.time()
    assert depth > 0

    for move in board.legal_moves:
        board.make_move(move)
        value, counter = _negamax(board, depth - 1, counter)
        value = value * -1
        board.unmake_move()

        if value > score:
            score = value
            best_move = move

    if print_count:
        elapsed = time.time() - start_time
        log.info(
            f"Evaluations: {counter} in {elapsed}s at {counter/elapsed} evals/s."
        )
    return best_move


def terminal_score(outcome: Outcome, turn: bool, depth: int) -> int:
    """
    Scores a finished position from the point of view of the player to move in it.

    Losing is the worst thing that can happen to the side it happens to, so a lost position
    scores negatively whoever is searching - which is what lets one function serve every node
    of a negamax tree, where each node speaks for whoever is to move in it.

    Deeper wins score lower, `depth` being the depth still remaining, so a mate in one is
    preferred to a mate in five and the engine does not shuffle in a won position.
    """
    if outcome.winner is None:
        return 0
    return (MATE + depth) if outcome.winner == turn else -(MATE + depth)


def _negamax_ab(state: GameState, depth: int, alpha: int, beta: int, evaluate: Callable) -> int:
    """
    Alpha-beta in the negamax formulation: every node returns a score from the point of view
    of the player to move in it, and its parent negates what it gets back.

    The window is checked before the horizon so that a game already won scores as a win rather
    than being handed to the evaluation. For chess `outcome` is the inherited None - it has no
    win condition that can be spotted without generating moves - so this costs one attribute
    load and the node order is unchanged.
    """
    outcome = state.outcome
    if outcome is not None:
        return terminal_score(outcome, state.turn, depth)

    if depth == 0:
        return evaluate(state)

    any_move = False
    for move in state.legal_moves:
        any_move = True
        state.make_move(move)
        score = -_negamax_ab(state, depth - 1, -beta, -alpha, evaluate)
        state.unmake_move()

        if score >= beta:
            return beta  # Fail-hard, matching the search this replaces bound for bound
        if score > alpha:
            alpha = score

    if not any_move:  # Nothing to play: checkmate, stalemate, or a full board
        return terminal_score(state.outcome_without_moves, state.turn, depth)

    return alpha


def _negamax_root_move(state: GameState, depth: int, move, evaluate: Callable):
    """One root move, searched on a full window so that root moves stay independent."""
    state.make_move(move)
    value = -_negamax_ab(state, depth - 1, LOW_BOUND, HIGH_BOUND, evaluate)
    state.unmake_move()
    return move, value


def _alpha_beta_min(board: Board, depth: int, alpha: int, beta: int, player: bool, board_eval: Callable, counter: int):
    if depth == 0:
        counter += 1
        value = board_eval(board) if not player else board_eval(board) * -1
        return value, counter

    best = HIGH_BOUND
    i = 0
    for move in board.legal_moves:
        board.make_move(move)
        score, counter = _alpha_beta_max(board, depth - 1, alpha, beta, player, board_eval, counter)
        board.unmake_move()

        best = min([score, best])
        beta = min([beta, best])
        if beta <= alpha:
            return best, counter
        i += 1

    if i == 0:  # No legal moves: checkmate or stalemate
        return _terminal_value(board, depth, player), counter
    return beta, counter


def _alpha_beta_max(board: Board, depth: int, alpha: int, beta: int, player: bool, board_eval: Callable, counter: int):
    if depth == 0:
        counter += 1
        value = board_eval(board) if not player else board_eval(board) * -1
        return value, counter

    best = LOW_BOUND
    i = 0
    for move in board.legal_moves:
        board.make_move(move)
        score, counter = _alpha_beta_min(board, depth - 1, alpha, beta, player, board_eval, counter)
        board.unmake_move()

        best = max([score, best])
        alpha = max([alpha, best])
        if beta <= alpha:
            return best, counter
        i += 1

    if i == 0:  # No legal moves: checkmate or stalemate
        return _terminal_value(board, depth, player), counter

    return alpha, counter


def _get_best_move(_board: Board, _depth: int, _base_move: Move, _board_eval: Callable):
    """Helper function for running alpha_beta on multiple threads."""

    alpha = LOW_BOUND
    beta = HIGH_BOUND
    _counter = 0
    _player = _board.turn

    _board.make_move(_base_move)

    _value, _counter = _alpha_beta_max(_board, _depth - 1, alpha, beta, _player, _board_eval, _counter)
    _value *= -1

    _board.unmake_move()

    return _base_move, _value, _counter


def alpha_beta(board: Board, depth: int = 3, board_eval: Callable = weighted_eval, print_count: bool = False):
    """
    Implementation of Alpha-Beta pruning to optimise the MiniMax algorithm. This stops evaluating a move when at least
    one possibility has been found that proves the move to be worse than a previously examined move. Should play
    identically to negamax for the same search depth.

    https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning
    """

    best_move = None
    start_time = time.time()
    assert depth > 0

    legal_moves = list(board.legal_moves)

    jobs = []
    for move in legal_moves:
        new_board = Board(board.fen)
        jobs.append((new_board, depth, move, board_eval))

    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    result = pool.starmap(_get_best_move, jobs)

    counter = 0
    score = LOW_BOUND
    for move, value, sub_counter in result:
        counter += sub_counter
        if value > score:
            score = value
            best_move = move

    if print_count:
        elapsed = time.time() - start_time
        log.info(
            f"Evaluations: {counter} in {elapsed}s at {counter / elapsed} evals/s."
        )
    return best_move
