import random
import multiprocessing
from typing import Callable, Optional

from games.base import GameState, Outcome

LOW_BOUND = -9999999
HIGH_BOUND = 9999999

# Score of a forced win, kept well inside the search window so it never collides with the
# alpha/beta bounds themselves.
MATE = 1000000


def random_move(state: GameState):
    """Ultra terrible, but less predictable."""
    return random.choice(list(state.legal_moves))


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
    of the player to move in it, and its parent negates what it gets back. That is what lets
    one function serve both players, and one `terminal_score` serve every node.

    Terminality is asked about before the horizon, so a game already won scores as a win
    rather than being handed to the evaluation. Games won while moves remain - a line of
    three, a line of four - need that. Chess has no win condition it can spot without
    generating moves, so its `outcome` is the inherited None and this costs one attribute
    load.

    https://www.chessprogramming.org/Negamax
    https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning
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
            return beta  # Fail-hard
        if score > alpha:
            alpha = score

    if not any_move:  # Nothing to play: checkmate, stalemate, or a full board
        return terminal_score(state.outcome_without_moves, state.turn, depth)

    return alpha


def _root_move_score(state: GameState, depth: int, move, evaluate: Callable):
    """
    Scores one root move on a full window.

    A full window per root move is weaker pruning than carrying alpha across them, but it is
    what keeps root moves independent of each other, and that independence is what makes them
    safe to hand to separate processes.
    """
    state.make_move(move)
    value = -_negamax_ab(state, depth - 1, LOW_BOUND, HIGH_BOUND, evaluate)
    state.unmake_move()
    return move, value


def alpha_beta(state: GameState, depth: int = 3, evaluate: Optional[Callable] = None):
    """
    Picks a move by searching `depth` plies with alpha-beta pruning, splitting the root across
    a process pool for games that branch widely enough to be worth it.

    Ties go to the move generated first, which keeps the search reproducible.
    """
    assert depth > 0
    if evaluate is None:
        evaluate = state.DEFAULT_EVAL

    jobs = [(state.copy(), depth, move, evaluate) for move in state.legal_moves]

    if state.PARALLEL_ROOT:
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            results = pool.starmap(_root_move_score, jobs)
    else:
        results = [_root_move_score(*job) for job in jobs]

    best_move, best_score = None, LOW_BOUND
    for move, value in results:
        if value > best_score:
            best_score = value
            best_move = move
    return best_move
