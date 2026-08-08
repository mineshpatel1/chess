"""
A trained checkpoint as a player, with or without search.

Both halves matter and they answer different questions. `simulations=0` asks the network alone:
one forward pass, take the most likely legal move, no lookahead whatsoever - the intuition, with
nothing to hide behind. Anything above that puts PUCT on top and lets the same network think.

Comparing the two is the most informative thing you can do with a trained model. A network whose
raw policy is already near-perfect has genuinely learned the game; one that only plays well with
500 simulations has learned to be a useful prior for a search that is doing the real work. Both
are legitimate results, and telling them apart requires being able to run the network with the
search switched off - which is why this is a parameter rather than two implementations.
"""

from typing import Any, Callable

from games.base import GameState
from ai.zero.checkpoint import load
from ai.zero.mcts import MCTS
from ai.zero.net import evaluate


def model_player(path: str, simulations: int = 0) -> Callable[[GameState], Any]:
    """
    The move chooser a checkpoint plus a simulation count names.

    The network is loaded once and closed over, so a benchmark walking five thousand positions
    pays for the file exactly once. Deterministic by construction: no Dirichlet noise, and the
    move is the most-visited rather than a sample, because a player being measured should give
    its actual opinion rather than a draw from it.
    """
    blob = load(path)
    net = blob['net']

    def chooser(state: GameState):
        encoder = state.ENCODER
        if encoder is None:
            raise ValueError(f'{type(state).__name__} has no ENCODER, so it cannot be learned')
        _check_shape(blob, encoder, path)

        if simulations <= 0:
            return _raw_move(net, state, encoder)

        search = MCTS(_evaluator(net, encoder), encoder, simulations=simulations)
        return search.search(state, noise=False).move

    chooser.checkpoint = path  # type: ignore[attr-defined]  Useful in logs and reports
    chooser.simulations = simulations  # type: ignore[attr-defined]
    return chooser


def _check_shape(blob, encoder, path: str) -> None:
    """
    Refuses a checkpoint whose network does not fit the encoder now in the source.

    An encoding is part of a trained network, not a detail around it: change the planes and every
    weight in the first layer means something else. Without this the mismatch surfaces as
    `mat1 and mat2 shapes cannot be multiplied (1x9 and 18x64)` from inside torch, which says
    nothing about what to do - and if the shapes happened to still line up it would not surface
    at all, it would just play badly.
    """
    config = blob.get('config', {})
    stored = tuple(config.get('plane_shape', ()))
    current = tuple(encoder.PLANE_SHAPE)

    if stored != current:
        raise ValueError(
            f'{path} was trained on planes of {stored}, but {encoder.__name__} now produces '
            f'{current}. The encoding changed since this was trained - retrain it.'
        )

    if config.get('policy_size') != encoder.POLICY_SIZE:
        raise ValueError(
            f'{path} has {config.get("policy_size")} actions, but {encoder.__name__} has '
            f'{encoder.POLICY_SIZE}. Retrain it.'
        )


def _evaluator(net, encoder):
    """Wraps a network as the (priors, value) callable MCTS expects."""
    def evaluator(state: GameState):
        return evaluate(net, state, encoder)
    return evaluator


def _raw_move(net, state: GameState, encoder) -> Any:
    """
    The network's own favourite legal move, with no search at all.

    Argmax over the *masked* policy, so an illegal move cannot win by having the largest logit.
    """
    priors, _ = evaluate(net, state, encoder)

    best_move, best_prior = None, -1.0
    for move in state.legal_moves:
        prior = priors[encoder.action_index(move)]
        if prior > best_prior:
            best_move, best_prior = move, prior
    return best_move


def model_value(path: str) -> Callable[[GameState], float]:
    """
    A checkpoint's value head on its own, for grading calibration against the true value.

    Separate from the move chooser because it answers a separate question. A network can pick
    good moves from a badly calibrated value, and it can evaluate positions well while its policy
    fumbles; `ai.oracle.benchmark` takes this as `value_fn` so the two are scored apart.
    """
    blob = load(path)
    net = blob['net']

    def value_of(state: GameState) -> float:
        encoder = state.ENCODER
        if state.is_game_over:
            from ai.zero.mcts import terminal_value
            return terminal_value(state)
        return evaluate(net, state, encoder)[1]

    return value_of
