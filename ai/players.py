"""
Naming a player, in one place, so every harness means the same thing by it.

A player in this project has always been a callable taking a state and returning a move -
`ai.search.random_move` is one, and so is a lambda closing over a depth. That is a good contract
and nothing here replaces it. What was missing is a way to *name* one, so that a terminal front
end, a match harness and a benchmark can all be pointed at the same opponent without each
growing its own notion of what opponents exist.

    random                             a uniformly random legal move
    minimax                            alpha-beta at the game's default depth
    minimax:9                          alpha-beta at a given depth
    model:models/tictactoe-best.pt     a trained network's policy, with no search at all
    model:models/tictactoe-best.pt+mcts:200    the same network, thinking 200 simulations ahead
    human                              read a move from the terminal

The last two lines are the reason this exists. "The network on its own" and "the network with
search" are the same player differing by one clause, so comparing raw intuition against intuition
that thinks ahead is a change of spec rather than a change of code - and both go through the same
benchmark, the same match harness and the same game loop as the classical players.

Kept free of third-party imports. `model:` reaches into `ai.zero`, which needs PyTorch, but it
does so only when a model is actually asked for, so the engine and its tests stay installable
with nothing but the standard library. `minimax:` can reach into `ai.native` the same way, for
the Rust alpha-beta - needs no third-party package, only the extension built.
"""

from typing import Any, Callable, List, Optional, Type

import log
from games.base import GameState

# The depth a computer opponent searches when the game does not declare a better answer.
DEFAULT_DEPTH = 4

# Simulations a model player runs when asked for search without being told how much.
DEFAULT_SIMULATIONS = 200


def default_depth(game: Type[GameState]) -> int:
    """
    How deep to search unless told otherwise.

    A game that declares SOLVED_DEPTH can be searched to the end of itself, and a search to the
    end is perfect play, so there is no reason to offer anything shallower.
    """
    return DEFAULT_DEPTH if game.SOLVED_DEPTH is None else game.SOLVED_DEPTH


class UnknownPlayer(ValueError):
    """A spec naming a kind of player that does not exist, or malformed."""


def _parse(spec: str) -> List[List[str]]:
    """`a:1+b:2` into `[['a', '1'], ['b', '2']]`, which is as much syntax as this has."""
    parts = []
    for clause in spec.split('+'):
        clause = clause.strip()
        if not clause:
            raise UnknownPlayer(f'{spec!r} has an empty clause')
        parts.append(clause.split(':', 1))
    return parts


def _int(text: str, what: str) -> int:
    try:
        return int(text)
    except ValueError:
        raise UnknownPlayer(f'{text!r} is not a number of {what}') from None


def player(
    spec: str, human: Optional[Callable] = None, engine: str = 'auto',
) -> Callable[[GameState], Any]:
    """
    The move chooser a spec names.

    `human` is passed in rather than imported because reading from a terminal belongs to the
    terminal front end, not here - a match harness has no business owning a prompt.

    `engine` chooses which alpha-beta a `minimax:` player searches with, and only that clause
    reads it - it has nothing to say about a random, human or model player. `'auto'` (the
    default) takes the Rust search where it is built and speaks for the game being played,
    `'python'` never looks for it, and `'rust'` insists on it rather than quietly taking the
    slow path. Same three-valued contract as `ai.zero.train.choose_engine`.
    """
    from ai.search import alpha_beta, random_move  # Local, to keep import order obvious

    clauses = _parse(spec)
    head, tail = clauses[0], clauses[1:]
    kind = head[0].lower()

    if kind == 'human':
        if human is None:
            raise UnknownPlayer('a human player has nothing to read moves from here')
        chooser = human
    elif kind == 'random':
        chooser = random_move
    elif kind == 'minimax':
        depth = _int(head[1], 'plies') if len(head) > 1 else None
        chooser = _minimax(alpha_beta, depth, engine)
    elif kind == 'model':
        if len(head) < 2 or not head[1]:
            raise UnknownPlayer('a model player needs a checkpoint: model:<path>')
        return _model(head[1], tail)
    else:
        raise UnknownPlayer(f'{head[0]!r} is not a kind of player. Try one of: {", ".join(KINDS)}')

    if tail:
        raise UnknownPlayer(f'{kind} does not take {"+".join(":".join(c) for c in tail)!r}')
    return chooser


def _minimax(alpha_beta: Callable, depth: Optional[int], engine: str) -> Callable:
    """
    Alpha-beta, resolving its depth per position and its implementation once per player.

    The implementation is decided on the first move rather than here, because the game is not
    known until a position turns up to search - and decided once, rather than on every move,
    so `'auto'` logs its fallback reason once per player instead of once per ply.
    """
    search = alpha_beta if engine == 'python' else None

    def chooser(state: GameState):
        nonlocal search
        if search is None:
            search = _choose_search(engine, type(state))
        ply_depth = depth if depth is not None else default_depth(type(state))
        return search(state, depth=ply_depth)

    return chooser


def _choose_search(requested: str, game: Type[GameState]) -> Callable:
    """
    Which alpha-beta a `minimax:` player searches with, and a line in the log saying why.

    Only reached when `_minimax`'s chooser was left to decide for itself - the `python` case is
    resolved without this, and never touches `ai.native` at all.
    """
    from ai import native
    from ai.search import alpha_beta as python_alpha_beta

    if native.available(game):
        return native.alpha_beta
    if requested == 'rust':
        raise ValueError(native.why_unavailable(game))

    log.info(f'{game.__name__} minimax on the Python search: {native.why_unavailable(game)}')
    return python_alpha_beta


def _model(path: str, tail: List[List[str]]) -> Callable:
    """
    A trained network, with or without search on top.

    The import is deliberately late. `ai.zero` needs PyTorch, and nothing else in this project
    does, so an engine that never mentions a model never has to have it installed.
    """
    simulations = 0
    for clause in tail:
        if clause[0].lower() != 'mcts':
            raise UnknownPlayer(f'{clause[0]!r} is not something a model player can be given')
        simulations = _int(clause[1], 'simulations') if len(clause) > 1 else DEFAULT_SIMULATIONS

    try:
        from ai.zero.player import model_player
    except ImportError as error:  # pragma: no cover - depends on the environment, not the code
        raise UnknownPlayer(
            f'a model player needs the learning extras: pip install -r requirements-zero.txt '
            f'({error})'
        ) from None

    return model_player(path, simulations=simulations)


KINDS = ('human', 'random', 'minimax', 'model')


def describe(spec: str) -> str:
    """A spec in words, for a log line that should not need the reader to parse a spec."""
    clauses = _parse(spec)
    kind = clauses[0][0].lower()

    if kind == 'random':
        return 'random moves'
    if kind == 'minimax':
        depth = clauses[0][1] if len(clauses[0]) > 1 else 'default'
        return f'alpha-beta at depth {depth}'
    if kind == 'model':
        name = clauses[0][1] if len(clauses[0]) > 1 else '?'
        for clause in clauses[1:]:
            if clause[0].lower() == 'mcts':
                count = clause[1] if len(clause) > 1 else DEFAULT_SIMULATIONS
                return f'{name} with {count} simulations'
        return f'{name}, raw policy with no search'
    return kind
