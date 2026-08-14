"""
Alpha-beta through the Rust engine, when it is built.

The ladder's opponents are `ai.search.alpha_beta`, unbatched and cold every move - and at
`minimax:8` that is 391ms a move, most of a generation once the challenger's own search is fast.
This is the same division of labour `ai/zero/fast.py` already uses for self-play, at a much
smaller scale: a position crosses the boundary as two bitboards and a turn, Rust searches it, and
the answer comes straight back. There is no network here and nothing to keep in step across a
call - `c4_core::search` is stateless from one move to the next, exactly as `ai.search` is.

Optional in exactly the way the self-play engine is optional: `available()` is false when the
extension is not built, `ai.players.player` falls back to the Python search, and
`tests/connect4/test_native.py` skips itself. See `rust/README.md` for how to build it.

The two searches are the *same* search, not merely as fast one. `ai.search.alpha_beta` has no
transposition table, no move ordering beyond `CENTRE_FIRST`, and no shortcut for an in-progress
win, so this port has none either - adding any of them would change which move comes back, and a
change of player is not something a parity test can wave through.
"""

from typing import Callable, Optional

try:
    import zero_rs
except ImportError:  # pragma: no cover - depends on the environment, not the code
    zero_rs = None

from games.base import GameState
from games.connect4.constants import RED, YELLOW

GAME = 'Connect4'


def available(game=None) -> bool:
    """
    Whether the Rust search is built, speaks for this game, and shares its scoring convention.

    `MATE` is checked rather than merely the module's presence: a build made before `best_move`
    existed has no such attribute, and one built against a different `ai.search.MATE` would score
    forced wins differently, which `alpha_beta` here has no way to notice on its own.
    """
    if zero_rs is None or not hasattr(zero_rs, 'best_move'):
        return False
    if game is not None and game.__name__ != zero_rs.GAME:
        return False

    from ai.search import MATE

    return zero_rs.MATE == MATE


def why_unavailable(game=None) -> str:
    """A line for the log when `available` is false, since falling back silently hides a build."""
    if zero_rs is None:
        return 'the Rust engine is not built; see rust/README.md'
    if not hasattr(zero_rs, 'best_move'):
        return 'the Rust engine is built without the alpha-beta search; rebuild it'
    if game is not None and game.__name__ != zero_rs.GAME:
        return f'the Rust engine plays {zero_rs.GAME}, not {game.__name__}'
    return 'the Rust engine was built against a different MATE score'


def alpha_beta(state: GameState, depth: int, evaluate: Optional[Callable] = None):
    """
    `ai.search.alpha_beta`, run in Rust.

    Declines a custom `evaluate` rather than ignoring it: the Rust side always plays
    `games.connect4.evaluation.weighted_eval`, and a caller that measures move ordering or ties
    with a different one - `tests/connect4/test_search_equivalence.py` does both - needs the
    Python search, silently getting a different evaluation would be a wrong answer, not a fast one.
    """
    if evaluate is not None:
        raise ValueError('the Rust search does not take a custom evaluation; use ai.search')
    if not available():
        raise RuntimeError(why_unavailable())

    return zero_rs.best_move(state.discs[YELLOW], state.discs[RED], state.turn, depth)
