"""
The ladder's network challenger through the Rust engine, when it is built.

`ai/zero/train.py::_climb` plays the challenger's own MCTS one game at a time in Python - a batch
of one forward pass per network call, which is why it runs on the CPU rather than the GPU a
training run otherwise uses: a batch of one costs more on a card than off it. This is the same fix
self-play already got: play every game a rung needs *in flight* together, so the network's forward
passes batch across games and the GPU is worth using again. The opponent's `minimax:N` moves cost
nothing extra - they resolve inside Rust with no round trip at all, exactly as `ai.native.alpha_beta`
already does alone.

Optional in exactly the way `ai.native` is optional: `available()` is false when the extension is
not built or does not speak for this game, `ai.ladder.climb` falls back to playing the rung one
game at a time, and `tests/zero/test_ladder_fast.py` skips itself.
"""

import time
from typing import Callable, Optional, Sequence

try:
    import zero_rs
except ImportError:  # pragma: no cover - depends on the environment, not the code
    zero_rs = None

from ai.match import MatchResult
from ai.zero.mcts import EXPLORATION

GAME = 'Connect4'


def available(game=None) -> bool:
    """Whether the Rust ladder driver is built and speaks for this game."""
    if zero_rs is None or not hasattr(zero_rs, 'LadderMatch'):
        return False
    if game is not None and game.__name__ != GAME:
        return False

    from ai.search import MATE
    return zero_rs.MATE == MATE


def why_unavailable(game=None) -> str:
    """A line for the log when `available` is false, since falling back silently hides a build."""
    if zero_rs is None:
        return 'the Rust engine is not built; see rust/README.md'
    if not hasattr(zero_rs, 'LadderMatch'):
        return 'the Rust engine is built without the ladder driver; rebuild it'
    if game is not None and game.__name__ != GAME:
        return f'the Rust ladder driver plays {GAME}, not {game.__name__}'
    return 'the Rust engine was built against a different search'


def climb(
    net,
    encoder,
    depth: int,
    chosen: Sequence,
    simulations: int,
    exploration: float = EXPLORATION,
    in_flight: Optional[int] = None,
    device=None,
    on_batch: Optional[Callable] = None,
) -> MatchResult:
    """
    Plays one rung - `chosen` against a `depth`-ply alpha-beta - with every game in flight
    together, and returns the challenger's `MatchResult`.

    `chosen` is `ai.ladder.climb`'s already-selected, already-paired openings - the same list
    reused for every rung - so this plays exactly the games the slow path would, only batched.
    Run on whatever device `net` already lives on: unlike the one-game-at-a-time path, a real
    batch is what makes the GPU worth it again.

    `on_batch(positions, network_seconds)` is called once per forward pass, the same hook
    `ai/zero/fast.py::play_games` offers, so a caller can see how much of the cost is the network.
    """
    import torch
    import torch.nn.functional as F

    from ai.zero.net import MASKED
    from games.connect4.constants import RED, YELLOW

    if not available():
        raise RuntimeError(why_unavailable())

    openings = [(state.discs[YELLOW], state.discs[RED], state.turn) for state in chosen]
    device = device or next(net.parameters()).device
    engine = zero_rs.LadderMatch(
        openings=openings, depth=depth, simulations=simulations, exploration=exploration,
        in_flight=in_flight)

    net.eval()
    with torch.inference_mode():
        while True:
            batch = engine.pending()
            if batch is None:
                break
            planes, legal = batch
            started = time.perf_counter()

            logits, values = net(torch.from_numpy(planes).to(device, torch.float32))

            # Added rather than assigned, and before the softmax: the same masking
            # `net.evaluate_batch` does, so this sees the same priors the slow path would.
            mask = torch.where(torch.from_numpy(legal).to(device), 0.0, MASKED)
            priors = F.softmax(logits + mask, dim=1)

            answers = (priors.to('cpu', torch.float32).numpy(),
                       values.to('cpu', torch.float32).numpy())
            if on_batch:
                on_batch(len(planes), time.perf_counter() - started)
            engine.submit(*answers)

    wins, draws, losses = engine.tally()
    return MatchResult(wins, draws, losses)
