"""
Self-play through the Rust engine, when it is built.

The division of labour is the one `selfplay.play_games` already uses - the search hands out the
positions it needs evaluated and is handed back the answers - moved across a language boundary.
Rust owns the games, the trees and the encoding; the network stays in PyTorch, so there is no
second copy of the model and nothing to export between generations.

What that buys is the batch. The Python driver evaluates one position per game per pass, so its
batch is capped by how many trees Python can afford to walk; this one holds every game in the
generation at once, which is the batch size a GPU is worth having for.

Optional in exactly the way PyTorch is optional: `available()` is false when the extension is not
built, `train` falls back to the Python engine, and `tests/zero/test_fast.py` skips itself. See
`rust/README.md` for how to build it.

The two engines play the *same games*. That is checked rather than hoped - the Rust search
reproduces the PUCT arithmetic and CPython's Mersenne Twister, so a generation played either way
comes out example for example identical, which is what makes `--engine` a fair comparison.
"""

from typing import Any, List, NamedTuple, Optional, Tuple

try:
    import zero_rs
except ImportError:  # pragma: no cover - depends on the environment, not the code
    zero_rs = None

from ai.zero.selfplay import FINAL_TEMPERATURE, OPENING_PLIES, TEMPERATURE, TEMPERATURE_MOVES
from ai.zero.mcts import DIRICHLET_ALPHA, DIRICHLET_EPSILON, EXPLORATION

GAME = 'Connect4'


class Generation(NamedTuple):
    """One generation of self-play, in the stacked form the replay buffer already stores."""

    examples: Tuple[Any, Any, Any]  # planes int8, policy float32, value float32
    lengths: List[int]
    drawn: int

    @property
    def positions(self) -> int:
        return len(self.examples[2])


def available(game=None) -> bool:
    """
    Whether the Rust engine is built and speaks for this game.

    The encoder's shape is checked too, so a plane layout that has moved on since the extension
    was built refuses here rather than training on a board read the wrong way round.
    """
    if zero_rs is None:
        return False
    if game is not None and game.__name__ != zero_rs.GAME:
        return False

    from games.connect4.encoding import Connect4Encoder

    return (tuple(zero_rs.PLANE_SHAPE) == tuple(Connect4Encoder.PLANE_SHAPE)
            and zero_rs.POLICY_SIZE == Connect4Encoder.POLICY_SIZE)


def why_unavailable(game=None) -> str:
    """A line for the log when `available` is false, since falling back silently hides a build."""
    if zero_rs is None:
        return 'the Rust engine is not built; see rust/README.md'
    if game is not None and game.__name__ != zero_rs.GAME:
        return f'the Rust engine plays {zero_rs.GAME}, not {game.__name__}'
    return 'the Rust engine was built against a different encoder'


def play_games(
    net,
    count: int,
    simulations: int,
    exploration: float = EXPLORATION,
    seed: int = 0,
    opening_plies: int = OPENING_PLIES,
    temperature: float = TEMPERATURE,
    final_temperature: float = FINAL_TEMPERATURE,
    temperature_moves: int = TEMPERATURE_MOVES,
    dirichlet_alpha: float = DIRICHLET_ALPHA,
    dirichlet_epsilon: float = DIRICHLET_EPSILON,
    in_flight: Optional[int] = None,
    device: Optional[str] = None,
    on_finished=None,
) -> Generation:
    """
    Plays `count` games, evaluating every game's pending position in one pass.

    `in_flight` defaults to the whole generation, which is the point of the exercise; lowering it
    only shrinks the batch, and cannot change what is played.
    """
    import torch
    import torch.nn.functional as F

    from ai.zero.net import MASKED

    if not available():
        raise RuntimeError(why_unavailable())

    device = device or next(net.parameters()).device
    engine = zero_rs.SelfPlay(
        games=count, simulations=simulations, exploration=exploration, seed=str(seed),
        dirichlet_alpha=dirichlet_alpha, dirichlet_epsilon=dirichlet_epsilon,
        temperature=temperature, final_temperature=final_temperature,
        temperature_moves=temperature_moves, opening_plies=opening_plies, in_flight=in_flight)

    net.eval()
    reported = 0
    with torch.inference_mode():
        while True:
            batch = engine.pending()
            if batch is None:
                break
            planes, legal = batch

            logits, values = net(torch.from_numpy(planes).to(device, torch.float32))

            # Added rather than assigned, and before the softmax: the same masking
            # `net.evaluate_batch` does, so the two engines see the same priors.
            mask = torch.where(torch.from_numpy(legal).to(device), 0.0, MASKED)
            priors = F.softmax(logits + mask, dim=1)

            engine.submit(priors.to('cpu', torch.float32).numpy(),
                          values.to('cpu', torch.float32).numpy())

            if on_finished and engine.completed != reported:
                reported = engine.completed
                on_finished(reported, count)

    return Generation(engine.examples(), engine.lengths(), engine.drawn())


def as_examples(generation: Generation) -> List:
    """
    The stacked arrays back as `selfplay.Example`s, for a caller that still wants them one by one.

    A copy of the whole generation into Python objects, which is what the arrays exist to avoid -
    so this is for tests and for the buffer's slow path, not for a training run.
    """
    from ai.zero.selfplay import Example

    planes, policy, value = generation.examples
    return [
        Example(planes[index].tolist(), policy[index].tolist(), float(value[index]))
        for index in range(len(value))
    ]
