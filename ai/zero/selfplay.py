"""
Generating training data by playing the network against itself.

Each position played produces one example: the board as its mover saw it, the search's visit
counts as the policy to imitate, and - once the game is finished - what the game was worth to
that mover. The network is then trained to predict, in one position, what the search concluded
after looking ahead from it. That is the whole trick: search is slow and better than the network,
so the network chases it, and the better network makes the next search better still.

Two things about the value target are easy to get wrong and silent when wrong:

* **It is from the mover's own point of view.** A position whose player to move went on to win is
  `+1` for that position, whichever player it was.
* **A draw is 0, not a sign.** Most tic-tac-toe self-play games end level and most examples should
  say so.
"""

import random
from typing import (
    Any,
    Callable,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

from games.base import Encoder, GameState
from ai.zero.mcts import (
    DIRICHLET_ALPHA,
    DIRICHLET_EPSILON,
    EXPLORATION,
    MCTS,
    Evaluator,
    drive,
)

# Plies played by sampling from the visit counts rather than taking the best move, without which
# every self-play game from a deterministic network is the same game. AlphaZero's number, which
# for tic-tac-toe covers the whole game: cutting it to three plies cost 10 points of agreement.
TEMPERATURE_MOVES = 30
TEMPERATURE = 1.0

# What the temperature drops to afterwards. Zero is greedy - more aggressive than AlphaZero, which
# keeps sampling at a low temperature. A game whose self-play needs late diversity can raise it.
FINAL_TEMPERATURE = 0.0

# Random plies played before a game starts being recorded, drawn uniformly from 0 to this.
#
# Off, because exploration belongs in PUCT and root noise: random starts here reached the same
# agreement as raising c_puct to 5.0 did, and needing them is a sign that one of those is mistuned.
# Kept for benchmarks, which want coverage of positions real play never reaches.
OPENING_PLIES = 0


class Example(NamedTuple):
    """One position, and what the search and the eventual result had to say about it."""

    planes: Any
    policy: Sequence[float]
    value: float


def game_steps(
    encoder: Encoder,
    game: Callable[[], GameState],
    simulations: int,
    temperature_moves: int = TEMPERATURE_MOVES,
    temperature: float = TEMPERATURE,
    final_temperature: float = FINAL_TEMPERATURE,
    opening_plies: int = 0,
    exploration: float = EXPLORATION,
    dirichlet_alpha: float = DIRICHLET_ALPHA,
    dirichlet_epsilon: float = DIRICHLET_EPSILON,
    rng: Optional[random.Random] = None,
) -> Iterator[GameState]:
    """
    A whole game as a generator: it yields every position it needs evaluated and returns
    `(examples, finished state)`.

    Suspendable for the same reason the search is, and `yield from` is all it takes to stay so.

    The finished state comes back so a caller can report on the games as well as learn from them.

    `opening_plies` puts the game somewhere random before any of it is recorded, so the randomness
    moves where games happen without changing what is learned from them.
    """
    rng = rng or random.Random()
    mcts = MCTS(
        _refuses, encoder, simulations=simulations, exploration=exploration,
        dirichlet_alpha=dirichlet_alpha, dirichlet_epsilon=dirichlet_epsilon, rng=rng,
    )

    state = _opening(game, opening_plies, rng)
    played: List[Tuple[Any, Sequence[float], bool]] = []

    while not state.is_game_over:
        result = yield from mcts.steps(state, noise=True)
        tau = temperature if len(played) < temperature_moves else final_temperature

        # Recorded before the move: the example belongs to the position it was searched from.
        played.append((encoder.planes(state), result.policy, state.turn))
        state.make_move(mcts.sample(result.visits, tau))

    outcome = state.result
    examples = [
        Example(planes, policy, _value_to(mover, outcome.winner))
        for planes, policy, mover in played
    ]
    return examples, state


def _refuses(state: GameState):
    """
    The evaluator an `MCTS` built for `game_steps` will never call.

    `MCTS.steps` hands every request to whoever drives it, which here is the batched caller. This
    raises so that a stray `search()` fails loudly rather than quietly using a different evaluator.
    """
    raise RuntimeError('this search is driven by its caller and has no evaluator of its own')


def play_game(
    evaluator: Evaluator,
    encoder: Encoder,
    game: Callable[[], GameState],
    simulations: int,
    **kwargs,
) -> Tuple[List[Example], GameState]:
    """One game, answering the search one position at a time. `play_games` is the batched form."""
    return drive(game_steps(encoder, game, simulations, **kwargs), evaluator)


def play_games(
    batch_evaluator: Callable[[Sequence[GameState]], Sequence[Tuple[Sequence[float], float]]],
    encoder: Encoder,
    game: Callable[[], GameState],
    count: int,
    simulations: int,
    batch_size: int = 1,
    seed: int = 0,
    on_finished: Optional[Callable[[int, int], None]] = None,
    **kwargs,
) -> List[Tuple[List[Example], GameState]]:
    """
    Plays `count` games, keeping up to `batch_size` of them in flight and evaluating the positions
    they are waiting on together.

    This is where most of a run's time goes and where it is won back: the search asks for one
    position per simulation, and a batch of sixty-four is ten times the throughput of a batch of
    one.

    **Each game keeps its own random stream**, seeded from `seed` and the game's index, so
    `batch_size` cannot change what is played. Games sharing a stream would draw from it in
    whatever order they happened to reach it.

    The evaluator is handed every pending position at once and must read what it needs from all of
    them before returning: the states are live and resuming a game mutates its own.

    `on_finished(completed, count)` is called as each game ends, which is the only sign of life a
    twenty-minute Connect 4 generation gives.
    """
    started, done = 0, {}
    active: List[Tuple[int, Iterator[GameState], GameState]] = []

    while started < count or active:
        while started < count and len(active) < batch_size:
            steps = game_steps(
                encoder, game, simulations, rng=random.Random(f'{seed}:{started}'), **kwargs)
            _advance(started, steps, None, active, done, on_finished, count)
            started += 1

        if not active:
            break

        # Every pending position, answered in one pass, before any of them is resumed.
        answers = batch_evaluator([pending for _, _, pending in active])
        waiting, active = active, []
        for (index, steps, _), answer in zip(waiting, answers):
            _advance(index, steps, answer, active, done, on_finished, count)

    # By game index rather than completion order, which is a function of `batch_size`.
    return [done[index] for index in range(count)]


def _advance(index, steps, answer, active, done, on_finished=None, count=0) -> None:
    """Pushes one game forward, onto `active` if it wants another position or `done` if finished."""
    try:
        active.append((index, steps, steps.send(answer)))
    except StopIteration as finished:
        done[index] = finished.value
        if on_finished:
            on_finished(len(done), count)


def _opening(game: Callable[[], GameState], plies: int, rng: random.Random) -> GameState:
    """
    A position a random number of random plies in, and still running.

    The count is drawn per game so openings land at every depth rather than piling up at one. A
    game that finishes inside its own opening is redrawn.
    """
    if plies <= 0:
        return game()

    for _ in range(50):
        state = game()
        for _ in range(rng.randint(0, plies)):
            if state.is_game_over:
                break
            state.make_move(rng.choice(list(state.legal_moves)))

        if not state.is_game_over:
            return state
    return game()


def _value_to(mover: bool, winner: Optional[bool]) -> float:
    """What a finished game was worth to the player who was about to move."""
    if winner is None:
        return 0.0
    return 1.0 if winner == mover else -1.0


def augment(examples: Sequence[Example], encoder: Encoder) -> List[Example]:
    """
    Every example under every symmetry the game admits.

    The value is carried through untouched: rotating a board does not change who is winning.
    """
    grown: List[Example] = []
    for example in examples:
        for planes, policy in encoder.symmetries(example.planes, list(example.policy)):
            grown.append(Example(planes, policy, example.value))
    return grown
