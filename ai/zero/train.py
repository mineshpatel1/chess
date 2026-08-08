"""
The loop: play games against yourself, learn from them, repeat.

Each generation plays `games` self-play games with the current network, adds the positions to a
replay buffer, and takes some gradient steps against a sample of it. Nothing else. There is no
opponent, no supervision and no opening book - the only external fact the process ever sees is
who won.

What makes it work is that the search is better than the network that guides it. MCTS with a
mediocre prior still finds the mate in one; the network trains on the search's visit counts and
so absorbs it; the improved network makes the next search better still. The improvement has to
come from somewhere, and it comes from lookahead being cheaper than knowledge.

The loss has two terms and they are weighted equally, as in the paper: cross-entropy between the
predicted policy and the search's visit counts, and squared error between the predicted value and
the game's eventual result. Both heads read the same trunk, so each is regularising the other.

Progress is measured against `ai.oracle`, not against the previous network. Tic-tac-toe is solved,
so "what fraction of all 4,520 decision positions does this network get right" is available every
generation for a fraction of a second - and unlike self-play win rates, it cannot be gamed by both
sides getting worse together. A run that is working shows it climbing; a run that has stalled says
so plainly.
"""

import random
import time
from collections import deque
from typing import Callable, Deque, List, NamedTuple, Optional, Type

import torch
import torch.nn.functional as F

import log
from games.base import GameState
from ai.oracle import benchmark, enumerate_positions
from ai.zero.checkpoint import save
from ai.zero.net import ZeroNet, evaluate, to_tensor
from ai.zero.mcts import DIRICHLET_EPSILON
from ai.zero.selfplay import (
    OPENING_PLIES, TEMPERATURE_MOVES, Example, augment, play_game,
)

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 128
STEPS_PER_GENERATION = 40

# Positions kept. Old generations are worth keeping - they were produced by a weaker network but
# the game results in them are just as true - but not forever, or the network spends its capacity
# imitating a version of itself it has outgrown.
BUFFER_SIZE = 20_000

# How often to grade the network against the oracle. Exact and useful, but it is instrumentation
# rather than training: measured at 80 games and 150 simulations it was 32% of a generation's wall
# clock, against 3.5% for the gradient steps. Every generation is right while tuning; less often is
# right when the answer is already known and the run just has to finish.
BENCHMARK_EVERY = 1

# How hard PUCT explores *while learning*, which is not the same question as how hard it should
# explore while playing. Learning wants the search to check alternatives it currently believes are
# worse, because that is where the training signal comes from; playing wants it to trust a prior
# that is now good. `ai.zero.mcts.EXPLORATION` stays at the lower value for play.
#
# Measured over 90 generations at 50 simulations, on-policy:
#
#     c_puct   1.5      3.0      5.0      8.0
#     agree   92.83%   95.38%   95.84%   94.65%
#
# An inverted U with a clear peak, and the default was sitting well below it - visit counts were
# concentrating before the alternatives had been checked, so the network fit targets that were
# confidently slightly wrong.
SELF_PLAY_EXPLORATION = 5.0

# Whether to add every symmetric copy of each example, which for tic-tac-toe is eight for one.
#
# **Off**, on both an argument and a measurement, and it was on by default here for a while
# without either being made.
#
# The argument is that it is hand-injected domain knowledge: the network is *told* the board has
# eight symmetries rather than left to notice, which cuts against learning the game from self-play
# alone. The paper history runs the same way - AlphaGo Zero used eight-fold dihedral augmentation
# and AlphaZero dropped it, chess and shogi not being symmetric.
#
# The measurement says it buys nothing anyway. Over 150 generations at 50 simulations, on-policy:
#
#     generation      50       100      150
#     with          93.05%   95.84%   96.31%
#     without       93.94%   96.22%   96.46%
#
# Without leads at every checkpoint while seeing eight times fewer examples. The final margin is
# inside single-seed noise, but there is no case for paying capacity for it: this network is an
# MLP over a flattened board with no built-in equivariance, so all eight copies buy is that it
# spends parameters learning the symmetry group instead of learning positions.
SYMMETRIES = False

GENERATIONS = 30
GAMES_PER_GENERATION = 40
SIMULATIONS = 50


class Progress(NamedTuple):
    """One generation's report."""

    generation: int
    examples: int
    loss: float
    policy_loss: float
    value_loss: float
    draw_rate: float
    optimal_rate: float
    seconds: float

    def __str__(self) -> str:
        return (
            f'gen {self.generation:>3}  loss {self.loss:6.4f} '
            f'(policy {self.policy_loss:6.4f}, value {self.value_loss:6.4f})  '
            f'self-play drawn {self.draw_rate:5.1%}  '
            f'vs perfect play {self.optimal_rate:6.2%}  '
            f'{self.seconds:5.1f}s'
        )


def train(
    game: Type[GameState],
    generations: int = GENERATIONS,
    games_per_generation: int = GAMES_PER_GENERATION,
    simulations: int = SIMULATIONS,
    steps: int = STEPS_PER_GENERATION,
    batch_size: int = BATCH_SIZE,
    opening_plies: int = OPENING_PLIES,
    temperature_moves: int = TEMPERATURE_MOVES,
    exploration: float = SELF_PLAY_EXPLORATION,
    dirichlet_epsilon: float = DIRICHLET_EPSILON,
    learning_rate: float = LEARNING_RATE,
    benchmark_every: int = BENCHMARK_EVERY,
    symmetries: bool = SYMMETRIES,
    checkpoint_path: Optional[str] = None,
    seed: int = 0,
    on_generation: Optional[Callable[[Progress], None]] = None,
) -> ZeroNet:
    """
    Trains a network from nothing, returning the best one it saw.

    "Best" is by the oracle benchmark rather than by the last generation, because self-play is
    noisy and the final network is not reliably the strongest one. Since the benchmark is exact
    and cheap here there is no reason to guess.
    """
    encoder = game.ENCODER
    if encoder is None:
        raise ValueError(f'{game.__name__} has no ENCODER, so it cannot be learned')

    rng = random.Random(seed)
    torch.manual_seed(seed)

    net = ZeroNet(encoder.PLANE_SHAPE, encoder.POLICY_SIZE)
    optimiser = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    buffer: Deque[Example] = deque(maxlen=BUFFER_SIZE)

    best_state, best_rate, last_rate = None, -1.0, 0.0

    for generation in range(1, generations + 1):
        started = time.perf_counter()

        fresh, drawn = _self_play(
            net, encoder, game, games_per_generation, simulations, opening_plies,
            temperature_moves, exploration, dirichlet_epsilon, rng)
        buffer.extend(augment(fresh, encoder) if symmetries else fresh)

        loss, policy_loss, value_loss = _learn(net, optimiser, buffer, steps, batch_size, rng)

        graded = generation % max(benchmark_every, 1) == 0 or generation == generations
        rate = _optimal_rate(net, encoder, game) if graded else last_rate
        last_rate = rate

        progress = Progress(
            generation=generation,
            examples=len(buffer),
            loss=loss,
            policy_loss=policy_loss,
            value_loss=value_loss,
            draw_rate=drawn / max(games_per_generation, 1),
            optimal_rate=rate,
            seconds=time.perf_counter() - started,
        )
        log.info(str(progress))
        if on_generation:
            on_generation(progress)

        if graded and rate > best_rate:
            best_rate = rate
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            if checkpoint_path:
                save(net, checkpoint_path, game=game.__name__, generation=generation,
                     metadata={'optimal_rate': rate, 'simulations': simulations})

    if best_state is not None:
        net.load_state_dict(best_state)
    log.info(f'best raw-policy agreement with perfect play: {best_rate:.2%}')
    return net


def _self_play(net, encoder, game, count, simulations, opening_plies,
               temperature_moves, exploration, dirichlet_epsilon, rng):
    """One generation's games, and how many of them were drawn."""
    def evaluator(state):
        return evaluate(net, state, encoder)

    examples: List[Example] = []
    drawn = 0
    for _ in range(count):
        played, finished = play_game(
            evaluator, encoder, game, simulations, opening_plies=opening_plies,
            temperature_moves=temperature_moves, exploration=exploration,
            dirichlet_epsilon=dirichlet_epsilon, rng=rng)
        examples.extend(played)
        drawn += int(finished.result.winner is None)
    return examples, drawn


def _learn(net, optimiser, buffer, steps, batch_size, rng):
    """
    Gradient steps against a sample of the buffer.

    The policy term is cross-entropy against a *distribution*, not a label: the search's visit
    counts say a position has two equally good moves, and a network told to pick one of them is
    being taught something false. `-(target * log_softmax(logits)).sum()` is that, and it reduces
    to ordinary cross-entropy when the target happens to be one-hot.
    """
    if not buffer:
        return 0.0, 0.0, 0.0

    net.train()
    totals = [0.0, 0.0, 0.0]

    for _ in range(steps):
        batch = rng.sample(list(buffer), k=min(batch_size, len(buffer)))
        planes = to_tensor([example.planes for example in batch])
        target_policy = torch.tensor([list(e.policy) for e in batch], dtype=torch.float32)
        target_value = torch.tensor([e.value for e in batch], dtype=torch.float32)

        logits, value = net(planes)
        policy_loss = -(target_policy * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        value_loss = F.mse_loss(value, target_value)
        loss = policy_loss + value_loss

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        totals[0] += loss.item()
        totals[1] += policy_loss.item()
        totals[2] += value_loss.item()

    return tuple(total / steps for total in totals)


def _optimal_rate(net, encoder, game) -> float:
    """
    How often the raw policy picks a best move, over every position in the game.

    The network alone, with no search - which is the honest measure of what has actually been
    learned, and is one forward pass per position, so it costs a fraction of a second.
    """
    def raw(state):
        priors, _ = evaluate(net, state, encoder)
        return max(state.legal_moves, key=lambda move: priors[encoder.action_index(move)])

    return benchmark(raw, enumerate_positions(game)).overall.rate
