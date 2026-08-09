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

import math
import random
import time
from collections import deque
from typing import Callable, Deque, List, NamedTuple, Optional, Tuple, Type

import torch
import torch.nn.functional as F

import log
from games.base import GameState
from ai import corpus
from ai.oracle import Grade, Report, Table, benchmark, enumerate_positions, move_values
from ai.zero.checkpoint import save
from ai.zero.metrics import Recorder
from ai.zero.net import ZeroNet, evaluate, evaluate_batch, for_game as architecture, to_tensor
from ai.zero.mcts import DIRICHLET_EPSILON
from ai.zero.selfplay import (
    OPENING_PLIES, TEMPERATURE_MOVES, Example, augment, play_games,
)

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 128
STEPS_PER_GENERATION = 40

# Positions kept. Old generations are worth keeping - they were produced by a weaker network but
# the game results in them are just as true - but not forever, or the network spends its capacity
# imitating a version of itself it has outgrown.
BUFFER_SIZE = 20_000

# Self-play games advanced together, so their pending positions go through the network in one
# pass. Purely a throughput setting: every game gets the same tree it would have got alone, and
# `tests/zero/test_selfplay.py` asserts that a batch of sixty-four plays what a batch of one plays.
# One runs exactly the position-at-a-time path that existed before batching.
GAMES_IN_FLIGHT = 32

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
#
# **Tuned on tic-tac-toe, and it does not obviously transfer.** PUCT is
# `Q + c * P * sqrt(N_parent) / (1 + n)`, so with 50 simulations spread over Connect 4's seven
# columns each child gets about seven visits and the exploration term swamps Q throughout - the
# search never commits. Measured on a 30-generation Connect 4 run, the visit distributions it
# produced sat at 82% of the entropy of a uniform distribution, which is a search that has
# concluded almost nothing, and the network learned almost nothing from them: 74.6% agreement
# against the 73.5% scored by "always play nearest the centre".
#
# Tic-tac-toe gets away with a high value because it is nine plies deep and terminal values reach
# the root almost immediately, so Q is a strong signal from the first visits.
#
# **And yet changing it does nothing here, which was worth finding out.** A grid of twelve-
# generation Connect 4 runs, one seed each, metrics under `runs/grid/`:
#
#     c_puct  sims   target entropy   agreement   value mse   game plies
#        1.5    50      64% of unif       69.0%       0.759         18.6
#        2.5    50      74%               70.7%       0.812         17.4
#        5.0    50      87%               69.9%       0.704         17.1
#        1.5   200      55%               69.9%       0.789         20.2
#        2.5   200      63%               69.1%       0.864         20.3
#        2.0   600      47%               72.9%       0.683         24.2
#
# c_puct moves the entropy of the search's targets across a wide range and moves nothing else:
# agreement sits in a 1.7-point band that a single seed cannot separate. Near-uniform targets were
# a symptom, not the cause. Simulations do matter, but not between 50 and 200 - only at 600, which
# is the `REFERENCE_CONNECT4` setting and best on all four columns.
#
# It is still not worth paying for. Thirty generations at 50 simulations reached 74.6% in 10.7
# minutes; twelve at 600 reached 72.9% in 62 minutes. Cheap search bought a better number in a
# sixth of the time, so generations are the thing to buy on a CPU. A GPU amortising deep searches
# over 5,000 games an iteration is a different trade, which is why the reference makes the other
# choice.
SELF_PLAY_EXPLORATION = 5.0

# What a published Connect 4 AlphaZero uses, for reference rather than as configuration.
#
# AlphaZero.jl's Connect 4 example: 128 filters over 5 residual blocks with 32-filter heads (about
# 1.6M parameters), **600 simulations**, **c_puct 2.0**, Dirichlet (0.25, 1.0), temperature 1.0 for
# 20 moves then 0.3, Adam at 2e-3, batch 1,024, and 5,000 self-play games per iteration over 15
# iterations - one to two hours per iteration on an RTX 2070.
#
# Worth writing down because almost every number here differs from ours, and the two that differ
# most are exactly the two the entropy measurement points at. It is not a target: 5,000 games an
# iteration on a GPU is a different budget from 64 games on four CPU cores, and copying the whole
# configuration would be reasoning by analogy rather than about the problem. It is a sanity check
# on which direction the defaults should move.
REFERENCE_CONNECT4 = {
    'filters': 128, 'blocks': 5, 'head_filters': 32,
    'simulations': 600, 'exploration': 2.0,
    'temperature_moves': 20, 'final_temperature': 0.3,
    'learning_rate': 2e-3, 'batch_size': 1024, 'games_per_generation': 5000,
}

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

# Which corpus tier a game without an enumerable state space is graded on. See `grading_set`.
GRADING_TIER = 'E'

# Positions per forward pass when grading. Large enough that per-call overhead is amortised away,
# small enough that a five-block tower's activations for the batch still fit comfortably.
GRADING_CHUNK = 2048

GENERATIONS = 30
GAMES_PER_GENERATION = 40
SIMULATIONS = 50

# What a generation reports before it has been graded, so the first ungraded generations have
# something to carry rather than a special case at every use.
_EMPTY_REPORT = Report(overall=Grade(0, 0, 0, 0), by_seat={}, by_ply={}, worst=[], value_error=0.0)


def _entropy(examples) -> float:
    """
    Mean entropy of the search's policy targets, in nats.

    The number that tells a stalled run apart from a finished one. Policy loss cannot fall below
    the entropy of the targets it is fitting, so a loss that has flattened *at* this value is a
    network fitting its targets perfectly - and the fault is then in the search producing them,
    not in the network. Tic-tac-toe's c_puct being too low looked exactly like that: confident,
    slightly wrong targets, learned faithfully.
    """
    if not examples:
        return 0.0

    total = 0.0
    for example in examples:
        total -= sum(p * math.log(p) for p in example.policy if p > 0.0)
    return total / len(examples)


class Progress(NamedTuple):
    """One generation's report. `ai.zero.metrics` says why each field is here."""

    generation: int
    examples: int
    loss: float
    policy_loss: float
    value_loss: float
    draw_rate: float
    optimal_rate: float
    seconds: float
    value_mse: float = 0.0
    first_rate: float = 0.0
    second_rate: float = 0.0
    target_entropy: float = 0.0
    distinct_positions: int = 0
    game_length: float = 0.0
    self_play_seconds: float = 0.0
    learn_seconds: float = 0.0
    grade_seconds: float = 0.0

    def __str__(self) -> str:
        return (
            f'gen {self.generation:>3}  loss {self.loss:6.4f} '
            f'(policy {self.policy_loss:6.4f}, value {self.value_loss:6.4f})  '
            f'self-play drawn {self.draw_rate:5.1%}  '
            f'vs perfect play {self.optimal_rate:6.2%}  '
            f'value mse {self.value_mse:5.3f}  '
            f'{self.seconds:5.1f}s'
        )


def train(
    game: Type[GameState],
    generations: int = GENERATIONS,
    games_per_generation: int = GAMES_PER_GENERATION,
    simulations: int = SIMULATIONS,
    steps: int = STEPS_PER_GENERATION,
    batch_size: int = BATCH_SIZE,
    games_in_flight: int = GAMES_IN_FLIGHT,
    opening_plies: int = OPENING_PLIES,
    temperature_moves: int = TEMPERATURE_MOVES,
    exploration: float = SELF_PLAY_EXPLORATION,
    dirichlet_epsilon: float = DIRICHLET_EPSILON,
    learning_rate: float = LEARNING_RATE,
    benchmark_every: int = BENCHMARK_EVERY,
    symmetries: bool = SYMMETRIES,
    checkpoint_path: Optional[str] = None,
    metrics_path: Optional[str] = None,
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

    net = ZeroNet(encoder.PLANE_SHAPE, encoder.POLICY_SIZE, **architecture(game.__name__))
    positions, values = grading_set(game)
    recorder = Recorder(metrics_path)
    optimiser = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    buffer: Deque[Example] = deque(maxlen=BUFFER_SIZE)

    best_state, best_rate = None, -1.0
    last_report = _EMPTY_REPORT

    for generation in range(1, generations + 1):
        started = time.perf_counter()

        play_started = time.perf_counter()
        fresh, drawn, lengths = _self_play(
            net, encoder, game, games_per_generation, simulations, opening_plies,
            temperature_moves, exploration, dirichlet_epsilon, batch_size=games_in_flight,
            seed=f'{seed}:{generation}')
        play_seconds = time.perf_counter() - play_started
        buffer.extend(augment(fresh, encoder) if symmetries else fresh)

        learn_started = time.perf_counter()
        loss, policy_loss, value_loss = _learn(net, optimiser, buffer, steps, batch_size, rng)
        learn_seconds = time.perf_counter() - learn_started

        graded = generation % max(benchmark_every, 1) == 0 or generation == generations
        grade_started = time.perf_counter()
        report = _optimal_rate(net, encoder, positions, values) if graded else last_report
        grade_seconds = time.perf_counter() - grade_started
        last_report = report

        progress = Progress(
            generation=generation,
            examples=len(buffer),
            loss=loss,
            policy_loss=policy_loss,
            value_loss=value_loss,
            draw_rate=drawn / max(games_per_generation, 1),
            optimal_rate=report.overall.rate,
            seconds=time.perf_counter() - started,
            value_mse=report.value_error or 0.0,
            first_rate=report.by_seat[True].rate if True in report.by_seat else 0.0,
            second_rate=report.by_seat[False].rate if False in report.by_seat else 0.0,
            target_entropy=_entropy(fresh),
            distinct_positions=len({str(example.planes) for example in buffer}),
            game_length=sum(lengths) / max(len(lengths), 1),
            self_play_seconds=play_seconds,
            learn_seconds=learn_seconds,
            grade_seconds=grade_seconds,
        )
        log.info(str(progress))
        recorder.write(progress._asdict())
        if on_generation:
            on_generation(progress)

        rate = report.overall.rate
        if graded and rate > best_rate:
            best_rate = rate
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            if checkpoint_path:
                save(net, checkpoint_path, game=game.__name__, generation=generation,
                     metadata={'optimal_rate': rate, 'simulations': simulations})

    recorder.close()
    if best_state is not None:
        net.load_state_dict(best_state)
    log.info(f'best raw-policy agreement with perfect play: {best_rate:.2%}')
    return net


def _self_play(net, encoder, game, count, simulations, opening_plies,
               temperature_moves, exploration, dirichlet_epsilon, batch_size, seed):
    """
    One generation's games, played concurrently and evaluated in batches.

    `batch_size` games are in flight at once and the positions they are waiting on go through the
    network together. It changes nothing about any individual game - each runs an ordinary
    sequential search and gets the tree it would have got alone - and it is most of the difference
    between a Connect 4 generation costing seven minutes and costing one.
    """
    def batch_evaluator(states):
        return evaluate_batch(net, states, encoder)

    played = play_games(
        batch_evaluator, encoder, game, count, simulations,
        batch_size=batch_size, seed=seed, opening_plies=opening_plies,
        temperature_moves=temperature_moves, exploration=exploration,
        dirichlet_epsilon=dirichlet_epsilon)

    examples: List[Example] = []
    drawn, lengths = 0, []
    for examples_from_game, finished in played:
        examples.extend(examples_from_game)
        drawn += int(finished.result.winner is None)
        lengths.append(len(examples_from_game))
    return examples, drawn, lengths


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


def grading_set(game):
    """
    The positions a game is graded over each generation, and how to value them.

    Built once and reused, because the alternative for Connect 4 is re-reading a 33,300-line file
    and rebuilding 22,100 boards every generation.

    Dispatches the way `zero.py grade` does. A game that declares `SOLVED_DEPTH` can be searched to
    the end of itself, so its whole state space is walkable and solvable on the spot. Anything
    larger needs answers computed in advance - Connect 4's tree is 4.5e12 positions and
    `enumerate_positions` would never return.

    Connect 4 is graded on the **enumerated opening tier alone**: every distinct position out to
    six discs, 22,100 of them. Exhaustive rather than sampled, so the number carries no sampling
    noise - which matters when it is choosing which checkpoint to keep, generation after
    generation. It is also the region with the most room to show improvement, `minimax:4` scoring
    79.1% there against 95.5% on deep random positions.
    """
    if game.SOLVED_DEPTH is not None:
        # `.copy()` is not optional. `enumerate_positions` yields the *live* state as it walks the
        # tree, so keeping the references without copying leaves every entry pointing at one
        # object in whatever position the walk finished in - a grading set of 5,478 empty boards,
        # which scores 100% and takes eighty seconds to say so.
        positions = [(state.copy(), ply) for state, ply in enumerate_positions(game)]

        # Solved once here rather than on demand every generation. The answers are properties of
        # the game and cannot change, so re-deriving them each time is the same work repeated for
        # the length of the run - and a shared table makes the one pass nearly free.
        table = Table()
        solved = {
            state.solver_key: move_values(state, table)
            for state, _ in positions if not state.is_game_over
        }
        return positions, lambda state: solved[state.solver_key]

    path = corpus.CORPORA.get(game.__name__)
    if path is None:
        raise ValueError(
            f'{game.__name__} can neither be enumerated nor has a solved corpus, so a training '
            f'run has nothing to grade itself against'
        )

    entries = corpus.load(path, tiers=(GRADING_TIER,))
    return list(corpus.positions(entries, game)), corpus.values(entries)


def _optimal_rate(net, encoder, positions, values):
    """
    How often the raw policy picks a best move, and how wrong the value head is.

    The network alone, with no search - which is the honest measure of what has actually been
    learned. Search papers over a weak prior, so grading with it on would flatter a network that
    is not improving.

    **Every position in one forward pass.** Asked one at a time this is 22,100 batch-1 calls for
    Connect 4, about 37 seconds, or roughly a third of a generation. There is no tree here, just
    independent positions, so batching cannot change an answer - it is the one place in the whole
    system where speed is free.
    """
    states = [state for state, _ in positions]

    # In chunks rather than one pass over all 22,100. A single batch that large allocates
    # activations for every position at once - for a five-block tower that is gigabytes - and
    # spends longer in memory traffic than it saves in dispatch. A couple of thousand is well past
    # the point where per-call overhead stops mattering.
    answers = []
    for start in range(0, len(states), GRADING_CHUNK):
        answers.extend(evaluate_batch(net, states[start:start + GRADING_CHUNK], encoder))

    priors = {id(state): answer[0] for state, answer in zip(states, answers)}
    heads = {id(state): answer[1] for state, answer in zip(states, answers)}

    def raw(state):
        prior = priors[id(state)]
        return max(state.legal_moves, key=lambda move: prior[encoder.action_index(move)])

    return benchmark(raw, positions, values=values, value_fn=lambda state: heads[id(state)])
