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

Progress is measured against `ai.oracle` and `ai.ladder` rather than against the previous network,
because a self-play win rate can be gamed by both sides getting worse together.
"""

import copy
import math
import random
import time
from collections import deque
from typing import Callable, Deque, Dict, List, NamedTuple, Optional, Sequence, Tuple, Type

import torch
import torch.nn.functional as F

import log
from games.base import GameState
from ai import corpus
from ai.oracle import Grade, Report, Table, benchmark, enumerate_positions, move_values
from ai.zero import checkpoint, replay
from ai.zero.checkpoint import save
from ai.zero.metrics import Recorder, truncate_after
from ai.zero.net import (
    ZeroNet, choose_device, device_of, evaluate, evaluate_batch, flush_denormals,
    for_game as architecture, make_deterministic, to_tensor,
)
from ai.zero.mcts import DIRICHLET_EPSILON
from ai.zero.selfplay import (
    FINAL_TEMPERATURE, OPENING_PLIES, TEMPERATURE_MOVES, Example, augment, play_games,
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
# pass. Purely a throughput setting: every game gets the same tree it would have got alone, which
# `tests/zero/test_selfplay.py` asserts.
GAMES_IN_FLIGHT = 32

# How often to grade the network against the oracle. Instrumentation rather than training, and not
# free: at 80 games and 150 simulations it is 32% of a generation against 3.5% for the gradient
# steps. Every generation while tuning, less often for a run that just has to finish.
BENCHMARK_EVERY = 1

# Games between progress reports during self-play, where a generation spends its time. A rate
# rather than a bare count, since that is what tells a slow generation from a hung one.
REPORT_EVERY = 25

# How hard PUCT explores *while learning*, which is a different question from how hard to explore
# while playing: learning wants alternatives checked, because that is where the training signal
# comes from. `ai.zero.mcts.EXPLORATION` stays lower for play.
#
# Tuned on tic-tac-toe, where the curve is an inverted U with a clear peak. Whether it transfers to
# Connect 4 could not be measured at the budget available - see that game's README.
SELF_PLAY_EXPLORATION = 5.0

# AlphaZero.jl's Connect 4 example, for reference rather than as configuration: 5,000 games an
# iteration on a GPU is a different budget from ours, so this is a sanity check on which direction
# the defaults should move rather than a target.
REFERENCE_CONNECT4 = {
    'filters': 128, 'blocks': 5, 'head_filters': 32,
    'simulations': 600, 'exploration': 2.0,
    'temperature_moves': 20, 'final_temperature': 0.3,
    'learning_rate': 2e-3, 'batch_size': 1024, 'games_per_generation': 5000,
}

# Whether to add every symmetric copy of each example, which for tic-tac-toe is eight for one.
#
# Off. It is hand-injected domain knowledge - the network is told the board is symmetric rather
# than left to notice - and it buys nothing measurable, while a dense trunk with no built-in
# equivariance spends parameters learning the symmetry group instead of learning positions.
SYMMETRIES = False

# The opponents a generation is scored against, and how many games each. `None` means the game's
# own ladder from `ai.ladder.LADDERS`; a caller steering a run usually wants fewer, since rungs a
# player has saturated or cannot touch cost the same to play and move the mean by nothing.
LADDER_RUNGS = None
LADDER_GAMES = 100

# Simulations the challenger searches with when it climbs. Zero is the raw policy: about 15s a
# rung, and a direction indicator only - it understates the player badly, so a final claim needs
# `zero.py ladder` with simulations, at roughly 7 minutes a rung.
LADDER_SIMULATIONS = 0

# How often to play them. Every generation is under 2% of a Connect 4 generation; a game whose
# generations are seconds wants this far less often.
LADDER_EVERY = 1

# Which measure picks the best checkpoint, per game: whichever one still *moves* where the player
# actually is. Tic-tac-toe saturates the ladder, reaching perfect play while agreement is still
# resolving real differences. Connect 4's agreement saturates in usefulness instead - graded on the
# opening tier it asks about a sixth of a game, and two networks half a point apart on it scored
# 0.055 and 0.635 against `minimax:4`.
#
# A game with no ladder, or a run with the ladder off, falls back to agreement.
SELECTION_METRIC = {
    'TicTacToe': 'agreement',
    'Connect4': 'ladder',
}
SELECTION_DEFAULT = 'ladder'


def metric_for(game_name: str) -> str:
    """Which measure `best` means for a game. See `SELECTION_METRIC`."""
    return SELECTION_METRIC.get(game_name, SELECTION_DEFAULT)

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

    Tells a stalled run from a finished one. Policy loss cannot fall below the entropy of the
    targets it fits, so a loss flattened *at* this value is a network fitting its targets
    perfectly - and the fault is then in the search producing them.
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
    denormal_weights: int = 0

    # The primary metric: mean score against `LADDER_RUNGS`, and the strongest rung beaten
    # significantly rather than merely led. `tier_rates` carries agreement per corpus tier.
    ladder_score: float = 0.0
    highest_rung: str = ''
    tier_rates: Optional[Dict[str, float]] = None
    ladder_seconds: float = 0.0
    self_play_seconds: float = 0.0
    learn_seconds: float = 0.0
    grade_seconds: float = 0.0

    def __str__(self) -> str:
        """The ladder first, that being the number the run is steered by."""
        beaten = f' (beats {self.highest_rung})' if self.highest_rung else ''
        return (
            f'gen {self.generation:>3}  ladder {self.ladder_score:5.3f}{beaten}  '
            f'agreement {self.optimal_rate:6.2%}  '
            f'value mse {self.value_mse:5.3f}  '
            f'loss {self.loss:6.4f} '
            f'(policy {self.policy_loss:6.4f}, value {self.value_loss:6.4f})  '
            f'self-play drawn {self.draw_rate:5.1%}  '
            f'{self.seconds:5.1f}s'
        )


def train(
    game: Type[GameState],
    generations: int = GENERATIONS,
    games_per_generation: int = GAMES_PER_GENERATION,
    simulations: int = SIMULATIONS,
    steps: int = STEPS_PER_GENERATION,
    batch_size: int = BATCH_SIZE,
    buffer_size: int = BUFFER_SIZE,
    games_in_flight: int = GAMES_IN_FLIGHT,
    opening_plies: int = OPENING_PLIES,
    temperature_moves: int = TEMPERATURE_MOVES,
    final_temperature: float = FINAL_TEMPERATURE,
    exploration: float = SELF_PLAY_EXPLORATION,
    dirichlet_epsilon: float = DIRICHLET_EPSILON,
    learning_rate: float = LEARNING_RATE,
    benchmark_every: int = BENCHMARK_EVERY,
    ladder_every: int = LADDER_EVERY,
    ladder_rungs: Sequence[str] = LADDER_RUNGS,
    ladder_games: int = LADDER_GAMES,
    ladder_simulations: int = LADDER_SIMULATIONS,
    metric: Optional[str] = None,
    symmetries: bool = SYMMETRIES,
    checkpoint_path: Optional[str] = None,
    latest_path: Optional[str] = None,
    metrics_path: Optional[str] = None,
    resume_from: Optional[str] = None,
    seed: int = 0,
    device: Optional[str] = None,
    on_generation: Optional[Callable[[Progress], None]] = None,
) -> ZeroNet:
    """
    Trains a network from nothing, returning the best one it saw.

    "Best" is by `measure` rather than by the last generation, because self-play is noisy and the
    final network is not reliably the strongest one.
    """
    encoder = game.ENCODER
    if encoder is None:
        raise ValueError(f'{game.__name__} has no ENCODER, so it cannot be learned')

    rng = random.Random(seed)
    torch.manual_seed(seed)

    device = choose_device(device)
    make_deterministic(device)

    net = ZeroNet(encoder.PLANE_SHAPE, encoder.POLICY_SIZE, **architecture(game.__name__))
    net.to(device)
    log.info(f'training on {device}')
    sets = grading_sets(game)
    optimiser = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    buffer: Deque[Example] = deque(maxlen=buffer_size)

    best_state, best_rate = None, -1.0
    first = 1

    # Which measure `best` means, recorded into every checkpoint so a resume can refuse to compare
    # against a bar set on a different one. Per game unless the caller says otherwise.
    measure = metric or metric_for(game.__name__)
    if measure not in ('ladder', 'agreement'):
        raise ValueError(f'unknown selection metric {measure!r}; use "ladder" or "agreement"')
    if measure == 'ladder' and ladder_every <= 0:
        log.warning('selection asked for the ladder but it is switched off; using agreement')
        measure = 'agreement'

    # Where the buffer is written and read, which is beside the checkpoint a resume reads rather
    # than inside it - see `ai/zero/replay.py`. No `--latest` means no resume point at all, and a
    # run with nothing to resume from has nothing to keep a buffer for either.
    buffer_path = replay.path_for(latest_path) if latest_path else None

    if resume_from:
        blob = checkpoint.load(resume_from, game=game.__name__)
        net.load_state_dict(blob['weights'])
        net.to(device)
        if blob.get('optimiser'):
            optimiser.load_state_dict(blob['optimiser'])
        first = blob['generation'] + 1
        best_state = {k: v.clone() for k, v in net.state_dict().items()}

        # Only when the bar was set on the same measure. A ladder score and an agreement rate are
        # both numbers between 0 and 1 and are not the same quantity, so carrying one over as the
        # other sets a bar no checkpoint clears and `best` then silently never updates again.
        stored = blob['metadata'].get('metric')
        if stored == _measure_identity(measure, game, ladder_rungs):
            best_rate = blob['metadata'].get('best_rate', -1.0)
            log.info(f'Resuming {resume_from} from generation {first}, '
                     f'best {measure} so far {best_rate:.3f}')
        else:
            log.info(f'Resuming {resume_from} from generation {first}; it was scored on '
                     f'{stored or "an older metric"} and this run scores on {measure}, '
                     f'so the best-so-far bar starts again')

        # A generation is recorded before its weights are written, so the metrics file can be one
        # generation ahead of the checkpoint. See `truncate_after`.
        if metrics_path:
            dropped = truncate_after(metrics_path, blob['generation'])
            if dropped:
                log.info(f'dropped {dropped} recorded generation(s) past the checkpoint')

        # Into the deque rather than around it, so a run that lowered `--buffer-size` keeps the
        # newest positions and drops the rest.
        restored = replay.load(replay.path_for(resume_from), game.__name__, encoder)
        if restored:
            buffer.extend(restored)
            log.info(f'restored {len(buffer):,} positions to the replay buffer')
        else:
            log.info('no replay buffer beside the checkpoint; the first generation or two will '
                     'learn from less data than usual and score lower for it')

    recorder = Recorder(metrics_path, append=bool(resume_from))
    last_reports = {tier: _EMPTY_REPORT for tier in sets}
    last_standing = None

    # Which tier's agreement travels as `optimal_rate`: the opening for a corpus-graded game, the
    # whole space for one small enough to enumerate.
    headline = 'E' if 'E' in sets else next(iter(sets))

    for generation in range(first, generations + 1):
        started = time.perf_counter()

        play_started = time.perf_counter()
        fresh, drawn, lengths = _self_play(
            net, encoder, game, games_per_generation, simulations, opening_plies,
            temperature_moves, final_temperature, exploration, dirichlet_epsilon,
            batch_size=games_in_flight, seed=f'{seed}:{generation}')
        play_seconds = time.perf_counter() - play_started
        buffer.extend(augment(fresh, encoder) if symmetries else fresh)

        learn_started = time.perf_counter()
        loss, policy_loss, value_loss = _learn(net, optimiser, buffer, steps, batch_size, rng)

        # Immediately after the steps that create them, so the next generation's self-play,
        # gradient steps and benchmark all run on normal floats. See `flush_denormals`.
        denormals = flush_denormals(net)
        learn_seconds = time.perf_counter() - learn_started

        graded = generation % max(benchmark_every, 1) == 0 or generation == generations
        grade_started = time.perf_counter()
        reports = _grade(net, encoder, sets) if graded else last_reports
        grade_seconds = time.perf_counter() - grade_started
        last_reports = reports
        report = reports[headline]

        climbed = ladder_every > 0 and (generation % ladder_every == 0
                                        or generation == generations)
        ladder_started = time.perf_counter()
        standing = (_climb(net, encoder, game, ladder_rungs, ladder_games, seed, ladder_simulations)
                    if climbed else None)
        ladder_seconds = time.perf_counter() - ladder_started
        if standing is not None:
            last_standing = standing
        standing = last_standing

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
            denormal_weights=denormals,
            ladder_score=_mean_score(standing),
            highest_rung=(standing.highest_beaten or '') if standing else '',
            tier_rates={tier: report.overall.rate for tier, report in reports.items()},
            ladder_seconds=ladder_seconds,
            self_play_seconds=play_seconds,
            learn_seconds=learn_seconds,
            grade_seconds=grade_seconds,
        )
        log.info(str(progress))
        recorder.write(progress._asdict())

        # Computed once so the checkpoints written below and the comparison made afterwards cannot
        # disagree about what `best` means.
        rate = progress.ladder_score if measure == 'ladder' else report.overall.rate

        # Written every generation whether or not it improved, because this is the file a resume
        # reads. Resuming from `checkpoint_path` instead would replay every generation since the
        # best network was set.
        if latest_path:
            save(net, latest_path, game=game.__name__, generation=generation,
                 metadata={'best_rate': max(best_rate, rate),
                           'metric': _measure_identity(measure, game, ladder_rungs),
                           'ladder_score': progress.ladder_score,
                           'optimal_rate': report.overall.rate,
                           'simulations': simulations},
                 optimiser=optimiser)

        # After the checkpoint rather than before, so an interruption between the two leaves the
        # buffer a generation *behind* the weights. A buffer a generation ahead would hold the
        # replayed generation's games twice.
        if buffer_path:
            replay.save(buffer, buffer_path, game=game.__name__, generation=generation)

        # After the checkpoint, since `--commit-every` copies the pair somewhere a resume will read
        # it: called first, the hook takes generation N's metrics with generation N-1's weights.
        if on_generation:
            on_generation(progress)

        if (graded or climbed) and rate > best_rate:
            best_rate = rate
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            if checkpoint_path:
                save(net, checkpoint_path, game=game.__name__, generation=generation,
                     metadata={'ladder_score': progress.ladder_score,
                               'optimal_rate': report.overall.rate,
                               'chosen_on': _measure_identity(measure, game, ladder_rungs),
                               'simulations': simulations},
                     optimiser=optimiser)

    recorder.close()
    if best_state is not None:
        net.load_state_dict(best_state)

    # The measure is named rather than assumed, the two numbers being easy to mistake.
    if measure == 'ladder' and last_standing is not None:
        opponents = ', '.join(rung.spec for rung in last_standing.rungs)
        log.info(f'best ladder score against {opponents}: {best_rate:.3f}')
    else:
        log.info(f'best raw-policy agreement with perfect play: {best_rate:.2%}')
    return net


def _measure_identity(measure: str, game, rungs) -> str:
    """
    The measure `best` is chosen on, including which opponents when it is the ladder.

    The rungs are part of the measure's identity: a bar of 0.740 set against depths 4, 5 and 6 is
    unreachable for the same network measured against 7 and 8, so a resume comparing "ladder" to
    "ladder" would freeze the best checkpoint for the rest of the run.
    """
    if measure != 'ladder':
        return measure

    from ai import ladder as ladders  # Local, as in `_climb`

    used = tuple(rungs) if rungs else ladders.for_game(game).rungs
    return 'ladder:' + ','.join(used)


def _mean_score(standing) -> float:
    """The ladder as one number: the mean score across its rungs, or 0.0 if it was not played."""
    if standing is None or not standing.rungs:
        return 0.0
    return sum(rung.result.score for rung in standing.rungs) / len(standing.rungs)


def _self_play(net, encoder, game, count, simulations, opening_plies,
               temperature_moves, final_temperature, exploration, dirichlet_epsilon,
               batch_size, seed, report_every=REPORT_EVERY):
    """
    One generation's games, played concurrently and evaluated in batches.

    `batch_size` games are in flight at once and the positions they wait on go through the network
    together, which changes nothing about any individual game and is most of the difference between
    a Connect 4 generation costing seven minutes and costing one.

    Progress is reported with a rate and a projected finish, which is what tells a generation that
    has slowed down from one that has hung.
    """
    def batch_evaluator(states):
        return evaluate_batch(net, states, encoder)

    started = time.perf_counter()

    def report(completed, total):
        if report_every <= 0 or completed % report_every or completed == total:
            return
        elapsed = time.perf_counter() - started
        rate = completed / elapsed
        log.info(f'    self-play {completed}/{total} games, {elapsed:.0f}s elapsed, '
                 f'{rate * 60:.1f}/min, ~{(total - completed) / rate:.0f}s left')

    played = play_games(
        batch_evaluator, encoder, game, count, simulations,
        batch_size=batch_size, seed=seed, opening_plies=opening_plies,
        temperature_moves=temperature_moves, final_temperature=final_temperature,
        exploration=exploration, dirichlet_epsilon=dirichlet_epsilon,
        on_finished=report)

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

    The policy term is cross-entropy against a *distribution*, not a label: when the visit counts
    say two moves are equally good, a network told to pick one is being taught something false.
    `-(target * log_softmax(logits)).sum()` reduces to ordinary cross-entropy for a one-hot target.
    """
    if not buffer:
        return 0.0, 0.0, 0.0

    net.train()
    totals = [0.0, 0.0, 0.0]
    device = device_of(net)

    for _ in range(steps):
        batch = rng.sample(list(buffer), k=min(batch_size, len(buffer)))
        planes = to_tensor([example.planes for example in batch], device)
        target_policy = torch.tensor(
            [list(e.policy) for e in batch], dtype=torch.float32, device=device)
        target_value = torch.tensor(
            [e.value for e in batch], dtype=torch.float32, device=device)

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


def grading_sets(game):
    """
    The positions a game is graded over each generation, one entry per tier.

    Every tier rather than the opening alone, which is a sixth of a game: a network can be tuned to
    it and remain a novice everywhere the benchmark cannot see.

    Kept as separate tiers rather than pooled, which `ai.corpus` argues for at length: `R` reaches
    positions no sensible game visits and `P` never asks a player to recover from a bad one, so one
    figure over both would hide whichever is worse.
    """
    if game.SOLVED_DEPTH is not None:
        # `.copy()` because `enumerate_positions` yields the live state as it walks the tree, and
        # keeping the references leaves every entry pointing at one object.
        positions = [(state.copy(), ply) for state, ply in enumerate_positions(game)]

        # Solved once here rather than every generation: the answers are properties of the game,
        # and a shared table makes the one pass nearly free.
        table = Table()
        solved = {
            state.solver_key: move_values(state, table)
            for state, _ in positions if not state.is_game_over
        }
        # One tier: the whole state space is here, so there is nothing to divide.
        return {'all': (positions, lambda state: solved[state.solver_key])}

    path = corpus.CORPORA.get(game.__name__)
    if path is None:
        raise ValueError(
            f'{game.__name__} can neither be enumerated nor has a solved corpus, so a training '
            f'run has nothing to grade itself against'
        )

    sets = {}
    for tier, _ in corpus.TIERS:
        entries = corpus.load(path, tiers=(tier,))
        sets[tier] = (list(corpus.positions(entries, game)), corpus.values(entries))
    return sets


def _climb(net, encoder, game, rungs, games, seed, simulations=0):
    """
    The network against each rung, which is the metric the run is steered by.

    `simulations` chooses which player is measured. Raw (0) costs about 15s a rung and understates
    the player badly - it says which way a run is going and nothing about how strong it is.
    Searched is the player itself, at roughly 7 minutes a rung over 100 games.

    Played on the CPU whatever the run trains on. This is the one path with no batch in it - a
    ladder rung is two hundred thousand single positions - and a batch of one costs half as much
    again on a card as it does here. Copying the weights across is a couple of megabytes once.
    """
    from ai import ladder as ladders  # Local: keeps `ai.zero` importable without the match harness
    from ai.zero.mcts import MCTS

    if device_of(net).type != 'cpu':
        net = copy.deepcopy(net).to('cpu')

    def raw(state):
        priors, _ = evaluate(net, state, encoder)
        return max(state.legal_moves, key=lambda move: priors[encoder.action_index(move)])

    def searched(state):
        # No noise, and the most-visited move rather than a sample: a player being measured should
        # give its actual opinion.
        search = MCTS(lambda s: evaluate(net, s, encoder), encoder, simulations=simulations)
        return search.search(state, noise=False).move

    challenger = searched if simulations > 0 else raw

    default = ladders.for_game(game)
    rung_ladder = ladders.Ladder(
        rungs=tuple(rungs) if rungs else default.rungs,
        opening_plies=default.opening_plies)
    return ladders.climb(game, challenger, ladder=rung_ladder, games=games, seed=seed,
                         print_progress=False)


def _grade(net, encoder, sets):
    """Every tier, each its own report. See `grading_sets` for why they are not pooled."""
    return {tier: _optimal_rate(net, encoder, positions, values)
            for tier, (positions, values) in sets.items()}


def _optimal_rate(net, encoder, positions, values):
    """
    How often the raw policy picks a best move, and how wrong the value head is.

    The network alone, with no search: search papers over a weak prior, so grading with it on would
    flatter a network that is not improving.

    Batched, and there is no tree here to disturb - just independent positions - so this is the one
    place in the system where speed is free. Connect 4 one at a time is 37s, or a third of a
    generation.
    """
    states = [state for state, _ in positions]

    # In chunks: one batch of 22,100 allocates a five-block tower's activations for every position
    # at once, and spends longer in memory traffic than it saves in dispatch.
    answers = []
    for start in range(0, len(states), GRADING_CHUNK):
        answers.extend(evaluate_batch(net, states[start:start + GRADING_CHUNK], encoder))

    priors = {id(state): answer[0] for state, answer in zip(states, answers)}
    heads = {id(state): answer[1] for state, answer in zip(states, answers)}

    def raw(state):
        prior = priors[id(state)]
        return max(state.legal_moves, key=lambda move: prior[encoder.action_index(move)])

    return benchmark(raw, positions, values=values, value_fn=lambda state: heads[id(state)])
