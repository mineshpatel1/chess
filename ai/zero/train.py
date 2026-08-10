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
from typing import Callable, Deque, Dict, List, NamedTuple, Optional, Sequence, Tuple, Type

import torch
import torch.nn.functional as F

import log
from games.base import GameState
from ai import corpus
from ai.oracle import Grade, Report, Table, benchmark, enumerate_positions, move_values
from ai.zero import checkpoint
from ai.zero.checkpoint import save
from ai.zero.metrics import Recorder, truncate_after
from ai.zero.net import (
    ZeroNet, evaluate, evaluate_batch, flush_denormals, for_game as architecture, to_tensor,
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
# pass. Purely a throughput setting: every game gets the same tree it would have got alone, and
# `tests/zero/test_selfplay.py` asserts that a batch of sixty-four plays what a batch of one plays.
# One runs exactly the position-at-a-time path that existed before batching.
GAMES_IN_FLIGHT = 32

# How often to grade the network against the oracle. Exact and useful, but it is instrumentation
# rather than training: measured at 80 games and 150 simulations it was 32% of a generation's wall
# clock, against 3.5% for the gradient steps. Every generation is right while tuning; less often is
# right when the answer is already known and the run just has to finish.
BENCHMARK_EVERY = 1

# Games between progress reports during self-play, which is where a generation spends its time.
#
# A tic-tac-toe generation is a second and says nothing until it is done, which is right. A Connect
# 4 generation at 600 simulations is twenty minutes, and a run that prints nothing for twenty
# minutes is one where a slowdown and a hang look identical - the afternoon that produced this
# constant was spent telling those apart by reading `/proc`. Reporting a *rate* rather than a bare
# count is the point: the rate is what shows a generation getting slower while it happens.
REPORT_EVERY = 25

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
# generation Connect 4 runs, one seed each (the metrics files are not kept - the finding is):
#
#     c_puct  sims   target entropy   agreement   value mse   game plies
#        1.5    50      64% of unif       69.0%       0.759         18.6
#        2.5    50      74%               70.7%       0.812         17.4
#        5.0    50      87%               69.9%       0.704         17.1
#        1.5   200      55%               69.9%       0.789         20.2
#        2.5   200      63%               69.1%       0.864         20.3
#        2.0   600      47%               72.9%       0.683         24.2
#
# **None of those differences is measurable at that budget, which is the real finding.** Four runs
# of the *same* configuration differing only in seed spread 5.5 points of agreement (64.5% to
# 69.9%, sd 2.5) and 0.176 of value MSE - wider than every cell-to-cell difference in the grid
# above, and wider than the 4.5-point spread of a temperature schedule sweep run afterwards.
#
# So the grid establishes nothing about c_puct, and nothing about simulations either: 600 looked
# best on all four columns by 2.2 points, which is under half the noise floor. Twelve-generation
# single-seed runs cannot see effects of the size hyperparameters plausibly have. Detecting two
# points would need five or six seeds a cell, at half an hour each.
#
# What does clear the floor: thirty generations reached 74.6% where twelve average 68.1% across
# four seeds. More data is the only lever with evidence behind it, so it is the one to pull, and
# these values stay where they are until something can actually measure them.
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

# The opponents a generation is scored against, and how many games each.
#
# **This is the primary metric, and agreement is now a diagnostic.** The two questions - does it
# know the game, can it be beaten - come apart much harder than this project assumed. Two Connect 4
# networks half a point apart on agreement scored 0.055 and 0.635 against `minimax:4`. Whichever
# number chooses checkpoints and stops runs should be the one anybody actually cares about.
#
# Raw policy rather than with search, because it costs about 15s a rung against 100 games and the
# same run measures both: search only ever helped here, so a raw score that climbs is a player that
# is improving, and the expensive `zero.py ladder` with simulations is what settles a final claim.
# `None` means the game's own ladder from `ai.ladder.LADDERS`. A caller steering a run usually
# wants fewer: the rungs a player has already saturated (`random` at 1.000) or cannot touch cost
# the same to play and move the mean by nothing, so naming the two or three either side of the
# player's current strength makes the metric sharper for the same time.
LADDER_RUNGS = None
LADDER_GAMES = 100

# Simulations the challenger searches with when it climbs. Zero is the raw policy: cheap, and
# a direction indicator only - a network scoring 0.39 raw across depths 2, 4 and 6 beat depth 5
# at 0.635 with a hundred simulations, so the raw number says which way a run is going and
# nothing about how strong it is. Above zero measures the player itself, at roughly 7 minutes a
# rung over 100 games.
LADDER_SIMULATIONS = 0

# How often to play them. Every generation at Connect 4's ~40 minutes is under 2% of the time; a
# game whose generations are seconds wants this far less often.
LADDER_EVERY = 1

# Which measure picks the best checkpoint, **per game**, because the right answer differs and both
# wrong answers are silent.
#
# Tic-tac-toe: agreement. Its ladder saturates - the network reaches perfect play early and then
# every rung returns the same score for the rest of the run, so choosing on it is choosing
# arbitrarily among ties for most of the training, while agreement is still resolving real
# differences (77 positions still wrong in a network that cannot be beaten).
#
# Connect 4: the ladder. Its agreement saturates in usefulness rather than in value - graded on the
# opening tier it asks about a sixth of a game, and two networks half a point apart on it scored
# 0.055 and 0.635 against `minimax:4`.
#
# The general shape: pick whichever measure still *moves* where the player actually is. A game with
# no ladder, or a run with the ladder off, falls back to agreement.
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
    denormal_weights: int = 0

    # The primary metric: mean score against `LADDER_RUNGS`, and the strongest rung actually beaten
    # (significantly, not merely led). `tier_rates` carries agreement per corpus tier, which is now
    # a diagnostic rather than the headline.
    ladder_score: float = 0.0
    highest_rung: str = ''
    tier_rates: Optional[Dict[str, float]] = None
    ladder_seconds: float = 0.0
    self_play_seconds: float = 0.0
    learn_seconds: float = 0.0
    grade_seconds: float = 0.0

    def __str__(self) -> str:
        """The ladder first, because that is the number the run is now steered by."""
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
    sets = grading_sets(game)
    optimiser = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    buffer: Deque[Example] = deque(maxlen=buffer_size)

    best_state, best_rate = None, -1.0
    first = 1

    # Which measure `best` means, recorded into every checkpoint so a resume can refuse to compare
    # against a bar set on a different one. Per game unless the caller says otherwise.
    #
    # **The rungs are part of the measure's identity, not a detail of it.** "ladder" against depths
    # 4, 5 and 6 scores about 0.73 for a network that scores 0.59 against 7 and 8 - so a resume that
    # changed the rungs and compared names alone would carry a bar no checkpoint could clear, and
    # the best network would silently never update again. That is the same failure as resuming a
    # ladder run from an agreement bar, one level down, and it was caught the same way: by reading
    # the number in the startup line.
    measure = metric or metric_for(game.__name__)
    if measure not in ('ladder', 'agreement'):
        raise ValueError(f'unknown selection metric {measure!r}; use "ladder" or "agreement"')
    if measure == 'ladder' and ladder_every <= 0:
        log.warning('selection asked for the ladder but it is switched off; using agreement')
        measure = 'agreement'

    if resume_from:
        # The replay buffer is deliberately not restored - see ai/zero/checkpoint.py. The first
        # generation after a resume therefore learns from a buffer holding one generation rather
        # than several, which costs a little and saves carrying hundreds of megabytes around.
        blob = checkpoint.load(resume_from, game=game.__name__)
        net.load_state_dict(blob['weights'])
        if blob.get('optimiser'):
            optimiser.load_state_dict(blob['optimiser'])
        first = blob['generation'] + 1
        best_state = {k: v.clone() for k, v in net.state_dict().items()}

        # Only when the bar was set on the same measure. A ladder score and an agreement rate are
        # both numbers between 0 and 1 and are not remotely the same quantity: resuming a ladder
        # run from an agreement bar of 0.812 sets a target no ladder score will ever reach, and the
        # best checkpoint then silently never updates again for the rest of the run. Caught on the
        # first resume after the metric changed, from a startup line reading "best so far 81.20%".
        stored = blob['metadata'].get('metric')
        if stored == _measure_identity(measure, game, ladder_rungs):
            best_rate = blob['metadata'].get('best_rate', -1.0)
            log.info(f'Resuming {resume_from} from generation {first}, '
                     f'best {measure} so far {best_rate:.3f}')
        else:
            log.info(f'Resuming {resume_from} from generation {first}; it was scored on '
                     f'{stored or "an older metric"} and this run scores on {measure}, '
                     f'so the best-so-far bar starts again')

        # The metrics file can be one generation ahead of the checkpoint: a generation is recorded
        # before its weights are written, because the record is what says the generation happened.
        # A run killed inside that window would otherwise record the same generation number twice.
        if metrics_path:
            dropped = truncate_after(metrics_path, blob['generation'])
            if dropped:
                log.info(f'dropped {dropped} recorded generation(s) past the checkpoint')

    recorder = Recorder(metrics_path, append=bool(resume_from))
    last_reports = {tier: _EMPTY_REPORT for tier in sets}
    last_standing = None

    # Which tier's agreement travels as `optimal_rate`. The opening for a corpus-graded game, so
    # the field keeps meaning what it meant for every run already recorded; the whole space for a
    # game small enough to enumerate.
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

        # Immediately after the steps that create them, so the next generation's self-play, its
        # gradient steps and its benchmark all run on normal floats. See `flush_denormals` - left
        # alone these accumulate under weight decay and make the whole loop several times slower
        # while doing exactly the same work.
        denormals = flush_denormals(net)
        learn_seconds = time.perf_counter() - learn_started

        graded = generation % max(benchmark_every, 1) == 0 or generation == generations
        grade_started = time.perf_counter()
        reports = _grade(net, encoder, sets) if graded else last_reports
        grade_seconds = time.perf_counter() - grade_started
        last_reports = reports
        report = reports[headline]

        # The primary metric, and the one `best` is chosen on. Agreement is kept as a diagnostic
        # because it is exact and free; it is not the headline any more, having twice said two
        # networks were the same player when one of them beat `minimax:4` and the other lost 93
        # games in 100 to it.
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

        # The run's own measure, computed once here so the checkpoints written below and the
        # comparison made afterwards cannot disagree about what `best` means.
        rate = progress.ladder_score if measure == 'ladder' else report.overall.rate

        # Written every generation whether or not it improved, because this is the file a
        # resume reads. `checkpoint_path` holds the *best* network, which is what you want to
        # play against; resuming from that would replay every generation since it was set.
        if latest_path:
            save(net, latest_path, game=game.__name__, generation=generation,
                 metadata={'best_rate': max(best_rate, rate),
                           'metric': _measure_identity(measure, game, ladder_rungs),
                           'ladder_score': progress.ladder_score,
                           'optimal_rate': report.overall.rate,
                           'simulations': simulations},
                 optimiser=optimiser)

        # After the checkpoint, not before. A hook that copies the run somewhere - which is what
        # `zero.py train --commit-every` does - would otherwise take generation N's metrics line
        # together with generation N-1's weights, and a resume from that pair replays a generation
        # and records its number twice. Caught in flight: the first automatic commit of a long run
        # carried the metrics and no checkpoint at all, because the checkpoint had not changed yet.
        if on_generation:
            on_generation(progress)

        # Chosen on the ladder when there is one, and on agreement otherwise. The ladder carries
        # sampling noise that the exhaustive corpus does not - 100 games is worth about +/-0.045 -
        # but a precise measurement of the wrong thing is what kept a network that loses 93 games
        # in 100 looking like the best checkpoint of its run.
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

    # Named rather than assumed. This line used to say "agreement with perfect play" whatever the
    # number was, and after the metric changed it went on saying it about a ladder score.
    if measure == 'ladder' and last_standing is not None:
        opponents = ', '.join(rung.spec for rung in last_standing.rungs)
        log.info(f'best ladder score against {opponents}: {best_rate:.3f}')
    else:
        log.info(f'best raw-policy agreement with perfect play: {best_rate:.2%}')
    return net


def _measure_identity(measure: str, game, rungs) -> str:
    """
    The measure `best` is chosen on, including which opponents when it is the ladder.

    A bar of 0.740 set against depths 4, 5 and 6 is unreachable for the same network measured
    against 7 and 8, so a resume comparing "ladder" to "ladder" would freeze the best checkpoint
    for the rest of the run without saying anything.
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

    `batch_size` games are in flight at once and the positions they are waiting on go through the
    network together. It changes nothing about any individual game - each runs an ordinary
    sequential search and gets the tree it would have got alone - and it is most of the difference
    between a Connect 4 generation costing seven minutes and costing one.

    Progress is reported as it goes, with a rate and a projected finish. This is the only thing a
    long generation says while it is running, and the rate is the useful part: it is what makes a
    generation that has slowed down distinguishable from one that has hung, without which the
    difference can only be found by inspecting the process.
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


def grading_sets(game):
    """
    The positions a game is graded over each generation, one entry per tier.

    **Every tier, not just the opening, and the reason is a measurement failure worth recording.**
    Grading used to run on tier `E` alone - every position out to six discs. Two Connect 4 networks
    then scored 81.72% and 81.20% on it, half a point apart, while one lost 93 games in 100 to
    `minimax:4` and the other beat it. The opening is a sixth of a game; a network can be tuned to
    it and remain a novice everywhere the benchmark cannot see. The tell was in the metrics all
    along - the stronger network's self-play games ran 28.7 plies against 19 - and nobody was
    looking, because the headline number said the two were the same player.

    Kept as separate tiers rather than pooled, which `ai.corpus` argues for at length: `R` reaches
    positions no sensible game visits and `P` never asks a player to recover from a bad one, so one
    figure over both would hide whichever is worse - and which is worse is the interesting part.
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
        # One tier, because there is nothing to divide: the whole state space is here.
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

    `simulations` chooses which player is measured, and the two answer different questions.

    **Raw (0) is cheap and understates the player badly.** It costs about 15s a rung over 100 games
    and it is a fine *direction* indicator, but measured here a network scoring 0.39 raw across
    depths 2, 4 and 6 beat depth 5 at 0.635 once it was allowed 100 simulations. No claim about
    playing strength should come from the raw number.

    **Searched is what the player actually is**, at roughly 7 minutes per rung over 100 games -
    about 28% on top of a Connect 4 generation for two rungs, which is worth paying when the
    question is how strong the thing is rather than which way it is moving.
    """
    from ai import ladder as ladders  # Local: keeps `ai.zero` importable without the match harness
    from ai.zero.mcts import MCTS

    def raw(state):
        priors, _ = evaluate(net, state, encoder)
        return max(state.legal_moves, key=lambda move: priors[encoder.action_index(move)])

    def searched(state):
        # No noise and the most-visited move rather than a sample: a player being measured should
        # give its actual opinion rather than a draw from it.
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
