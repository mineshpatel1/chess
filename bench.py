"""
Measuring the two things here whose whole point is speed: the solver, and self-play.

    python3 bench.py corpus                     # re-solve the pinned corpus, timed
    python3 bench.py frontier --plies 20 18 16  # how expensive fresh positions are
    python3 bench.py selfplay --games 200       # what a generation costs, per engine

The third of the root scripts, joining `play.py` (play a game) and `zero.py` (learn one). The
split is by kind of use: this one exists because "identical results but faster" is a claim, and a
claim about speed that nobody can re-run is worth nothing. Every timing on the Connect 4 page
comes from here.

Why the solver's speed is a project in its own right. A learned Connect 4 player has to be graded
against exact answers, the way the tic-tac-toe one was - every finding there (c_puct 1.5 to 5.0
worth three points, the signed encoding 1.2, symmetry augmentation nothing) came from grading
against a solver, and none of them would have been visible through match results, where the noise
floor is wider than the effects. Connect 4 has 4.5e12 positions and cannot be enumerated, so the
benchmark has to be a *sample* of positions solved exactly - and how early in a game those
positions can be drawn from is decided by nothing but how fast this solver is.

`corpus` is the regression measurement: the same 280 positions tests/connect4/solved.py pins, with
timings rather than assertions. It reports per-stratum, because the strata differ by orders of
magnitude and a single total would be the endgame positions contributing nothing to a number the
mid-game positions dominate.

`frontier` is the forward-looking one: fresh seeded positions at plies of your choosing, to find
where solving stops being affordable. That number is the deliverable of the optimisation work,
because it decides how the eventual Connect 4 benchmark can be sampled.

`selfplay` times a generation through each engine. It needs PyTorch, and the Rust engine for the
row that names it; the solver commands need neither.

Nothing here asserts a time. Timing assertions belong in a report, not a test suite - how long a
solve takes depends on the machine, and a test that fails on a slow one is a test that gets
ignored. tests/connect4/test_solver.py decides whether the answers are right; this only says what
they cost.
"""

import argparse
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import log
from ai.corpus import load
from ai.oracle import Table, move_values
from games.base import has_moves
from games.connect4.board import Connect4
from tests.connect4.solved import SOLVED

# The solved corpus a learned player is graded against. See ai/corpus.py.
CORPUS = 'ai/corpora/connect4.txt'


def _percentile(sorted_times: List[float], fraction: float) -> float:
    """A percentile by nearest rank, which needs no interpolation and no numpy."""
    index = min(len(sorted_times) - 1, int(fraction * len(sorted_times)))
    return sorted_times[index]


def _report(strata: Dict[int, List[float]]) -> None:
    """
    A row per stratum, and the maximum alongside the median.

    Both, because they say different things and the gap between them is large. The median says
    what a position costs; the maximum says whether sampling a few hundred of them will finish.
    A stratum whose median is a tenth of a second and whose worst case is half a minute is
    affordable in bulk and unusable interactively, and one number cannot tell you that.
    """
    log.info(f'  {"ply":>5}  {"n":>4}  {"median":>9}  {"p90":>9}  {"worst":>9}  {"total":>9}')
    for ply in sorted(strata, reverse=True):
        times = sorted(strata[ply])
        log.info(
            f'  {ply:>5}  {len(times):>4}  {_percentile(times, 0.5):>8.3f}s  '
            f'{_percentile(times, 0.9):>8.3f}s  {times[-1]:>8.3f}s  {sum(times):>8.1f}s'
        )

    everything = [seconds for times in strata.values() for seconds in times]
    log.info(f'  {len(everything)} positions in {sum(everything):.1f}s')


def _time_solve(moves: List[int]) -> Tuple[float, int, List[int]]:
    """
    Solves one position from scratch and returns how long it took, with the answer.

    A fresh `Table` per position on purpose. Sharing one across a run would be faster and would be
    measuring the wrong thing: the benchmark solves positions that have nothing to do with each
    other, so what matters is the cost of a position solved cold.
    """
    state = Connect4(moves)
    table = Table()

    start = time.perf_counter()
    values = move_values(state, table)
    elapsed = time.perf_counter() - start

    best = max(values.values())
    return elapsed, best, sorted(move for move, value in values.items() if value == best)


def corpus(args) -> None:
    """Re-solves the pinned corpus, checking every answer and timing every position."""
    log.info(f'Solving the {len(SOLVED)} pinned positions in tests/connect4/solved.py...')

    strata: Dict[int, List[float]] = {}
    wrong = 0
    for ply, moves, value, optimal in SOLVED:
        elapsed, found, best = _time_solve(moves)
        strata.setdefault(ply, []).append(elapsed)

        if (found, best) != (value, optimal):
            wrong += 1
            log.warning(f'  ply {ply} {moves}: pinned {value} {optimal}, got {found} {best}')

    log.newline()
    _report(strata)
    log.newline()
    if wrong:
        raise SystemExit(f'{wrong} positions disagree with the pinned answers')
    log.info('Every answer matches what was pinned.')


def sample(ply: int, count: int, seed: int) -> List[List[int]]:
    """
    Distinct unfinished positions at exactly `ply` discs, reached by seeded random play.

    Random rather than played well, and that is the right choice for a benchmark rather than a
    convenience. Positions from strong play are the ones a strong player already handles; a
    grading set drawn from them would ask a learned player only about the ground it already holds.
    Discarding the games that finish early is what keeps the ply exact.

    Deduplicated on `solver_key`, so a stratum is `count` distinct questions rather than `count`
    draws with the easy ones repeated.
    """
    rng = random.Random(seed * 1000 + ply)
    found: List[List[int]] = []
    seen = set()

    while len(found) < count:
        state = Connect4()
        for _ in range(ply):
            if state.outcome is not None:
                break
            state.make_move(rng.choice(list(state.legal_moves)))
        else:
            # `has_moves`, not `state.legal_moves` - it is a generator and so always truthy, and
            # `any()` on it would ask whether column 0 is falsy. See games/base.py.
            if state.outcome is None and has_moves(state.legal_moves) \
                    and state.solver_key not in seen:
                seen.add(state.solver_key)
                found.append(state.columns_played)

    return found


def frontier(args) -> None:
    """
    Solves fresh positions at each requested ply, to find where it stops being affordable.

    Deepest first, so a run cut short has still produced the rows that were going to arrive.
    """
    strata: Dict[int, List[float]] = {}
    for ply in sorted(args.plies, reverse=True):
        log.info(f'Solving {args.count} positions at ply {ply}...')
        times = []
        for moves in sample(ply, args.count, args.seed):
            elapsed, _, _ = _time_solve(moves)
            times.append(elapsed)
        strata[ply] = times
        log.info(f'  median {_percentile(sorted(times), 0.5):.3f}s, '
                 f'worst {max(times):.3f}s, total {sum(times):.1f}s')

    log.newline()
    _report(strata)


def verify(args) -> None:
    """
    Re-solves the corpus with our own solver, as far back into the game as it can reach.

    The corpus is computed by an external solver, so on its own it is one program's opinion. This
    is the second opinion, and it is worth having precisely because the two share no code: ours is
    Python negamax over a sentinel-padded bitboard, Pons' is C++ over a different layout with a
    32MB opening book. Agreement between them is evidence; agreement of either with itself is not.

    Only the plies our solver can afford, deepest first, so a run cut short has still checked
    something. Everything shallower is covered instead by the game-tree consistency of the
    enumerated tier and by the published opening - see tests/connect4/test_corpus.py.
    """
    entries = [entry for entry in load(args.corpus) if entry.ply >= args.from_ply]
    entries.sort(key=lambda entry: -entry.ply)
    log.info(f'Re-solving {len(entries)} corpus positions at ply {args.from_ply} and deeper...')

    wrong = 0
    by_ply: Dict[int, int] = {}
    for entry in entries:
        state = Connect4(entry.moves)
        ours = move_values(state, Table())
        if ours != entry.values:
            wrong += 1
            log.warning(f'  ply {entry.ply} {entry.moves}: corpus {entry.values}, ours {ours}')
        by_ply[entry.ply] = by_ply.get(entry.ply, 0) + 1

    log.newline()
    for ply in sorted(by_ply, reverse=True):
        log.info(f'  ply {ply:>2}: {by_ply[ply]} checked')
    log.newline()
    if wrong:
        raise SystemExit(f'{wrong} positions disagree with our own solver')
    log.info(f'Our solver agrees with the corpus in all {len(entries)} positions it can reach.')


class Throughput:
    """What one self-play run cost, and what it says about where the time went."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.batches = 0
        self.evaluations = 0
        self.network_seconds = 0.0
        self.seconds = 0.0
        self.positions = 0

    def batch(self, positions: int, network_seconds: float) -> None:
        self.batches += 1
        self.evaluations += positions
        self.network_seconds += network_seconds

    def row(self, games: int) -> str:
        per_game = self.seconds / games
        mean_batch = self.evaluations / self.batches if self.batches else 0.0
        return (
            f'  {self.label:<30}  {self.seconds:>8.1f}s  {per_game * 1000:>8.1f}  '
            f'{mean_batch:>9.0f}  {self.evaluations / self.seconds:>11,.0f}  '
            f'{self.network_seconds / self.seconds:>7.0%}  {per_game * 1000 / 60:>9.1f}'
        )


def _play(engine: str, games: int, simulations: int, exploration: float, device, seed: int,
          in_flight: Optional[int]) -> Throughput:
    """One self-play run through whichever engine, timed from the caller's side."""
    from ai.zero import fast
    from ai.zero.net import evaluate_batch
    from ai.zero.selfplay import play_games
    from games.connect4.encoding import Connect4Encoder

    from ai.zero.net import ZeroNet, for_game
    import torch

    encoder = Connect4Encoder()
    torch.manual_seed(seed)
    net = ZeroNet(encoder.PLANE_SHAPE, encoder.POLICY_SIZE, **for_game('Connect4')).to(device)

    flight = in_flight or (games if engine == 'rust' else 32)
    measured = Throughput(f'{engine} / {device} / {flight} in flight')

    started = time.perf_counter()
    if engine == 'rust':
        result = fast.play_games(
            net, games, simulations, exploration=exploration, seed=seed, in_flight=flight,
            on_batch=measured.batch)
        measured.positions = result.positions
    else:
        def evaluator(states):
            at = time.perf_counter()
            answers = evaluate_batch(net, states, encoder)
            measured.batch(len(states), time.perf_counter() - at)
            return answers

        played = play_games(evaluator, encoder, Connect4, games, simulations,
                            batch_size=flight, seed=seed, exploration=exploration)
        measured.positions = sum(len(examples) for examples, _ in played)

    measured.seconds = time.perf_counter() - started
    return measured


def _forward(device, batches=(1, 64, 512, 4096)) -> None:
    """
    The network alone, at batch sizes spanning what the two engines can actually offer it.

    This is the table the rest of the command is about. A forward pass costs almost the same for
    one position as for sixty-four, so what a batch buys is nearly all of it, and how big a batch
    the engine can produce is the whole of the difference between the rows below.

    The best of several passes rather than the mean. A small batch is launch overhead rather than
    arithmetic, so it measures whatever else wants the machine - on a card that is also driving a
    display the same row varies twofold - and the fastest pass is the one least contaminated by
    that. Large batches are stable either way.
    """
    import torch

    from ai.zero.net import ZeroNet, for_game
    from games.connect4.encoding import Connect4Encoder

    encoder = Connect4Encoder()
    net = ZeroNet(encoder.PLANE_SHAPE, encoder.POLICY_SIZE, **for_game('Connect4')).to(device)
    net.eval()

    log.info(f'  {"batch":>7}  {"per call":>10}  {"per position":>13}  {"positions/sec":>14}')
    for size in batches:
        planes = torch.zeros((size, *encoder.PLANE_SHAPE), device=device)
        with torch.inference_mode():
            for _ in range(5):
                net(planes)
            _synchronise(device)

            repeats = 100 if size <= 512 else 20
            each = float('inf')
            for _ in range(5):
                started = time.perf_counter()
                for _ in range(repeats):
                    net(planes)
                _synchronise(device)
                each = min(each, (time.perf_counter() - started) / repeats)

        log.info(f'  {size:>7,}  {each * 1e6:>9.0f}us  {each / size * 1e6:>12.1f}us  '
                 f'{size / each:>14,.0f}')


def _synchronise(device) -> None:
    """A CUDA call returns before the card has finished, so a timing has to wait for it."""
    import torch

    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize()


def selfplay(args) -> None:
    """
    What a generation of Connect 4 self-play costs, per engine and per device.

    Every speed claim on the Connect 4 page comes from here, because a claim about speed that
    nobody can re-run is worth nothing - the same reason `corpus` exists beside the solver's tests.

    The column that decides what to do next is the last but one. Self-play is the network plus the
    engine that feeds it, and only one of those is worth attacking at a time: a run that is 90% in
    the forward pass wants a bigger batch, and one that is 20% in it does not.
    """
    try:
        from ai.zero import fast
        from ai.zero.net import choose_device, make_deterministic
    except ImportError:
        raise SystemExit('selfplay needs PyTorch: pip install -r requirements-zero.txt')

    device = choose_device(args.device)
    make_deterministic(device)

    engines = list(args.engines)
    if 'rust' in engines and not fast.available(Connect4):
        raise SystemExit(fast.why_unavailable(Connect4))

    log.info(f'The network alone on {device}:')
    log.newline()
    _forward(device)

    log.newline()
    log.info(f'{args.games} games at {args.simulations} simulations, c_puct {args.exploration}, '
             f'on {device}:')
    log.newline()
    log.info(f'  {"":<30}  {"total":>9}  {"ms/game":>8}  {"mean batch":>9}  {"evals/sec":>11}  '
             f'{"network":>7}  {"min/1000":>9}')

    for engine in engines:
        measured = _play(engine, args.games, args.simulations, args.exploration, device,
                         args.seed, args.games_in_flight)
        log.info(measured.row(args.games))

    log.newline()
    log.info(f'  {measured.positions / args.games:.1f} plies a game, so a position is worth '
             f'{args.simulations} simulations and a game about '
             f'{measured.positions / args.games * args.simulations:,.0f}.')


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    commands = parser.add_subparsers(dest='command', required=True)

    pinned = commands.add_parser('corpus', help='re-solve the pinned corpus, timed')
    pinned.set_defaults(run=corpus)

    edge = commands.add_parser('frontier', help='time fresh positions at chosen plies')
    edge.add_argument('--plies', type=int, nargs='+', default=[24, 22, 20, 18, 16],
                      help='how many discs are on the board')
    edge.add_argument('--count', type=int, default=20, help='positions per ply')
    edge.add_argument('--seed', type=int, default=0)
    edge.set_defaults(run=frontier)

    second = commands.add_parser('verify', help="re-solve the corpus with our own solver")
    second.add_argument('--corpus', default=CORPUS)
    second.add_argument('--from-ply', type=int, default=16,
                        help='the shallowest ply our solver can afford')
    second.set_defaults(run=verify)

    games = commands.add_parser('selfplay', help='what a generation costs, per engine')
    games.add_argument('--games', type=int, default=100)
    games.add_argument('--simulations', type=int, default=600)
    games.add_argument('--exploration', type=float, default=2.0,
                       help="Connect 4's, not the tic-tac-toe default")
    games.add_argument('--engines', nargs='+', default=['python', 'rust'],
                       choices=('python', 'rust'), help='which to time, in order')
    games.add_argument('--device', default=None,
                       help="'auto' takes a CUDA card if there is one")
    games.add_argument('--games-in-flight', type=int, default=None,
                       help='overrides each engine default: 32 for python, all of them for rust')
    games.add_argument('--seed', type=int, default=1)
    games.set_defaults(run=selfplay)

    args = parser.parse_args(argv)
    args.run(args)


if __name__ == '__main__':
    sys.exit(main())
