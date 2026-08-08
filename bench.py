"""
Measuring the solver: how long it takes, and how far back into a game it can reach.

    python3 bench.py corpus                     # re-solve the pinned corpus, timed
    python3 bench.py frontier --plies 20 18 16  # how expensive fresh positions are

The third of the root scripts, joining `play.py` (play a game) and `zero.py` (learn one). The
split is by kind of use: this one exists because "identical results but faster" is a claim, and a
claim about speed that nobody can re-run is worth nothing.

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
from ai.oracle import Table, move_values
from games.base import has_moves
from games.connect4.board import Connect4
from tests.connect4.solved import SOLVED


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

    args = parser.parse_args(argv)
    args.run(args)


if __name__ == '__main__':
    sys.exit(main())
