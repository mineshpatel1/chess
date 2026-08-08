"""
Building the solved corpus: choosing the positions, and getting the answers.

Run once, by hand, to produce `ai/corpora/connect4.txt`. Kept as a module rather than as a shell
history because a corpus whose provenance is "somebody ran something" is a corpus nobody can
argue with - and the sampling choices below are arguable, so they had better be readable.

**Three tiers, and they are never averaged together.**

`E` is every distinct position from the empty board through six discs: 22,100 of them, and *all*
of them, so there is no sampling to be biased. It is affordable because the game opens narrowly -
7, 49, 238, 1,120, 4,263, 16,422 distinct positions per ply - and it stops at six because ply
seven is the first at which a game can already be over (a win needs `2 * CONNECT - 1` discs).
This is the tier that matters most: Connect 4 is a first-player win and the opening is where that
win is kept or thrown away.

`R` is seeded random play, and `P` is play between real players. Both are needed and they measure
different things. Random play reaches positions no sensible game visits, so a score over `R` alone
says little about how a player will actually do; play between decent players never asks a player
to recover from a bad position, which is exactly where blunders live. A single number over the two
mixed together would hide whichever is worse, so `ai.corpus` keeps the tier on every line and the
benchmark reports them apart.

`P` is epsilon-greedy rather than pure: alpha-beta at depth 4, but with a random move `EXPLORATION`
of the time. Pure alpha-beta is deterministic, so it would produce one game, not a corpus - and
some deviation is closer to what a learning player's opponents actually do than perfect
consistency is.

**The answers come from outside this project.** Our solver reaches ply 16 and the corpus needs ply
0, which no amount of optimisation closes, so the values are computed by Pascal Pons' Connect 4
solver <https://github.com/PascalPons/connect4> with its opening book. It is AGPL and is *not*
vendored: it is fetched and built into `third-party-engines/`, which is gitignored, and only the
numbers it produces are committed. See the README for the four commands.

Only the *sign* of its score is kept. Pons scores by distance to the end of the game, which is
more than this project's {-1, 0, 1} needs and would have to be decoded correctly to be used - and
decoding it wrong is a silent error. Taking the sign also matches how the benchmark grades: a
slower win is not a mistake.
"""

import random
import subprocess
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple

import log
from ai.corpus import Entry, format_entry
from ai.search import alpha_beta
from games.base import has_moves
from games.connect4.board import Connect4
from games.connect4.constants import COLS

# Where a hand-fetched Pons build is expected to be. Nothing downloads it automatically: it is
# AGPL, and pulling a copyleft binary in behind the user's back is not this script's business.
SOLVER = 'third-party-engines/connect4/c4solver'
BOOK = 'third-party-engines/connect4/7x6.book'

# What Pons returns for a column that cannot be played.
INVALID_MOVE = -1000

# How often the `P` tier deviates from its own best move, so that the games differ from each other.
EXPLORATION = 0.15

# The depth `P` plays at. Deep enough to be recognisably real play, shallow enough to be free.
PLAY_DEPTH = 4

ENUMERATED_TO = 6  # The last ply that fits in memory whole
SAMPLED_PLIES = range(7, 35)  # Where sampling takes over, to the point positions run out


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def enumerate_openings(last_ply: int = ENUMERATED_TO) -> List[List[int]]:
    """
    Every distinct position from the empty board out to `last_ply` discs.

    Breadth-first and deduplicated on `solver_key` at every level, which is what keeps this finite:
    there are 7^6 = 117,649 ways to play six moves but only 16,422 positions they reach, because
    the order columns are played in mostly does not matter.
    """
    frontier: Dict[int, List[int]] = {Connect4().solver_key: []}
    found = [[]]  # type: List[List[int]]

    for ply in range(1, last_ply + 1):
        nxt: Dict[int, List[int]] = {}
        for moves in frontier.values():
            state = Connect4(moves)
            for column in list(state.legal_moves):
                state.make_move(column)
                if state.solver_key not in nxt:
                    nxt[state.solver_key] = state.columns_played
                state.unmake_move()

        frontier = nxt
        found.extend(nxt.values())
        log.info(f'  ply {ply}: {len(nxt)} distinct positions')

    return found


def _collect(
    games: Iterator[List[int]], plies: Sequence[int], count: int, budget: int = 200000,
) -> List[List[int]]:
    """
    Walks games, keeping the first `count` distinct positions seen at each of `plies`.

    One pass over the games rather than one per ply, because a game passes through every ply on its
    way and re-playing it for each would be the same work several times over.

    `budget` caps how many games are played, and it is not a formality: **the deepest plies may not
    be fillable at all.** A position at ply 34 is one where thirty-four discs are down and nobody
    has four in a row, which is a rarer game the better the players are - so the `P` tier thins out
    at the bottom of the board where the `R` tier does not. Running out is reported rather than
    waited on, because the alternative is a generator that hangs and looks like it is working.
    """
    wanted = set(plies)
    seen: Set[int] = set()
    per_ply: Dict[int, List[List[int]]] = {ply: [] for ply in plies}

    played = 0
    for moves in games:
        played += 1
        state = Connect4()
        for ply, column in enumerate(moves + [None]):
            if ply in wanted and len(per_ply[ply]) < count:
                if state.outcome is None and has_moves(state.legal_moves):
                    if state.solver_key not in seen:
                        seen.add(state.solver_key)
                        per_ply[ply].append(state.columns_played)
            if column is None:
                break
            state.make_move(column)

        if all(len(found) >= count for found in per_ply.values()):
            break
        if played >= budget:
            short = {ply: len(found) for ply, found in per_ply.items() if len(found) < count}
            log.warning(f'  {played} games was not enough to fill every ply; short: {short}')
            break

    return [moves for ply in plies for moves in per_ply[ply]]


def _random_games(seed: int) -> Iterator[List[int]]:
    """Endless seeded random games, each played until somebody wins or the board fills."""
    rng = random.Random(seed)
    while True:
        state = Connect4()
        while state.outcome is None and has_moves(state.legal_moves):
            state.make_move(rng.choice(list(state.legal_moves)))
        yield state.columns_played


def _played_games(seed: int) -> Iterator[List[int]]:
    """
    Endless games between two alpha-beta players that deviate `EXPLORATION` of the time.

    Both sides are the same player, so this is self-play by a classical engine. That is the point:
    the positions it reaches are the ones real play reaches, which is the half of the corpus that
    random play cannot produce.
    """
    rng = random.Random(seed)
    while True:
        state = Connect4()
        while state.outcome is None and has_moves(state.legal_moves):
            if rng.random() < EXPLORATION:
                state.make_move(rng.choice(list(state.legal_moves)))
            else:
                state.make_move(alpha_beta(state, depth=PLAY_DEPTH))
        yield state.columns_played


def choose(count: int, seed: int) -> List[Tuple[str, List[int]]]:
    """Every position the corpus will hold, tagged with the tier that chose it."""
    log.info('Enumerating the opening...')
    chosen = [('E', moves) for moves in enumerate_openings()]

    log.info(f'Sampling {count} random-play positions at plies {SAMPLED_PLIES[0]}'
             f'-{SAMPLED_PLIES[-1]}...')
    chosen += [('R', moves) for moves in _collect(_random_games(seed), SAMPLED_PLIES, count)]

    log.info(f'Sampling {count} real-play positions at the same plies...')
    chosen += [('P', moves) for moves in _collect(_played_games(seed), SAMPLED_PLIES, count)]

    return chosen


def solve_with(
    positions: Sequence[List[int]],
    solver: str = SOLVER,
    book: str = BOOK,
    chunk: int = 2000,
) -> Dict[str, Dict[int, int]]:
    """
    Hands the positions to Pons' solver and reads back what every column is worth.

    Keyed by the move string rather than by position in the output, because the solver *skips* a
    line it will not accept - writing a complaint to stderr and nothing to stdout - so lining the
    two up by index would silently shift every answer after the first refusal.

    Our columns are zero-indexed and his move strings are one-indexed. The mapping is applied on
    the way in and undone on the way out, and it has to be the *same* mapping both ways: because
    the board is mirror-symmetric, a consistent relabelling is harmless and an inconsistent one
    leaves every value right and every move set reversed.
    """
    answers: Dict[str, Dict[int, int]] = {}

    for start in range(0, len(positions), chunk):
        batch = positions[start:start + chunk]
        stdin = '\n'.join(''.join(str(column + 1) for column in moves) for moves in batch) + '\n'
        result = subprocess.run([solver, '-a', '-b', book], input=stdin,
                                capture_output=True, text=True)

        for line in result.stdout.splitlines():
            fields = line.split()
            if len(fields) == COLS:  # The empty board echoes as nothing, then its seven scores
                moves, scores = '', fields
            elif len(fields) == COLS + 1:
                moves, scores = fields[0], fields[1:]
            else:
                raise ValueError(f'cannot read the solver output {line!r}')

            values = {
                column: _sign(int(score))
                for column, score in enumerate(scores)
                if int(score) != INVALID_MOVE
            }
            answers[moves] = values

        log.info(f'  solved {min(start + chunk, len(positions))}/{len(positions)}')

    return answers


def build(path: str, count: int, seed: int, solver: str = SOLVER, book: str = BOOK) -> None:
    """Chooses the positions, solves them, and writes the corpus."""
    chosen = choose(count, seed)
    log.info(f'{len(chosen)} positions chosen. Solving...')

    answers = solve_with([moves for _, moves in chosen], solver, book)

    missing = 0
    entries = []
    for tier, moves in chosen:
        key = ''.join(str(column + 1) for column in moves)
        if key not in answers:
            missing += 1
            continue
        entries.append(Entry(tier, moves, answers[key]))

    if missing:
        raise SystemExit(f'{missing} positions came back unsolved; refusing to write a partial file')

    with open(path, 'w') as handle:
        handle.write(
            '# Connect 4 positions solved exactly. See ai/corpus.py for the format and\n'
            '# ai/generate.py for how the positions were chosen and where the answers came from.\n'
            '#\n'
            '# tier  moves played (our columns, zero-indexed)  value of each column to the mover\n'
        )
        for entry in entries:
            handle.write(format_entry(entry, COLS) + '\n')

    log.info(f'Wrote {len(entries)} positions to {path}')
