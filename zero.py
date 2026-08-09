"""
Training, grading and comparing learned players.

    python3 zero.py train
    python3 zero.py benchmark --player model:models/tictactoe-best.pt
    python3 zero.py benchmark --player model:models/tictactoe-best.pt+mcts:100
    python3 zero.py match --a model:models/tictactoe-best.pt --b minimax:4 --games 200

Separate from `play.py` on purpose, and the split is by *kind of use* rather than by subject.
`play.py` is a person at a terminal: it prompts, it redraws the board every ply, and it has to
survive someone typing nonsense at it. Everything here is a batch job that wants flags, gets
pointed at a file and prints a table. Putting training behind an interactive menu would be
unpleasant to re-run, and putting the game loop behind argparse would make playing a game worse.

What they share is `ai.players`, so an opponent is named the same way in both. That is the part
worth keeping in one place: "the network with 200 simulations" means one thing everywhere, and
comparing it against "the same network with none" is a change of argument rather than of code.
"""

import argparse
import os
import random
import subprocess
import sys
from typing import Callable, Optional, Sequence, Type

import log
from ai import corpus, ladder
from ai.match import play_match
from ai.oracle import benchmark, enumerate_positions, optimal_moves, play_every_line
from ai.players import describe, player, UnknownPlayer
from games import GAMES
from games.base import GameState

DEFAULT_GAME = 'TicTacToe'
DEFAULT_CHECKPOINT = 'models/tictactoe-best.pt'


def _game(name: str) -> Type[GameState]:
    for game in GAMES:
        if game.__name__.lower() == name.lower():
            return game
    raise SystemExit(f'no such game: {name}. Try one of: '
                     f'{", ".join(g.__name__ for g in GAMES)}')


def _player(spec: str):
    try:
        return player(spec)
    except UnknownPlayer as error:
        raise SystemExit(str(error))


def train(args) -> None:
    """Trains a network from scratch and leaves the best one it saw on disk."""
    from ai.zero.train import train as run  # Local: only training needs torch

    game = _game(args.game)
    if game.ENCODER is None:
        raise SystemExit(f'{game.__name__} has no ENCODER, so it cannot be learned')

    log.info(f'Training {game.__name__}: {args.generations} generations of {args.games} games '
             f'at {args.simulations} simulations.')
    extra = {} if args.exploration is None else {'exploration': args.exploration}
    if args.games_in_flight:
        extra['games_in_flight'] = args.games_in_flight
    if args.buffer_size:
        extra['buffer_size'] = args.buffer_size

    run(
        game,
        generations=args.generations,
        games_per_generation=args.games,
        simulations=args.simulations,
        steps=args.steps,
        symmetries=args.symmetries,
        benchmark_every=args.benchmark_every,
        checkpoint_path=args.out,
        latest_path=args.latest,
        metrics_path=args.metrics,
        resume_from=_resume_from(args),
        seed=args.seed,
        on_generation=_committer([args.latest, args.metrics], args.commit_every),
        **extra,
    )
    log.info(f'Best checkpoint written to {args.out}')


def _resume_from(args) -> Optional[str]:
    """
    Where a resumed run picks up, which is the `--latest` file if `--resume` was asked for.

    A path rather than a flag would mean the relaunch command differs from the launch command,
    and the moment you want to relaunch is the moment you least want to be composing a new one.
    Resuming when there is nothing to resume from starts at generation one rather than failing:
    a run killed in its first few minutes should restart, not need a different invocation.
    """
    if not args.resume:
        return None
    if args.latest and os.path.exists(args.latest):
        return args.latest
    log.info(f'--resume given but there is no checkpoint at {args.latest}; starting from scratch')
    return None


def _committer(paths: Sequence[Optional[str]], every: int) -> Optional[Callable]:
    """
    A per-generation hook that commits and pushes the run's resume point every `every` generations.

    A Connect 4 run is measured in hours on a machine that can go away, and the checkpoint is only
    insurance if it survives the machine. So the resume point goes to the branch as the run
    produces it, and a relaunch anywhere clones its way back to within `every` generations of where
    it was.

    **The latest checkpoint and the metrics, not the best network.** Those two are exactly what a
    resume needs, and each 5MB checkpoint stays in the history forever - committing the best one on
    the same schedule would double that for a file which is a deliverable rather than insurance,
    and which is worth committing once, at the end, if it turns out to be worth having at all.

    Git failing never stops the run. A push can fail for reasons that have nothing to do with the
    training - the network, a race with another push - and losing four hours of self-play to a
    failed `git push` would be a strange way to make a run more robust.
    """
    tracked = [path for path in paths if path]
    if every <= 0 or not tracked:
        return None

    def commit(progress) -> None:
        if progress.generation % every:
            return

        present = [path for path in tracked if os.path.exists(path)]
        if not _git('add', '--', *present):
            return
        if _git('diff', '--cached', '--quiet'):
            return  # Nothing changed, so there is nothing to commit

        message = (f'Checkpoint Connect 4 training at generation {progress.generation}\n\n'
                   f'Agreement with perfect play {progress.optimal_rate:.2%}, '
                   f'value MSE {progress.value_mse:.3f}.')
        if _git('commit', '-m', message):
            _git('push', '-u', 'origin', 'HEAD')

    return commit


def _git(*arguments: str) -> bool:
    """Runs one git command, reporting rather than raising. See `_committer`."""
    finished = subprocess.run(['git', *arguments], capture_output=True, text=True)
    if finished.returncode and arguments[0] != 'diff':
        log.warning(f'git {arguments[0]} failed: {finished.stderr.strip() or finished.stdout}')
    return finished.returncode == 0


def _show_worst(report) -> None:
    if report.worst:
        log.newline()
        log.info('Worst positions:')
        for board, played, best in report.worst:
            log.info(f'{board}\n  played {played}, best was {sorted(best)}')


def _grade_enumerated(game, chooser, spec, value_fn) -> None:
    """
    A game whose whole state space fits in a loop, which is tic-tac-toe and nothing else here.

    Both questions get asked, because both can be. `benchmark` walks every position and asks
    whether the player *knows* the game; `play_every_line` plays it against every line an opponent
    could take it down and asks whether it *can be beaten*. They come apart sharply and in one
    direction: a player never reaches the positions it would get wrong if it never blunders on the
    path it actually walks.
    """
    report = benchmark(chooser, enumerate_positions(game), value_fn=value_fn)
    log.newline()
    log.info('Knowing the game — every position, against the solver:')
    log.info(str(report))

    log.newline()
    log.info('Playing the game — every line an opponent could take it down:')
    for seat, name in ((True, 'first '), (False, 'second')):
        against_best = play_every_line(chooser, game, seat, opponent=optimal_moves)
        against_all = play_every_line(chooser, game, seat)
        log.info(f'  as {name} vs perfect play : {against_best}')
        log.info(f'  as {name} vs any opponent : {against_all}')

    _show_worst(report)


def _grade_corpus(game, chooser, spec, value_fn, path, only_tier) -> None:
    """
    A game too big to enumerate, graded against positions solved ahead of time.

    One report per tier and never a combined one. The tiers ask different questions - the opening
    exhaustively, then random positions, then positions from real play - and averaging them would
    produce a number whose value depended mostly on how many of each happened to be sampled.

    `play_every_line` is not offered. It is exponential in the length of a game, which tic-tac-toe
    can afford and Connect 4 cannot; the question it answers needs a different instrument here.
    """
    entries = corpus.load(path)
    lookup = corpus.values(entries)
    log.info(f'{len(entries)} solved positions from {path}')

    for tier, description in corpus.TIERS:
        chosen = [entry for entry in entries if entry.tier == tier]
        if not chosen or (only_tier and tier != only_tier):
            continue

        log.newline()
        log.info(f'{description} ({len(chosen)} positions):')
        report = benchmark(
            chooser, corpus.positions(chosen, game), values=lookup, value_fn=value_fn,
        )
        log.info(str(report))
        _show_worst(report)


def grade(args) -> None:
    """Grades a player against exact play, however this game can supply it."""
    game = _game(args.game)
    chooser = _player(args.player)

    # Seeded, because `random` and a model sampling its policy both draw from the global RNG, and
    # a benchmark whose number moves between runs is a benchmark you cannot quote. Deterministic
    # players ignore this entirely.
    random.seed(args.seed)

    value_fn = None
    if args.value and args.player.startswith('model:'):
        from ai.zero.player import model_value
        value_fn = model_value(args.player.split('+')[0][len('model:'):])

    log.info(f'Grading {describe(args.player)} on {game.__name__} against perfect play...')

    # A game that declares SOLVED_DEPTH can be searched to the end of itself, which means its
    # state space is small enough to walk and to solve on the spot. Everything else needs answers
    # computed in advance.
    if game.SOLVED_DEPTH is not None:
        _grade_enumerated(game, chooser, args.player, value_fn)
    elif game.__name__ in corpus.CORPORA:
        _grade_corpus(game, chooser, args.player, value_fn,
                      corpus.CORPORA[game.__name__], args.tier)
    else:
        raise SystemExit(
            f'{game.__name__} can neither be enumerated nor has a solved corpus, so there is '
            f'nothing to grade it against'
        )


def climb_ladder(args) -> None:
    """
    Plays a player up a fixed sequence of opponents and says where it sits.

    The other half of the pair `benchmark` starts. That one asks whether a player knows the game,
    position by position against the exact answer; this asks whether it can be beaten, which is a
    different question and often a differently-answered one.
    """
    game = _game(args.game)
    chooser = _player(args.player)

    rungs = ladder.for_game(game)
    rungs = ladder.make(
        args.rungs or rungs.rungs,
        args.opening_plies or rungs.opening_plies,
        balanced=not args.unbalanced,
    )

    log.info(f'Climbing the {game.__name__} ladder with {describe(args.player)}...')
    random.seed(args.seed)  # For a challenger that samples; see grade()
    standing = ladder.climb(game, chooser, rungs, games=args.games, seed=args.seed)

    log.newline()
    log.info(str(standing))


def match(args) -> None:
    """Plays two players off against each other over many paired games."""
    game = _game(args.game)
    first, second = _player(args.a), _player(args.b)

    log.info(f'{describe(args.a)}  vs  {describe(args.b)}   ({args.games} games)')
    result = play_match(game, first, second, games=args.games, seed=args.seed)
    log.newline()
    log.info(f'From {describe(args.a)}\'s point of view: {result}')


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(prog='zero.py', description=__doc__.split('\n')[1])
    parser.add_argument('--game', default=DEFAULT_GAME, help='which game (default: TicTacToe)')
    commands = parser.add_subparsers(dest='command', required=True)

    # Defaults are the recipe the committed checkpoint was trained with - see the README. Every
    # one of them was arrived at by measuring against the oracle rather than by taste.
    trainer = commands.add_parser('train', help='train a network from scratch')
    trainer.add_argument('--generations', type=int, default=400)
    trainer.add_argument('--games', type=int, default=80, help='self-play games per generation')
    trainer.add_argument('--simulations', type=int, default=50, help='MCTS simulations per move')
    trainer.add_argument('--steps', type=int, default=60, help='gradient steps per generation')
    trainer.add_argument('--exploration', type=float, default=None,
                         help='self-play c_puct (default 5.0, measured)')
    trainer.add_argument('--symmetries', action='store_true',
                         help='augment with the board symmetries (measured: no benefit)')
    trainer.add_argument('--benchmark-every', type=int, default=2,
                         help='grade against the oracle every N generations')
    trainer.add_argument('--out', default=DEFAULT_CHECKPOINT,
                         help='where the best network seen is written')
    trainer.add_argument('--latest', default=None,
                         help='where every generation is written, whether or not it improved; '
                              'this is the file --resume reads')
    trainer.add_argument('--resume', action='store_true',
                         help='continue from --latest if it exists, rather than starting over')
    trainer.add_argument('--commit-every', type=int, default=0,
                         help='commit and push --latest and --metrics every N generations, so '
                              'losing the machine costs generations rather than the run')
    trainer.add_argument('--metrics', default=None,
                         help='append per-generation metrics here as JSON lines, for plot.py')
    trainer.add_argument('--buffer-size', type=int, default=None,
                         help='positions kept in the replay buffer (default 20,000); scale it '
                              'with --games or a generation overflows it')
    trainer.add_argument('--games-in-flight', type=int, default=None,
                         help='self-play games advanced together (throughput only)')
    trainer.add_argument('--seed', type=int, default=1)
    trainer.set_defaults(run=train)

    grader = commands.add_parser('benchmark', help='grade a player against perfect play')
    grader.add_argument('--player', required=True, help='e.g. minimax:9, model:PATH+mcts:100')
    grader.add_argument('--value', action='store_true',
                        help='also grade the value head against the true value')
    grader.add_argument('--tier', default=None,
                        help='grade only one corpus tier: E (opening), R (random), P (real play)')
    grader.add_argument('--seed', type=int, default=0,
                        help='seeds the global RNG, so a player that samples is reproducible')
    grader.set_defaults(run=grade)

    climber = commands.add_parser('ladder', help='place a player against a fixed set of opponents')
    climber.add_argument('--player', required=True, help='e.g. minimax:4, model:PATH+mcts:200')
    climber.add_argument('--games', type=int, default=ladder.GAMES,
                         help=f'games per rung (default {ladder.GAMES})')
    climber.add_argument('--rungs', nargs='+', default=None,
                         help='override the default sequence of opponents')
    climber.add_argument('--opening-plies', type=int, default=None,
                         help='how many random plies each game starts from')
    climber.add_argument('--unbalanced', action='store_true',
                         help='start from any opening, not only ones drawn with perfect play')
    climber.add_argument('--seed', type=int, default=0)
    climber.set_defaults(run=climb_ladder)

    matcher = commands.add_parser('match', help='play two players off against each other')
    matcher.add_argument('--a', required=True)
    matcher.add_argument('--b', required=True)
    matcher.add_argument('--games', type=int, default=100)
    matcher.add_argument('--seed', type=int, default=0)
    matcher.set_defaults(run=match)

    args = parser.parse_args(argv)
    args.run(args)


if __name__ == '__main__':
    sys.exit(main())
