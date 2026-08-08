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
import random
import sys
from typing import Optional, Type

import log
from ai import corpus
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
    run(
        game,
        generations=args.generations,
        games_per_generation=args.games,
        simulations=args.simulations,
        steps=args.steps,
        symmetries=args.symmetries,
        benchmark_every=args.benchmark_every,
        checkpoint_path=args.out,
        seed=args.seed,
        **extra,
    )
    log.info(f'Best checkpoint written to {args.out}')


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
    trainer.add_argument('--out', default=DEFAULT_CHECKPOINT)
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
