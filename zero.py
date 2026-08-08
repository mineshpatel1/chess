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
import sys
from typing import Optional, Type

import log
from ai.match import play_match
from ai.oracle import benchmark
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
    run(
        game,
        generations=args.generations,
        games_per_generation=args.games,
        simulations=args.simulations,
        checkpoint_path=args.out,
        seed=args.seed,
    )
    log.info(f'Best checkpoint written to {args.out}')


def grade(args) -> None:
    """Grades a player against perfect play in every position of the game."""
    game = _game(args.game)
    chooser = _player(args.player)

    value_fn = None
    if args.value and args.player.startswith('model:'):
        from ai.zero.player import model_value
        value_fn = model_value(args.player.split('+')[0][len('model:'):])

    log.info(f'Grading {describe(args.player)} on {game.__name__} against perfect play...')
    report = benchmark(chooser, game, value_fn=value_fn)
    log.newline()
    log.info(str(report))

    if report.worst:
        log.newline()
        log.info('Worst positions:')
        for board, played, best in report.worst:
            log.info(f'{board}\n  played {played}, best was {sorted(best)}')


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

    trainer = commands.add_parser('train', help='train a network from scratch')
    trainer.add_argument('--generations', type=int, default=60)
    trainer.add_argument('--games', type=int, default=40, help='self-play games per generation')
    trainer.add_argument('--simulations', type=int, default=60, help='MCTS simulations per move')
    trainer.add_argument('--out', default=DEFAULT_CHECKPOINT)
    trainer.add_argument('--seed', type=int, default=0)
    trainer.set_defaults(run=train)

    grader = commands.add_parser('benchmark', help='grade a player against perfect play')
    grader.add_argument('--player', required=True, help='e.g. minimax:9, model:PATH+mcts:100')
    grader.add_argument('--value', action='store_true',
                        help='also grade the value head against the true value')
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
