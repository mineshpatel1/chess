"""
Play any of the games in this project from a terminal.

Generic over the registry in `games/__init__.py`, which is what that module's docstring
anticipated it for. Nothing here knows what a chess move or a connect-4 move is: it renders a
position with `str`, asks the game to turn what a person typed into a move with `parse_move`,
and asks `ai.search` for the other kind of move. Adding a game to the registry is the whole of
adding it here.

The loop is written out rather than handed to `ai.simulate.simulate_game`, which looks like the
same thing and is not. That one plays a game as fast as it can and reports at the end, which is
what a play-off between two evaluations wants. This one has to show the board after every ply,
and has to survive a person typing nonsense at it - neither of which belongs in a harness.

    python3 play.py
"""

import glob
import os
import sys
from typing import Callable, List, Optional, Tuple, Type

import log
# DEFAULT_DEPTH and default_depth are re-exported: they used to live here, and both the
# prompt below and tests/tictactoe/test_play.py still reach for them by this name.
from ai.players import DEFAULT_DEPTH, UnknownPlayer, default_depth, player
from ai.search import random_move
from games import GAMES
from games.base import GameState, Outcome

# Where `zero.py train` leaves its checkpoints, and so where to look for something to play.
MODEL_DIR = 'models'


def _ask(prompt: str) -> str:
    """A line from the player, treating end of input as a quit rather than an error."""
    try:
        return input(prompt)
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)


def _choose(prompt: str, options: List[str], default: int = 0) -> int:
    """The index of one of `options`, asked for until the answer is one of them."""
    for number, option in enumerate(options):
        marker = ' (default)' if number == default else ''
        log.info(f'  {number}. {option}{marker}')

    while True:
        answer = _ask(f'{prompt} ').strip()
        if not answer:
            return default
        if answer.isdigit() and int(answer) < len(options):
            return int(answer)
        log.warning(f'Pick a number from 0 to {len(options) - 1}.')


def human_player(state: GameState):
    """Asks the person whose turn it is for a move, until they give a legal one."""
    while True:
        try:
            return state.parse_move(_ask('Your move: '))
        except Exception as error:
            log.warning(f"Invalid move, try again, moves in the form e2d2")


def computer_player(depth: int) -> Callable:
    """
    A move chooser that searches `depth` plies, or plays at random at depth 0.

    Routed through `ai.players.player` rather than closing over a search directly, so this is the
    one place that decides between the Python and Rust alpha-beta - `minimax:N` takes the Rust
    search where it is built, exactly as a `minimax:N` typed at `zero.py` would.
    """
    if depth == 0:
        return random_move
    return player(f'minimax:{depth}')


def checkpoints(game: Type[GameState]) -> List[str]:
    """
    Trained networks on disk that this game could play against.

    Matched by filename rather than by opening every file, so a directory of Connect 4
    checkpoints does not cost a torch import to skip. `ai.zero.checkpoint.load` does the real
    check when one is actually chosen, and refuses a network trained on another game.
    """
    if game.ENCODER is None:
        return []
    return sorted(glob.glob(os.path.join(MODEL_DIR, f'{game.__name__.lower()}*.pt')))


def _choose_model(game: Type[GameState]) -> Optional[Callable]:
    """Picks a checkpoint and how hard it should think. None if there is nothing to pick."""
    found = checkpoints(game)
    if not found:
        log.warning(f'No {game.__name__} checkpoints in {MODEL_DIR}/. Train one with zero.py.')
        return None

    log.newline()
    log.info('Which model?')
    path = found[_choose('Model:', found)]

    log.newline()
    log.info('How many simulations should it search? 0 plays the network\'s raw intuition with')
    log.info('no lookahead at all, which is worth trying at least once for the contrast.')
    answer = _ask('Simulations [200]: ').strip()
    simulations = int(answer) if answer.isdigit() else 200

    spec = f'model:{path}' + (f'+mcts:{simulations}' if simulations else '')
    try:
        return player(spec)
    except UnknownPlayer as error:
        log.warning(str(error))
        return None


def choose_players(game: Type[GameState]) -> Tuple[Callable, Callable]:
    """Asks who is playing each side, and how hard each computer should think."""
    kinds = ['Human', 'Computer', 'Random', 'Model']
    choosers: List[Optional[Callable]] = []
    depth = default_depth(game)

    for name, default in (('first', 0), ('second', 1)):
        log.newline()
        log.info(f'Who is playing {name}?')
        kind = _choose(f'Player to move {name}:', kinds, default)

        if kind == 0:
            choosers.append(human_player)
        elif kind == 2:
            choosers.append(random_move)
        elif kind == 3:
            choosers.append(_choose_model(game))  # None falls through to the computer below
        else:
            choosers.append(None)  # Filled in below, once the depth is known

    if None in choosers:
        if game.SOLVED_DEPTH is not None:
            log.newline()
            log.info(f'{game.__name__} is solved at depth {game.SOLVED_DEPTH}, so the default '
                     f'searches the whole game and cannot be beaten.')

        answer = _ask(f'Search depth [{depth}]: ').strip()
        if answer.isdigit():
            depth = int(answer)
        choosers = [computer_player(depth) if c is None else c for c in choosers]

    return choosers[0], choosers[1]


def play(state: GameState, first: Callable, second: Callable) -> Outcome:
    """
    Plays `state` out, showing the board before every move and at the end.

    `is_game_over` is the whole of the rules from here: it covers a win while moves remain and
    a board with nothing left to play, because the game decides which of those it has.
    """
    while not state.is_game_over:
        log.newline()
        log.info(state)
        log.info(f'{"First" if state.turn else "Second"} player to move.')

        move = first(state) if state.turn else second(state)
        state.make_move(move)
        log.info(f'Played {move}.')

    log.newline()
    log.info(state)
    log.info(str(state.result))
    return state.result


def main() -> None:
    log.info('Which game?')
    games: Tuple[Type[GameState], ...] = GAMES
    game = games[_choose('Game:', [cls.__name__ for cls in games])]

    first, second = choose_players(game)
    play(game(), first, second)


if __name__ == '__main__':
    main()
