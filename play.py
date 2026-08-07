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

import sys
from typing import Callable, List, Optional, Tuple, Type

import log
from ai.search import alpha_beta, random_move
from games import GAMES
from games.base import GameState, Outcome

DEFAULT_DEPTH = 4


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
        except ValueError as error:
            log.warning(str(error))


def computer_player(depth: int) -> Callable:
    """A move chooser that searches `depth` plies, or plays at random at depth 0."""
    if depth == 0:
        return random_move
    return lambda state: alpha_beta(state, depth=depth)


def choose_players(state: GameState) -> Tuple[Callable, Callable]:
    """Asks who is playing each side, and how hard the computer should think."""
    kinds = ['Human', 'Computer', 'Random']
    choosers: List[Optional[Callable]] = []
    depth = DEFAULT_DEPTH

    for name, default in (('first', 0), ('second', 1)):
        log.newline()
        log.info(f'Who is playing {name}?')
        kind = _choose(f'Player to move {name}:', kinds, default)

        if kind == 0:
            choosers.append(human_player)
        elif kind == 2:
            choosers.append(random_move)
        else:
            choosers.append(None)  # Filled in below, once the depth is known

    if None in choosers:
        answer = _ask(f'Search depth [{DEFAULT_DEPTH}]: ').strip()
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

    state = game()
    first, second = choose_players(state)
    play(state, first, second)


if __name__ == '__main__':
    main()
