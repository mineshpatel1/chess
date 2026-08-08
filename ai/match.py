"""
Playing two move-choosers off over many games, to find out which is better.

perft says whether a game generates the right moves and tests/*/test_search_equivalence.py says
whether the search computes the right scores, but neither can say whether an evaluation is any
good. There is no oracle for that. The only answer is to play the new one against the old one a
great many times and count, which is what this is for.

Two things make the count mean something, and both are easy to leave out:

*Paired openings.* Every opening is played twice with the sides swapped. Connect 4 is a first
player win with perfect play and chess is not far off level, so an unpaired result measures the
opening and the colour at least as much as the players. Pairing cancels both, and halves the
variance for the same number of games into the bargain.

*Random openings.* Two deterministic searches play the same game every time, so without them a
"match" is one game reported N times. The opening plies are random and the rest is not.

`seed` is a promise about the openings and nothing else. A player that draws on the global
random module - `ai.search.random_move` does - will still vary from run to run.

    from ai.match import play_match
    from games.connect4.board import Connect4
    play_match(Connect4, challenger, incumbent, games=400)

Hundreds of games, not a handful. An evaluation term that feels obviously right frequently is
not, and the difference between a real improvement and noise at fifty games is not visible.
"""

import random
from typing import Callable, NamedTuple, Type

import log
from games.base import GameState

# Random plies played before either side is asked to choose, so that the games differ.
OPENING_PLIES = 2

# Openings are drawn until one is still running; a game that is over before it starts would
# measure nothing. This caps the search so a game with very short openings cannot hang.
MAX_DRAWS = 1000


class MatchResult(NamedTuple):
    """A tally from the challenger's point of view."""

    wins: int
    draws: int
    losses: int

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        """Points per game, a draw being half a point. 0.5 is level."""
        if not self.games:
            return 0.0
        return (self.wins + self.draws / 2) / self.games

    @property
    def error(self) -> float:
        """
        One standard error on the score, so that a result can be read honestly.

        A challenger scoring 0.54 over 100 games has not been shown to be better than anything:
        the error there is about 0.04, so it is inside two of them and the match is noise. This
        slightly overstates the error, because paired games are correlated and pairing is
        precisely what reduces the true variance - so a result that clears two of these has
        cleared a real bar.
        """
        if self.games < 2:
            return 0.0
        mean = self.score
        variance = (
            self.wins * (1 - mean) ** 2
            + self.draws * (0.5 - mean) ** 2
            + self.losses * mean ** 2
        ) / self.games
        return (variance / self.games) ** 0.5

    @property
    def is_significant(self) -> bool:
        """Whether the score is more than two standard errors away from level."""
        return abs(self.score - 0.5) > 2 * self.error

    def __str__(self) -> str:
        verdict = 'significant' if self.is_significant else 'not significant'
        return (
            f'+{self.wins} ={self.draws} -{self.losses} '
            f'({self.score:.3f} +/- {self.error:.3f}, {verdict}) over {self.games} games'
        )


def _opening(new_game: Callable[[], GameState], rng: random.Random, plies: int) -> GameState:
    """A position a few random moves in, and still running."""
    for _ in range(MAX_DRAWS):
        state = new_game()
        for _ in range(plies):
            if state.is_game_over:
                break
            state.make_move(rng.choice(list(state.legal_moves)))

        if not state.is_game_over:
            return state

    raise RuntimeError(f'no opening of {plies} plies left a game running')


def _play(state: GameState, first: Callable, second: Callable):
    """One game, to a finish. Deliberately not ai.simulate.simulate_game, which logs per game."""
    while not state.is_game_over:
        move = first(state) if state.turn else second(state)
        state.make_move(move)
    return state.result


def play_match(
    new_game: Type[GameState],
    challenger: Callable,
    incumbent: Callable,
    games: int = 100,
    opening_plies: int = OPENING_PLIES,
    seed: int = 0,
    print_summary: bool = True,
) -> MatchResult:
    """
    Plays `challenger` against `incumbent` and returns the tally from the challenger's side.

    Games are played in pairs from a shared opening, the challenger moving first in one and
    second in the other, so an odd `games` is rounded down to the pair below it.
    """
    rng = random.Random(seed)
    wins = draws = losses = 0

    for pair in range(games // 2):
        opening = _opening(new_game, rng, opening_plies)

        for challenger_is_first in (True, False):
            state = opening.copy()
            if challenger_is_first:
                outcome = _play(state, challenger, incumbent)
            else:
                outcome = _play(state, incumbent, challenger)

            if outcome.winner is None:
                draws += 1
            elif outcome.winner == challenger_is_first:
                wins += 1
            else:
                losses += 1

        if print_summary and (pair + 1) % 25 == 0:
            log.info(f'  {2 * (pair + 1)} games: {MatchResult(wins, draws, losses)}')

    result = MatchResult(wins, draws, losses)
    if print_summary:
        log.info(str(result))
    return result
