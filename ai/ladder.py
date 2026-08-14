"""
How strong a player is, said in games rather than in positions.

There are two questions worth asking about a player and they are not the same one. `ai.oracle`
asks whether it **knows the game** - every position, graded against the exact answer. This asks
whether it **can be beaten**, which is what anyone actually wants to know, and the two come apart
sharply: the tic-tac-toe network was wrong in 77 positions and still unbeatable, because a player
that does not blunder on the path it walks never arrives at the positions it would get wrong.

Tic-tac-toe answers the second question exhaustively - `ai.oracle.play_every_line` plays every line
an opponent could take it down. That is exponential in the length of a game, so Connect 4 cannot
have it, and this is the affordable substitute: a fixed sequence of opponents of increasing
strength, a fixed number of games against each, one table.

Deliberately coarse. It is not trying to be the precise instrument - the solved corpus already is
one - it is trying to say roughly how strong a player is, reproducibly, in a way that can be
compared with the same measurement taken a week later.

**Every rung is played, always.** Stopping at the first bad loss would be cheaper and would match
the metaphor, but two runs would then have played different opponents and could not be put side by
side. Tracking a player over time is most of what this is for, and the whole ladder is about two
minutes.

Two things about the openings do most of the work here, and both were measured rather than assumed.

**They are distinct.** A minimax player and a trained network are both deterministic, so two games
from the same position are the same game move for move; the variety comes entirely from starting a
few random plies in. `ai.match.play_match` draws those at random *with replacement*, which at its
defaults gives about thirty distinct openings for fifty pairs of Connect 4 - so nearly half the
games are replays that pad the tally without informing it. The ladder draws from
`ai.oracle.openings_at` instead, which enumerates every distinct position at a given ply, and it
**raises rather than repeating** if there are not enough.

**They are level.** Most openings are already won for somebody - 920 of Connect 4's 1,120 at four
plies - and a pair played from a decided opening is forced to 0.5 whenever both players convert it,
so those pairs stop saying anything as players improve. Starting only from drawn positions turned
`minimax:2` against `minimax:6` from a result indistinguishable from noise into a significant one.
See `balanced_openings`.
"""

import random
from typing import Callable, List, NamedTuple, Optional, Sequence, Tuple, Type

import log
from ai.match import MatchResult, play_match
from ai.oracle import openings_at, solve
from ai.players import describe, player
from games.base import GameState

# Games per rung. Each opening is played twice with the sides swapped, so this is half as many
# distinct games - 50 of them, for a standard error of about 0.04 on each rung's score.
GAMES = 100


class Ladder(NamedTuple):
    """The opponents a game is measured against, and where its games start."""

    rungs: Tuple[str, ...]
    opening_plies: int

    # Whether to start only from openings that are drawn with perfect play. See `balanced_openings`.
    balanced: bool = True


# The default sequence per game. Deterministic and fixed: a ladder that changed between runs would
# make its own numbers incomparable, which is the one thing it must not do.
#
# Connect 4 stops at depth 6 because that is the last cheap rung - a full self-play game costs
# 0.69s at depth 6 against 4.27s at depth 7, six times the price for one more ply. Its games start
# four plies in, where 1,120 distinct openings exist; two plies would allow only 49 and so cap an
# honest ladder at 98 games. Starting four discs down costs nothing that matters here, because the
# corpus already grades plies 0-6 exhaustively.
#
# Tic-tac-toe runs to depth 9, which is perfect play, so its top rung is a control rather than an
# opponent: a challenger that is also perfect must draw it. Three opening plies, because the whole
# game is nine and two plies leave only 24 drawn openings - one short of the 50 a full ladder needs.
LADDERS = {
    'Connect4': Ladder(
        rungs=('random', 'minimax:1', 'minimax:2', 'minimax:3', 'minimax:4', 'minimax:5',
               'minimax:6'),
        opening_plies=4,
    ),
    'TicTacToe': Ladder(
        rungs=('random', 'minimax:1', 'minimax:2', 'minimax:3', 'minimax:4', 'minimax:9'),
        opening_plies=3,
    ),
}


class Rung(NamedTuple):
    """One opponent, and how the challenger did against it."""

    spec: str
    result: MatchResult

    @property
    def beaten(self) -> bool:
        """Whether the challenger is *shown* to be better, rather than merely ahead."""
        return self.result.score > 0.5 and self.result.is_significant

    @property
    def lost(self) -> bool:
        return self.result.score < 0.5 and self.result.is_significant

    @property
    def verdict(self) -> str:
        return 'beats' if self.beaten else ('loses' if self.lost else 'level')


class Standing(NamedTuple):
    """Where a player sits on the ladder."""

    rungs: List[Rung]
    games: int
    opening_plies: int
    balanced: bool = True

    @property
    def highest_beaten(self) -> Optional[str]:
        """The strongest opponent the challenger is shown to beat, or None if it beats none."""
        beaten = [rung.spec for rung in self.rungs if rung.beaten]
        return beaten[-1] if beaten else None

    @property
    def skipped(self) -> List[str]:
        """
        Rungs *below* the highest one beaten that were not themselves beaten.

        Reported rather than smoothed over. A player that beats depth 5 while only drawing depth 4
        has not simply "reached depth 5" - Connect 4 has real odd/even depth effects, and an
        opponent that searches to an even ply can be genuinely harder for some players than one
        that searches a ply deeper. Hiding that behind the highest number would be the sort of
        summary that is tidier than the thing it summarises.
        """
        best = self.highest_beaten
        if best is None:
            return []
        return [rung.spec for rung in self.rungs[:self._index(best)] if not rung.beaten]

    def _index(self, spec: str) -> int:
        return next(i for i, rung in enumerate(self.rungs) if rung.spec == spec)

    def __str__(self) -> str:
        lines = [
            f'  {self.games} games per rung, from {self.games // 2} distinct '
            f'{"level " if self.balanced else ""}openings {self.opening_plies} plies in',
            f'  {"opponent":<22}{"score":>16}  {"record":<18} verdict',
        ]
        for rung in self.rungs:
            result = rung.result
            record = f'+{result.wins} ={result.draws} -{result.losses}'
            lines.append(
                f'  {rung.spec:<22}{result.score:>7.3f} +/- {result.error:.3f}  '
                f'{record:<18} {rung.verdict}'
            )

        best = self.highest_beaten
        lines.append('')
        lines.append(f'  Highest rung beaten: {best if best else "none"}')
        if self.skipped:
            lines.append(f'  But did not beat: {", ".join(self.skipped)} — not a clean ladder')
        return '\n'.join(lines)


def balanced_openings(game: Type[GameState], states: Sequence[GameState]) -> List[GameState]:
    """
    Those openings that are drawn with perfect play, where the game can say which are.

    Most openings are not drawn, and that matters more than it sounds. At four plies, 920 of Connect
    4's 1,120 positions are already won for one side, and a pair played from a decided opening is
    forced to 0.5 whenever *both* players convert it - so as players get stronger, those pairs stop
    carrying any information at all and merely drag every score toward level.

    Starting level instead is worth a lot, and it was measured rather than assumed. Over the same
    fifty pairs:

        minimax:2 vs minimax:6    all openings 0.425 +/- 0.045 (noise)
                                drawn only    0.390 +/- 0.042 (significant)
        minimax:4 vs minimax:6    all openings 0.390 +/- 0.044
                                drawn only    0.315 +/- 0.041

    A comparison that could not be told from noise becomes one that can. Depth 2 against depth 4
    stays level either way, which is not the harness failing - the position benchmark says the same
    thing, and Connect 4 has real odd/even depth effects.

    Two sources of truth, and neither is new. A game small enough to solve gets solved outright; a
    game with a corpus of exactly-solved positions gets looked up in it, which is how Connect 4
    knows the value of a position four plies in that its own solver could not reach. A game with
    neither keeps all its openings, because an unbalanced ladder still ranks players - it just
    needs more games to do it.
    """
    from ai import corpus  # Local: corpus imports nothing from here, and this keeps it that way

    if game.SOLVED_DEPTH is not None:
        return [state for state in states if solve(state) == 0]

    path = corpus.CORPORA.get(game.__name__)
    if path is None:
        log.warning(f'  {game.__name__} cannot say which openings are level; using all of them')
        return list(states)

    values = corpus.values(corpus.load(path))
    drawn = []
    for state in states:
        try:
            worths = values(state)
        except KeyError:  # Deeper than the corpus enumerates; nothing to filter on
            return list(states)
        if max(worths.values()) == 0:
            drawn.append(state)
    return drawn


def climb(
    game: Type[GameState],
    challenger: Callable,
    ladder: Optional[Ladder] = None,
    games: int = GAMES,
    seed: int = 0,
    print_progress: bool = True,
    engine: str = 'auto',
) -> Standing:
    """
    Plays `challenger` against every rung and reports where it sits.

    The same openings are used for every rung, which is deliberate: it is a paired comparison down
    the whole ladder, so a rung looking harder than the one below it cannot be an artefact of the
    positions it happened to be given.

    `engine` picks which alpha-beta the `minimax:` rungs search with - `'auto'` (the default)
    takes the Rust one where it is built, which is most of what a Connect 4 ladder costs. It has
    nothing to say about the challenger, which is whatever `challenger` already is.
    """
    ladder = ladder or for_game(game)
    openings = openings_at(game, ladder.opening_plies)
    if ladder.balanced:
        openings = balanced_openings(game, openings)

    pairs = games // 2
    if len(openings) < pairs:
        raise ValueError(
            f'{game.__name__} has {len(openings)} usable openings {ladder.opening_plies} plies in'
            f'{" that are drawn" if ladder.balanced else ""}, and {games} games needs {pairs}. '
            f'Start the games deeper, or play fewer: repeating an opening between deterministic '
            f'players replays the game rather than playing a new one.'
        )

    # Shuffled so that a smaller `games` is a spread across the openings rather than the first
    # handful, which enumeration produces in a systematic order.
    chosen = list(openings)
    random.Random(seed).shuffle(chosen)
    chosen = chosen[:pairs]

    rungs = []
    for spec in ladder.rungs:
        if print_progress:
            log.info(f'  vs {describe(spec)}...')
        result = play_match(
            game, challenger, player(spec, engine=engine),
            games=games, seed=seed, print_summary=False, openings=chosen,
        )
        rungs.append(Rung(spec, result))

    return Standing(rungs, games, ladder.opening_plies, ladder.balanced)


def for_game(game: Type[GameState]) -> Ladder:
    """The default ladder for a game, or a complaint naming the games that have one."""
    try:
        return LADDERS[game.__name__]
    except KeyError:
        raise SystemExit(
            f'no ladder is defined for {game.__name__}. Games with one: '
            f'{", ".join(sorted(LADDERS))}'
        ) from None


def make(rungs: Sequence[str], opening_plies: int, balanced: bool = True) -> Ladder:
    """A ladder from specs given on a command line, for overriding the default sequence."""
    return Ladder(tuple(rungs), opening_plies, balanced)
