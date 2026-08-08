"""
Generating training data by playing the network against itself.

Each position played produces one example: the board as its mover saw it, the search's visit
counts as the policy to imitate, and - once the game is finished - what the game was worth to
that mover. The network is then trained to predict, in one position, what the search concluded
after looking ahead from it. That is the whole trick: search is slow and better than the network,
so the network chases it, and the better network makes the next search better still.

The value target is where the previous attempt in this project went wrong, twice over, so it is
worth being explicit about what is correct here:

* **From the mover's own point of view.** A position where the player to move went on to win is
  `+1` *for that position*, whichever player it was. The 2021 code compared each position's mover
  against `turn` after the final move - which, since the turn flips on the winning move too, is
  the *loser* - and labelled every position backwards.
* **A draw is 0, not a sign.** Tic-tac-toe is a draw under any decent play, so most self-play
  games end level and most examples should say so. The 2021 code had no zero case at all and
  labelled drawn games `+1`/`-1` alternately, teaching the network that a drawn position is won
  for whoever happens to be on move.
"""

import random
from typing import Any, Callable, List, NamedTuple, Optional, Sequence, Tuple

from games.base import Encoder, GameState
from ai.zero.mcts import DIRICHLET_ALPHA, DIRICHLET_EPSILON, EXPLORATION, MCTS, Evaluator

# Plies played by sampling from the visit counts rather than taking the best move. Without this
# every self-play game from a deterministic network is the same game, and the buffer fills with
# one line repeated.
#
# AlphaZero's number, which for a game of 200 moves is an opening and for tic-tac-toe is all nine
# plies. Sampling the whole of a short game is what it should mean: measured here, cutting it to
# three plies cost 10 points of agreement with perfect play (83.8% against 93.7%), because a
# nine-ply game played greedily from ply four is very nearly one game repeated.
TEMPERATURE_MOVES = 30
TEMPERATURE = 1.0

# Random plies played before a game starts being recorded, drawn uniformly from 0 to this.
#
# **Off by default, and the history of that is worth keeping.** Self-play is on-policy, so a
# network that has learned to play well stops visiting anything else - measured here, it reached
# 366 of the game's 4,520 decision positions. Forcing a share of games to start somewhere random
# fixed that, and looked like the single largest improvement in the implementation: 80.3% -> 96.8%.
#
# It was not. It was covering for `EXPLORATION` being set too low. Raising c_puct from 1.5 to 5.0
# reached 97.5% with this at zero - the same result, no crutch, and a better value head with it.
# A correct AlphaZero explores through PUCT and root noise; needing random starts on top is a sign
# that one of those is mistuned, not a technique to reach for.
#
# Kept because it is genuinely useful when a *benchmark* wants coverage of positions real play
# never reaches, which is a different goal from training a player.
OPENING_PLIES = 0


class Example(NamedTuple):
    """One position, and what the search and the eventual result had to say about it."""

    planes: Any
    policy: Sequence[float]
    value: float


def play_game(
    evaluator: Evaluator,
    encoder: Encoder,
    game: Callable[[], GameState],
    simulations: int,
    temperature_moves: int = TEMPERATURE_MOVES,
    temperature: float = TEMPERATURE,
    opening_plies: int = 0,
    exploration: float = EXPLORATION,
    dirichlet_alpha: float = DIRICHLET_ALPHA,
    dirichlet_epsilon: float = DIRICHLET_EPSILON,
    rng: Optional[random.Random] = None,
) -> Tuple[List[Example], GameState]:
    """
    Plays one game, returning an example per position and the finished game itself.

    The finished state comes back so a caller can report on the games as well as learn from them
    - how many were drawn is the single most informative number in tic-tac-toe self-play, since
    a healthy run converges on drawing almost everything.

    `opening_plies` puts the game somewhere random before any of it is recorded. Nothing played
    during those plies becomes an example, so every training target still comes from a real
    search - the randomness moves where the games happen, not what is learned from them.
    """
    rng = rng or random.Random()
    mcts = MCTS(
        evaluator, encoder, simulations=simulations, exploration=exploration,
        dirichlet_alpha=dirichlet_alpha, dirichlet_epsilon=dirichlet_epsilon, rng=rng,
    )

    state = _opening(game, opening_plies, rng)
    played: List[Tuple[Any, Sequence[float], bool]] = []

    while not state.is_game_over:
        result = mcts.search(state, noise=True)
        tau = temperature if len(played) < temperature_moves else 0.0

        # Recorded before the move: the example belongs to the position it was searched from.
        played.append((encoder.planes(state), result.policy, state.turn))
        state.make_move(mcts.sample(result.visits, tau))

    outcome = state.result
    examples = [
        Example(planes, policy, _value_to(mover, outcome.winner))
        for planes, policy, mover in played
    ]
    return examples, state


def _opening(game: Callable[[], GameState], plies: int, rng: random.Random) -> GameState:
    """
    A position a random number of random plies in, and still running.

    The count is drawn per game rather than fixed, so the openings land at every depth instead of
    piling up at one. A game that finishes inside its own opening is discarded and redrawn -
    there is nothing to learn from a position with no moves in it.
    """
    if plies <= 0:
        return game()

    for _ in range(50):
        state = game()
        for _ in range(rng.randint(0, plies)):
            if state.is_game_over:
                break
            state.make_move(rng.choice(list(state.legal_moves)))

        if not state.is_game_over:
            return state
    return game()


def _value_to(mover: bool, winner: Optional[bool]) -> float:
    """What a finished game was worth to the player who was about to move."""
    if winner is None:
        return 0.0
    return 1.0 if winner == mover else -1.0


def augment(examples: Sequence[Example], encoder: Encoder) -> List[Example]:
    """
    Every example under every symmetry the game admits.

    Eight-fold for tic-tac-toe, which is eight times the data for the cost of a list shuffle, and
    it also stops the network learning that one corner differs from another merely because
    self-play visited it more often. The value is carried through untouched: rotating a board
    does not change who is winning.
    """
    grown: List[Example] = []
    for example in examples:
        for planes, policy in encoder.symmetries(example.planes, list(example.policy)):
            grown.append(Example(planes, policy, example.value))
    return grown
