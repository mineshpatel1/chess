"""
PUCT search: the tree half of AlphaZero.

Three properties are load-bearing:

**A node is a path, not a position.** Children live in a dict on their parent, so a position
reached by two move orders is two nodes and there is nowhere for two paths to collide.

**Backup carries no anchor.** `_simulate` returns the value to the player to move at the node it
was called on and the caller negates it, so there is no "relative to whom" to get wrong.

**There are no rollouts.** A leaf is evaluated by the injected evaluator, which in training is the
network's value head. Nothing here imports PyTorch, so the search can be tested against a perfect
oracle with no network at all.
"""

import math
import random
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

from games.base import Encoder, GameState

# How much the search trusts the prior over what it has actually tried. AlphaZero's c_puct.
EXPLORATION = 1.5

# Root noise during self-play, so repeated games explore different openings. Never at evaluation.
DIRICHLET_ALPHA = 1.0
DIRICHLET_EPSILON = 0.25

SIMULATIONS = 200

# An evaluator answers: priors over every action, and the value of this position to its mover.
Evaluator = Callable[[GameState], Tuple[Sequence[float], float]]


class Node:
    """One edge of the tree, and everything the search knows about it."""

    __slots__ = ('prior', 'visits', 'value_sum', 'children')

    def __init__(self, prior: float) -> None:
        self.prior = prior
        self.visits = 0
        self.value_sum = 0.0
        self.children: Dict[Any, 'Node'] = {}

    @property
    def expanded(self) -> bool:
        return bool(self.children)

    @property
    def value(self) -> float:
        """
        Mean value, from the point of view of the player to move *at this node*.

        Unvisited reads as neutral: the prior and the exploration term are what decide whether to
        try an unvisited move, not an optimistic or pessimistic default here.
        """
        return self.value_sum / self.visits if self.visits else 0.0


class Result(NamedTuple):
    """What a search produced."""

    move: Any
    policy: List[float]  # Visit counts over the whole action space, normalised
    value: float  # The root's value, to the player to move there
    visits: Dict[Any, int]


def terminal_value(state: GameState) -> float:
    """
    The value of a finished position to the player whose turn it would be.

    The same convention as `ai.search.terminal_score` and the training targets: one convention
    everywhere is the cheapest way to never get a sign wrong.
    """
    result = state.result
    if result is None:
        raise ValueError('not a finished position')
    if result.winner is None:
        return 0.0
    return 1.0 if result.winner == state.turn else -1.0


def drive(steps: Iterator[GameState], evaluator: Evaluator) -> Result:
    """
    Runs a suspended search to completion, answering it one position at a time.

    `ai.zero.selfplay` has the other driver: many searches advanced together and answered in one
    batched pass. Neither contains tree logic, so the search cannot behave differently under them.
    """
    try:
        request = next(steps)
        while True:
            request = steps.send(evaluator(request))
    except StopIteration as finished:
        return finished.value


class MCTS:
    """
    A PUCT search over a game, driven by an injected evaluator.

    The evaluator is the only thing the search knows about networks: swap in a perfect oracle and
    this must play perfectly, which is a test needing no training and no torch.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        encoder: Encoder,
        simulations: int = SIMULATIONS,
        exploration: float = EXPLORATION,
        dirichlet_alpha: float = DIRICHLET_ALPHA,
        dirichlet_epsilon: float = DIRICHLET_EPSILON,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.evaluator = evaluator
        self.encoder = encoder
        self.simulations = simulations
        self.exploration = exploration
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.rng = rng or random.Random()

    def search(self, state: GameState, noise: bool = False) -> Result:
        """
        Runs the simulations and reports what they found.

        `noise` adds Dirichlet noise to the root priors and belongs to self-play only: its purpose
        is to make repeated self-play games differ, and a benchmark wants the player's real opinion.
        """
        return drive(self.steps(state, noise), self.evaluator)

    def steps(self, state: GameState, noise: bool = False) -> 'Iterator[GameState]':
        """
        The search as a generator: it yields each position it needs evaluated and is sent back
        `(priors, value)`. Returns a `Result` when the simulations are done.

        Suspendable so that many *separate* games can be advanced together and their pending
        positions evaluated in one batched pass. One tree's own simulations cannot be batched -
        each goes where the previous ones' statistics send it, and collecting several leaves at
        once needs virtual loss, which changes what the tree explores. Batching between games
        disturbs nothing, and each tree comes out identical to one searched alone.

        The state yielded is live and is mutated on resumption, so a driver must read everything it
        needs from every pending position before resuming any of them.
        """
        root = Node(prior=1.0)

        # Expanding counts as the root's first visit, as it does for any node inside `_simulate`.
        # At zero visits `sqrt(N_parent)` zeroes the exploration term for every child and the
        # first selection ignores the priors entirely.
        root.value_sum = yield from self._expand(state, root)
        root.visits = 1

        if noise:
            self._add_noise(root)

        for _ in range(self.simulations):
            yield from self._simulate(state, root)

        visits = {move: child.visits for move, child in root.children.items()}
        return Result(
            move=self._most_visited(visits),
            policy=self._policy(visits),
            value=root.value,
            visits=visits,
        )

    # ---- the tree ----------------------------------------------------------------------

    def _expand(self, state: GameState, node: Node) -> 'Iterator[GameState]':
        """
        Asks the evaluator about a position and hangs its legal moves off `node` as children.

        Returns the position's value to its own mover, which is what the caller backs up. Priors
        are renormalised over the legal moves only: the evaluator may spend mass on moves that do
        not exist here, and the search must not.
        """
        priors, value = yield state

        moves = list(state.legal_moves)
        weights = [max(priors[self.encoder.action_index(move)], 0.0) for move in moves]
        total = sum(weights)
        if total <= 0:  # An evaluator with no opinion, or one that masked everything away
            weights = [1.0] * len(moves)
            total = float(len(moves))

        for move, weight in zip(moves, weights):
            node.children[move] = Node(prior=weight / total)
        return value

    def _add_noise(self, root: Node) -> None:
        """
        Dirichlet noise over the root's children, so self-play does not play one game repeatedly.

        Sampled from gamma variates rather than numpy, which keeps this module dependency-free.
        """
        if not root.children:
            return
        noise = [self.rng.gammavariate(DIRICHLET_ALPHA, 1.0) for _ in root.children]
        total = sum(noise) or 1.0

        for child, sample in zip(root.children.values(), noise):
            child.prior = (1 - DIRICHLET_EPSILON) * child.prior + DIRICHLET_EPSILON * sample / total

    def _simulate(self, state: GameState, node: Node) -> 'Iterator[GameState]':
        """
        One simulation, returning the value of `state` to the player to move in it.

        Recursive on purpose: the caller negates what it gets back, so the perspective flip lives
        in one place and no node has to remember whose value it holds. `yield from` carries an
        evaluator request out through every frame and the answer back to where it was asked for.
        """
        if state.is_game_over:
            value = terminal_value(state)
        elif not node.expanded:
            value = yield from self._expand(state, node)
        else:
            move = self._select(node)
            state.make_move(move)
            value = -(yield from self._simulate(state, node.children[move]))
            state.unmake_move()

        node.visits += 1
        node.value_sum += value
        return value

    def _select(self, node: Node) -> Any:
        """
        The child maximising PUCT: `Q + c * P * sqrt(N_parent) / (1 + N_child)`.

        `-child.value` because a child holds its value from *its* mover's point of view, and the
        player choosing here is the other one.
        """
        best_score, best_move = -math.inf, None
        root_visits = math.sqrt(node.visits) if node.visits else 0.0

        for move, child in node.children.items():
            exploit = -child.value if child.visits else 0.0
            explore = self.exploration * child.prior * root_visits / (1 + child.visits)
            score = exploit + explore

            if score > best_score:
                best_score, best_move = score, move
        return best_move

    # ---- reading the tree --------------------------------------------------------------

    def _policy(self, visits: Dict[Any, int]) -> List[float]:
        """Visit counts as a distribution over the whole action space: the training target."""
        policy = [0.0] * self.encoder.POLICY_SIZE
        total = sum(visits.values())
        if not total:
            return policy

        for move, count in visits.items():
            policy[self.encoder.action_index(move)] = count / total
        return policy

    def _most_visited(self, visits: Dict[Any, int]) -> Any:
        """
        The move tried most often, ties broken by generation order.

        Visit count rather than mean value: a child visited twice with a lucky result has a
        wonderful mean, while PUCT only spends visits on moves that keep looking good.
        """
        best_move, best_count = None, -1
        for move, count in visits.items():
            if count > best_count:
                best_move, best_count = move, count
        return best_move

    def sample(self, visits: Dict[Any, int], temperature: float) -> Any:
        """
        A move drawn from the visit counts, for exploration early in a self-play game.

        Temperature 0 is the greedy choice. Above it, counts are raised to `1 / temperature` and
        sampled, so a temperature of 1 is proportional to visits and lower values sharpen toward
        the best move.
        """
        if temperature <= 0:
            return self._most_visited(visits)

        moves = list(visits)
        weights = [visits[move] ** (1.0 / temperature) for move in moves]
        total = sum(weights)
        if total <= 0:
            return self.rng.choice(moves)
        return self.rng.choices(moves, weights=weights, k=1)[0]
