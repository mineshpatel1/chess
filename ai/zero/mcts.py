"""
PUCT search: the tree half of AlphaZero.

Three properties of this file are load-bearing, and each one is a mistake that a previous attempt
in this project actually made.

**A node is a path, not a position.** Children live in a dict on their parent and are created when
that parent is expanded, so a position reached by two different move orders is two nodes. The 2021
version keyed one flat dict by position id, and since 97% of tic-tac-toe positions inside five
plies are reachable more than one way, 28% of expansions overwrote a live node - resetting its
statistics and re-pointing its parent. Measured against a perfect opponent, that search got
*worse* as simulations increased. Nothing here can develop that fault, because there is nowhere
for two paths to meet.

**Backup carries no anchor.** `_simulate` returns a value from the point of view of the player to
move at the node it was called on, and the caller negates it. There is no "which player is this
relative to" variable to get wrong, which is what the 2021 code got wrong: it anchored to whoever
happened to be on move at the expanded leaf, so the sign was right or inverted depending on the
parity of the depth at which expansion occurred.

**There are no rollouts.** A leaf is evaluated by asking the evaluator, which in training is the
network's value head. Replacing random playouts with a learned value is most of what separates
AlphaZero from classical MCTS, and the 2021 code kept the rollouts - so its value head, bugs and
all, was never consulted by the search at all.

Nothing here imports PyTorch. The evaluator is injected, so the search can be tested against a
perfect oracle with no network and no dependencies - which is exactly the test whose absence let
the 2021 search stay broken for months.
"""

import math
import random
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

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

        Unvisited reads as 0 - neutral - rather than as anything cleverer. Optimism here makes the
        search re-try losing moves and pessimism makes it ignore unexplored ones; the prior and
        the exploration term are what are supposed to decide that.
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

    Losing is the worst thing that can happen to the side it happens to, which is the same
    convention `ai.search.terminal_score` uses and the same one the training targets use. One
    convention everywhere is the cheapest way to never get a sign wrong.
    """
    result = state.result
    if result is None:
        raise ValueError('not a finished position')
    if result.winner is None:
        return 0.0
    return 1.0 if result.winner == state.turn else -1.0


class MCTS:
    """
    A PUCT search over a game, driven by an injected evaluator.

    The evaluator is the only thing the search knows about networks, which is deliberate: swap in
    a perfect oracle and this must play perfectly, and that test needs no training and no torch.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        encoder: Encoder,
        simulations: int = SIMULATIONS,
        exploration: float = EXPLORATION,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.evaluator = evaluator
        self.encoder = encoder
        self.simulations = simulations
        self.exploration = exploration
        self.rng = rng or random.Random()

    def search(self, state: GameState, noise: bool = False) -> Result:
        """
        Runs the simulations and reports what they found.

        `noise` adds Dirichlet noise to the root priors and belongs to self-play only. At
        evaluation it would make the player worse for no reason: its whole purpose is to make
        repeated self-play games differ, and a benchmark wants the player's actual opinion.
        """
        root = Node(prior=1.0)

        # Expanding counts as the root's first visit, exactly as it does for any other node
        # inside `_simulate`. Without this the root's visit count is still zero when the first
        # simulation selects, `sqrt(N_parent)` zeroes the exploration term for every child, and
        # the search picks the first move generated while ignoring the priors entirely - so a
        # one-simulation search is blind no matter how good the network is.
        root.value_sum = self._expand(state, root)
        root.visits = 1

        if noise:
            self._add_noise(root)

        for _ in range(self.simulations):
            self._simulate(state, root)

        visits = {move: child.visits for move, child in root.children.items()}
        return Result(
            move=self._most_visited(visits),
            policy=self._policy(visits),
            value=root.value,
            visits=visits,
        )

    # ---- the tree ----------------------------------------------------------------------

    def _expand(self, state: GameState, node: Node) -> float:
        """
        Asks the evaluator about a position and hangs its legal moves off `node` as children.

        Returns the position's value to its own mover, which is what the caller backs up. Priors
        are renormalised over the legal moves only: the evaluator is free to spend mass on moves
        that do not exist here, and the search must not.
        """
        priors, value = self.evaluator(state)

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
        The mix is `(1 - eps) * prior + eps * noise`, the way round AlphaZero has it - the 2021
        code had the two weights swapped, so what it called a prior was mostly noise.
        """
        if not root.children:
            return
        noise = [self.rng.gammavariate(DIRICHLET_ALPHA, 1.0) for _ in root.children]
        total = sum(noise) or 1.0

        for child, sample in zip(root.children.values(), noise):
            child.prior = (1 - DIRICHLET_EPSILON) * child.prior + DIRICHLET_EPSILON * sample / total

    def _simulate(self, state: GameState, node: Node) -> float:
        """
        One simulation, returning the value of `state` to the player to move in it.

        Recursive on purpose. The caller negates what it gets back, so the perspective flip lives
        in exactly one place and no node needs to remember whose value it is holding.
        """
        if state.is_game_over:
            value = terminal_value(state)
        elif not node.expanded:
            value = self._expand(state, node)
        else:
            move = self._select(node)
            state.make_move(move)
            value = -self._simulate(state, node.children[move])
            state.unmake_move()

        node.visits += 1
        node.value_sum += value
        return value

    def _select(self, node: Node) -> Any:
        """
        The child maximising PUCT: `Q + c * P * sqrt(N_parent) / (1 + N_child)`.

        `-child.value` because a child's value is held from *its* mover's point of view, and the
        player choosing here is the other one. Getting this negation wrong gives a search that
        walks confidently towards its own defeat.
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

        Visit count rather than mean value, which is the choice AlphaZero makes and the 2021 code
        did not. A child visited twice with a lucky result has a wonderful mean and means nothing;
        the visit count is what the search actually committed to, and it is the more robust
        statistic precisely because PUCT only spends visits on moves that keep looking good.
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
