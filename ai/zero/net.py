"""
The network: a position in, a policy and a value out.

Two trunks, and which one a game gets is a property of its board rather than a default copied from
a paper. `ARCHITECTURES` says which, and the encoder, the search and the training loop are all
unaware of the difference.

**Tic-tac-toe gets dense layers.** The planes are flattened into two hidden layers of 64 and fork
into a policy head of POLICY_SIZE logits and a value head of one tanh unit - about six thousand
parameters. A convolution would be reasoning by analogy: it buys translation invariance and
locality, but on a 3x3 board a 3x3 kernel already sees everything, so the tower degenerates into a
dense layer wearing weight-sharing constraints. Worse, tic-tac-toe is not translation invariant at
all - the centre sits on four winning lines, a corner on three, an edge on two - so the prior a
convolution encodes is actively the wrong one. Measured, it cost 2.5x the latency per call for no
benefit. Capacity was never the constraint either: 5,478 positions, about 765 once symmetries fold
together, and a deterministic function - this is memorising a solved game, not fitting a sample.

**Connect 4 gets a residual tower**, 64 filters and five blocks. Here the case for convolution is
real: a threat is a local shape - three of mine with a gap - and it means the same thing wherever
it sits, which is exactly what a shared filter encodes and exactly what
`games/connect4/evaluation.py` computes by hand today. The board is large enough that a dense layer
over it would have to learn each column's version of the same tactic separately.

Latency is the binding constraint on both, because MCTS calls this once per simulation and a
training run makes millions of forward passes. That is why `evaluate_batch` exists and why the
search was made suspendable to be able to use it: a Connect 4 pass costs 1101us alone and 111us
amortised in a batch of sixty-four.
"""

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from games.base import Encoder

# Wide enough to represent a solved 3x3 game several times over, small enough to stay fast.
HIDDEN_LAYERS = (64, 64)

# Channels each head narrows the tower down to before flattening. AlphaZero uses 2 for policy and
# 1 for value on a 19x19 board; on a 6x7 board that throws away too much, and 32 is what the
# Connect 4 implementations that work in public settle on.
HEAD_FILTERS = 32

# The trunk each game gets, keyed by game name, in the idiom `ai.corpus.CORPORA` and
# `ai.ladder.LADDERS` already use. Absent means the dense trunk, which is the right answer for a
# board too small for convolution to mean anything.
#
# Connect 4's numbers are the AlphaZero.jl reference for this game scaled to a machine without a
# GPU: 64 filters and 5 blocks against its 128 and 5. Latency is the binding constraint, not
# capacity - measured on this CPU, 128x5 costs 1665us a call against 1101us for 64x5, and the
# search makes one call per simulation.
ARCHITECTURES = {
    'Connect4': {'filters': 64, 'blocks': 5},
}


# What a masked-out action's logit becomes before the softmax. Large and negative rather than
# -inf, which would put a NaN through the loss the moment every action in a row was masked.
MASKED = -1e9


def for_game(game_name: str) -> dict:
    """The trunk settings for a game, empty for one that wants the dense default."""
    return dict(ARCHITECTURES.get(game_name, {}))


class Residual(nn.Module):
    """
    Two 3x3 convolutions and a skip connection, AlphaZero's block.

    The skip is what makes depth usable: without it the useful signal has to survive being
    multiplied by every layer's weights on the way through, and five blocks is enough for that to
    hurt. With it, a block starts as approximately the identity and has to earn its contribution.
    """

    def __init__(self, filters: int) -> None:
        super().__init__()
        self.first = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.first_norm = nn.BatchNorm2d(filters)
        self.second = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.second_norm = nn.BatchNorm2d(filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.first_norm(self.first(x)))
        y = self.second_norm(self.second(y))
        return F.relu(x + y)


class ZeroNet(nn.Module):
    """
    Input -> trunk -> (policy, value), the trunk being either dense layers or a residual tower.

    The two heads share the trunk, which is the point of the architecture rather than a saving:
    what makes a position winnable and what makes a move good are mostly the same features, and
    learning them once from both signals is why AlphaZero's two heads help each other.

    **Which trunk is a property of the board, not a default to copy.** Tic-tac-toe gets the dense
    one for the reasons in the module docstring - a 3x3 kernel on a 3x3 board is a dense layer
    wearing weight-sharing constraints, and the board is not translation invariant anyway. Connect
    4 gets the tower, because its threats genuinely are local shapes that mean the same thing
    wherever they sit: three of mine with a gap is the same tactic in every column, which is what
    a shared filter encodes and what `games/connect4/evaluation.py` computes by hand today.
    """

    def __init__(
        self,
        plane_shape: Tuple[int, int, int],
        policy_size: int,
        hidden: Sequence[int] = HIDDEN_LAYERS,
        filters: int = 0,
        blocks: int = 0,
        head_filters: int = HEAD_FILTERS,
    ) -> None:
        super().__init__()
        self.plane_shape = tuple(plane_shape)
        self.policy_size = policy_size
        self.hidden = tuple(hidden)
        self.filters = filters
        self.blocks = blocks
        self.head_filters = head_filters

        planes, rows, columns = self.plane_shape
        self.convolutional = bool(filters and blocks)

        if self.convolutional:
            self.stem = nn.Sequential(
                nn.Conv2d(planes, filters, 3, padding=1, bias=False),
                nn.BatchNorm2d(filters),
                nn.ReLU(),
            )
            self.tower = nn.Sequential(*[Residual(filters) for _ in range(blocks)])

            # One 1x1 convolution per head, which is the cheap way to collapse `filters` channels
            # down to a handful before flattening: the alternative is a dense layer over the whole
            # tower output, which for 64 filters on a 6x7 board is 2,688 inputs a head.
            self.policy_conv = nn.Sequential(
                nn.Conv2d(filters, head_filters, 1, bias=False),
                nn.BatchNorm2d(head_filters),
                nn.ReLU(),
            )
            self.value_conv = nn.Sequential(
                nn.Conv2d(filters, head_filters, 1, bias=False),
                nn.BatchNorm2d(head_filters),
                nn.ReLU(),
            )
            self.policy = nn.Linear(head_filters * rows * columns, policy_size)
            self.value_hidden = nn.Linear(head_filters * rows * columns, hidden[0])
            self.value = nn.Linear(hidden[0], 1)
        else:
            widths = [planes * rows * columns] + list(hidden)
            layers: List[nn.Module] = []
            for inputs, outputs in zip(widths, widths[1:]):
                layers += [nn.Linear(inputs, outputs), nn.ReLU()]
            self.trunk = nn.Sequential(*layers)

            self.policy = nn.Linear(widths[-1], policy_size)
            self.value = nn.Linear(widths[-1], 1)

    def forward(self, planes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Policy *logits* and a value in [-1, 1], both from the point of view of the player to move.

        Logits rather than probabilities because the legal moves are not known here and masking
        has to happen before the softmax, not after: renormalising a distribution that already
        spent mass on illegal moves is not the same distribution.
        """
        if not self.convolutional:
            trunk = self.trunk(planes.flatten(1))
            return self.policy(trunk), torch.tanh(self.value(trunk)).squeeze(-1)

        trunk = self.tower(self.stem(planes))
        logits = self.policy(self.policy_conv(trunk).flatten(1))
        value = F.relu(self.value_hidden(self.value_conv(trunk).flatten(1)))
        return logits, torch.tanh(self.value(value)).squeeze(-1)

    @property
    def config(self) -> dict:
        """Everything needed to rebuild this network, for the checkpoint to carry."""
        return {
            'plane_shape': list(self.plane_shape),
            'policy_size': self.policy_size,
            'hidden': list(self.hidden),
            'filters': self.filters,
            'blocks': self.blocks,
            'head_filters': self.head_filters,
        }


def to_tensor(planes_batch: Sequence, device: str = 'cpu') -> torch.Tensor:
    """Nested lists of ints from an encoder into a float batch."""
    return torch.tensor(planes_batch, dtype=torch.float32, device=device)


def masked_policy(logits: torch.Tensor, legal: Sequence[int], policy_size: int) -> List[float]:
    """
    A probability distribution over the legal actions only.

    Masking before the softmax rather than zeroing after it. The difference matters: a softmax
    over everything gives illegal moves probability mass, and taking it away afterwards leaves a
    distribution whose shape was set by moves that were never available.
    """
    mask = torch.full((policy_size,), MASKED, device=logits.device)
    mask[list(legal)] = 0.0
    return F.softmax(logits + mask, dim=-1).tolist()


def evaluate_batch(
    net: ZeroNet, states: Sequence, encoder: Encoder,
) -> List[Tuple[List[float], float]]:
    """
    Many positions in one forward pass, in the order they were given.

    The whole reason the search and self-play were made suspendable. A Connect 4 forward pass
    costs 1101us alone and 111us amortised in a batch of sixty-four: same arithmetic, ten times
    the throughput, because a batch of one leaves a four-core machine almost entirely idle while
    Python and the framework do the work of setting the call up.

    Positions are encoded before the pass and read for legality after it, both while every state
    is untouched - which matters, since self-play hands over live states it is about to mutate.
    """
    if not states:
        return []

    net.eval()
    with torch.no_grad():
        logits, values = net(to_tensor([encoder.planes(state) for state in states]))

        # Masked and softmaxed for the whole batch at once, and the mask is built as plain Python
        # lists before becoming a tensor exactly once.
        #
        # Both halves of that matter and the second one was learned the hard way. Writing the mask
        # element by element into a tensor - `mask[row][column] = 0.0` - looks like the same thing
        # and is roughly a hundred times slower per write, because each one is a dispatched tensor
        # operation rather than a list store. Done that way it made a batch of thirty-two *slower*
        # than a batch of one, which is the opposite of the entire point of this function.
        rows = []
        for state in states:
            row = [MASKED] * encoder.POLICY_SIZE
            for move in state.legal_moves:
                row[encoder.action_index(move)] = 0.0
            rows.append(row)

        policies = F.softmax(logits + torch.tensor(rows, dtype=torch.float32), dim=1)

    return list(zip(policies.tolist(), values.tolist()))


# The smallest positive float32 with a full mantissa. Below this the exponent is already at its
# minimum and precision is bought by leading zeros in the mantissa - a *subnormal*, or denormal.
SMALLEST_NORMAL = 1.18e-38


def flush_denormals(net: ZeroNet) -> int:
    """
    Zeroes weights that have decayed into the subnormal range, returning how many there were.

    **This is worth six times the speed of the whole training loop, and it cost a day to find.**

    Weight decay pulls unused weights toward zero and they do not arrive: they stall around 1e-40,
    which float32 can only represent as a denormal. x86 handles denormal arithmetic in microcode
    rather than in the vector units, so every multiply touching one costs orders of magnitude more
    than a normal multiply. By generation 8 of a Connect 4 run, 11% of the network's 471,000
    weights were denormal and it had become 6x slower than the same architecture freshly
    initialised - 0.95s against 0.16s for twenty batches of thirty-two.

    The slowdown compounds generation by generation and applies to *everything* that multiplies by
    these weights, so self-play, the gradient steps and the benchmark all slow by the same factor
    while the work they do is unchanged. That signature - fixed work getting uniformly slower - is
    what to recognise, and it is nothing to do with the machine, which is where a day went.

    Numerically this changes nothing. A weight of 1e-40 in a network whose meaningful weights are
    around 1e-2 contributes nothing any float32 sum can represent; `tests/zero/test_net.py` asserts
    the outputs are unchanged.

    `torch.set_flush_denormal(True)` is **not** an alternative. It sets the CPU's flush-to-zero
    flag on the calling thread only, and torch runs its intra-op pool on several others - measured
    here, it made no difference at all (0.93s against 0.97s) while this made it 0.16s.
    """
    total = 0
    with torch.no_grad():
        for parameter in net.parameters():
            # Non-zero matters: without it this counts every legitimately zero weight and reports
            # a five-figure denormal count for a network that has none.
            tiny = (parameter != 0.0) & (parameter.abs() < SMALLEST_NORMAL)
            count = int(tiny.sum())
            if count:
                parameter[tiny] = 0.0
                total += count
    return total


def evaluate(net: ZeroNet, state, encoder: Encoder) -> Tuple[List[float], float]:
    """
    One position through the network: priors over its legal moves, and its value.

    The value is from the point of view of the player to move, matching the planes and matching
    `ai.search.terminal_score`. Every sign in the search and in the training targets is anchored
    to that one convention, which is the cheapest way to never get a sign wrong.
    """
    net.eval()
    with torch.no_grad():
        logits, value = net(to_tensor([encoder.planes(state)]))

    legal = [encoder.action_index(move) for move in state.legal_moves]
    return masked_policy(logits[0], legal, encoder.POLICY_SIZE), float(value[0])
