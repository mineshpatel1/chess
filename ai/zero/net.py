"""
The network: a position in, a policy and a value out.

Two trunks, chosen per game by `ARCHITECTURES`; the encoder, the search and the training loop are
unaware of the difference.

**Tic-tac-toe gets dense layers.** On a 3x3 board a 3x3 kernel already sees everything, so a tower
degenerates into a dense layer wearing weight-sharing constraints - and the board is not
translation invariant anyway, the centre sitting on four winning lines to a corner's three.

**Connect 4 gets a residual tower.** A threat is a local shape - three of mine with a gap - that
means the same thing wherever it sits, which is what a shared filter encodes.

Latency is the binding constraint on both: MCTS calls this once per simulation and a training run
makes millions of forward passes, which is why `evaluate_batch` exists.
"""

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from games.base import Encoder

# Wide enough to represent a solved 3x3 game several times over, small enough to stay fast.
HIDDEN_LAYERS = (64, 64)

# Channels each head narrows the tower down to before flattening. AlphaZero's 2 and 1 are for a
# 19x19 board; on a 6x7 one that throws away too much.
HEAD_FILTERS = 32

# The trunk each game gets. Absent means the dense trunk. Connect 4's 64 filters over 5 blocks is
# the AlphaZero.jl reference scaled for a CPU, where latency binds rather than capacity: 128x5
# costs 1665us a call against 1101us for 64x5, and the search makes one call per simulation.
ARCHITECTURES = {
    'Connect4': {'filters': 64, 'blocks': 5},
}


# What a masked-out action's logit becomes before the softmax. Large and negative rather than
# -inf, which would put a NaN through the loss the moment every action in a row was masked.
MASKED = -1e9


def for_game(game_name: str) -> dict:
    """The trunk settings for a game, empty for one that wants the dense default."""
    return dict(ARCHITECTURES.get(game_name, {}))


def choose_device(requested: Optional[str] = None) -> str:
    """
    Where a training run should put the network. `None` or `'auto'` takes a CUDA card if there is
    one; anything else is passed through to torch, so an untested backend can still be asked for.

    A GPU is worth having only where the batch is. A batch of one costs 1875us on a 3080 against
    1221us on this CPU - the call is launch overhead either way, and the card loses - so the
    unbatched paths stay where they are and it is self-play, grading and gradient steps that move.
    That is a recent change: the batch used to be capped at `--games-in-flight`, and at 32 or 64
    positions there was nothing here for a card to do.
    """
    if requested not in (None, 'auto'):
        return requested
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def make_deterministic(device: str) -> None:
    """
    Keeps `--seed` meaning what it says on a GPU.

    cuDNN picks convolution algorithms by benchmarking unless told not to, and different
    algorithms give different last bits. Pinning them costs nothing measurable for a network this
    small. Runs on different devices still differ from each other; only the promise that a seed
    reproduces a run on *one* machine is being kept.
    """
    if torch.device(device).type == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def device_of(net: nn.Module) -> torch.device:
    """Where a network's weights live, which is where its inputs have to be built."""
    return next(net.parameters()).device


class Residual(nn.Module):
    """
    Two 3x3 convolutions and a skip connection, AlphaZero's block.

    The skip is what makes depth usable: a block starts as approximately the identity and has to
    earn its contribution, rather than the signal having to survive every layer's weights.
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

    The heads share the trunk by design rather than to save parameters: what makes a position
    winnable and what makes a move good are mostly the same features, so each head regularises the
    other. Which trunk a game gets is the module docstring's subject.
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

            # One 1x1 convolution per head collapses `filters` channels to a handful before
            # flattening. A dense layer over the whole tower output would be 2,688 inputs a head.
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

        Logits rather than probabilities: the legal moves are not known here, and masking has to
        happen before the softmax - renormalising afterwards gives a different distribution.
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


def to_tensor(planes_batch: Sequence, device='cpu') -> torch.Tensor:
    """Nested lists of ints from an encoder into a float batch, where the weights are."""
    return torch.tensor(planes_batch, dtype=torch.float32, device=device)


def masked_policy(logits: torch.Tensor, legal: Sequence[int], policy_size: int) -> List[float]:
    """
    A probability distribution over the legal actions only.

    Masking before the softmax rather than zeroing after it: taking mass away afterwards leaves a
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

    The reason the search and self-play were made suspendable: a Connect 4 forward pass costs
    1101us alone and 111us amortised in a batch of sixty-four.

    Positions are encoded before the pass and read for legality after it, both while every state is
    untouched - self-play hands over live states it is about to mutate.
    """
    if not states:
        return []

    net.eval()
    device = device_of(net)
    with torch.no_grad():
        logits, values = net(to_tensor([encoder.planes(state) for state in states], device))

        # The mask is built as plain Python lists and becomes a tensor exactly once. Writing it
        # element by element into a tensor is a dispatched operation per write, around a hundred
        # times slower, and enough to make a batch of thirty-two slower than a batch of one.
        rows = []
        for state in states:
            row = [MASKED] * encoder.POLICY_SIZE
            for move in state.legal_moves:
                row[encoder.action_index(move)] = 0.0
            rows.append(row)

        policies = F.softmax(
            logits + torch.tensor(rows, dtype=torch.float32, device=device), dim=1)

    return list(zip(policies.tolist(), values.tolist()))


# The smallest positive float32 with a full mantissa. Below this the exponent is already at its
# minimum and precision is bought by leading zeros in the mantissa - a *subnormal*, or denormal.
SMALLEST_NORMAL = 1.18e-38


def flush_denormals(net: ZeroNet) -> int:
    """
    Zeroes weights that have decayed into the subnormal range, returning how many there were.

    Worth up to six times the speed of the whole training loop. Weight decay pulls unused weights
    toward zero and they stall around 1e-40, which float32 can only hold as a denormal, and x86
    handles denormal arithmetic in microcode rather than the vector units. A Connect 4 network 11%
    denormal ran 6x slower than the same architecture freshly initialised.

    Numerically a no-op: 1e-40 contributes nothing any float32 sum of ~1e-2 weights can represent.

    `torch.set_flush_denormal(True)` is not an alternative - it sets the flush-to-zero flag on the
    calling thread only, and torch runs its intra-op pool on several others.
    """
    total = 0
    with torch.no_grad():
        for parameter in net.parameters():
            # Legitimately zero weights are not denormal, and there are many of them.
            tiny = (parameter != 0.0) & (parameter.abs() < SMALLEST_NORMAL)
            count = int(tiny.sum())
            if count:
                parameter[tiny] = 0.0
                total += count
    return total


def evaluate(net: ZeroNet, state, encoder: Encoder) -> Tuple[List[float], float]:
    """
    One position through the network: priors over its legal moves, and its value.

    The value is from the point of view of the player to move, matching the planes and
    `ai.search.terminal_score` - the one convention every sign in the project is anchored to.
    """
    net.eval()
    with torch.no_grad():
        logits, value = net(to_tensor([encoder.planes(state)], device_of(net)))

    legal = [encoder.action_index(move) for move in state.legal_moves]
    return masked_policy(logits[0], legal, encoder.POLICY_SIZE), float(value[0])
