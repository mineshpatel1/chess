"""
The network: a position in, a policy and a value out.

An MLP, and deliberately a small one - the planes are flattened into two hidden layers of 64 and
the trunk forks into a policy head of POLICY_SIZE logits and a value head of one tanh unit. About
six thousand parameters for tic-tac-toe.

AlphaZero uses a deep residual convolutional tower, and copying that here would be reasoning by
analogy rather than about the problem. Convolution buys translation invariance and locality; on a
3x3 board a 3x3 kernel already sees everything, so the tower degenerates into a dense layer
wearing weight-sharing constraints. Worse, tic-tac-toe is not translation invariant in the first
place - the centre sits on four winning lines, a corner on three, an edge on two - so the prior a
convolution encodes is actively the wrong one. Measured, the convolutional version cost 2.5x the
latency per call for no benefit, and latency is the whole game here: MCTS calls this once per
simulation, so a training run makes millions of forward passes and nothing else comes close in
cost.

Capacity is not the constraint either. Tic-tac-toe has 5,478 reachable positions, about 765 once
the eight symmetries are folded together, and the function being learned is deterministic - we
are not fitting a sample of a noisy process, we are memorising a solved game. Six thousand
parameters is ample for that.

The encoder still hands over *planes* rather than a flat vector, so a game whose board really
does reward convolution can have a different trunk without the encoder, the search or the
training loop noticing. Layer sizes are configuration, not constants.
"""

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from games.base import Encoder

# Wide enough to represent a solved 3x3 game several times over, small enough to stay fast.
HIDDEN_LAYERS = (64, 64)

# What a masked-out action's logit becomes before the softmax. Large and negative rather than
# -inf, which would put a NaN through the loss the moment every action in a row was masked.
MASKED = -1e9


class ZeroNet(nn.Module):
    """
    Input -> hidden -> hidden -> (policy, value).

    The two heads share the trunk, which is the point of the architecture rather than a saving:
    what makes a position winnable and what makes a move good are mostly the same features, and
    learning them once from both signals is why AlphaZero's two heads help each other.
    """

    def __init__(
        self,
        plane_shape: Tuple[int, int, int],
        policy_size: int,
        hidden: Sequence[int] = HIDDEN_LAYERS,
    ) -> None:
        super().__init__()
        self.plane_shape = tuple(plane_shape)
        self.policy_size = policy_size
        self.hidden = tuple(hidden)

        planes, rows, columns = self.plane_shape
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
        trunk = self.trunk(planes.flatten(1))
        return self.policy(trunk), torch.tanh(self.value(trunk)).squeeze(-1)

    @property
    def config(self) -> dict:
        """Everything needed to rebuild this network, for the checkpoint to carry."""
        return {
            'plane_shape': list(self.plane_shape),
            'policy_size': self.policy_size,
            'hidden': list(self.hidden),
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
