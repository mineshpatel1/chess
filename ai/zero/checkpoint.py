"""
Saving a network so that something else can load it without being told what it is.

A bare `state_dict` is not enough: rebuilding the module needs the plane shape, the action count
and the layer widths, and a file that carries weights but not its own shape is one refactor away
from being unloadable. So a checkpoint carries its configuration, and `load` rebuilds from that
rather than from whatever the caller happened to assume.

The game name travels too. It is not needed to rebuild the network, but it is needed to refuse:
a tic-tac-toe network loaded against Connect 4 would produce nine plausible-looking logits for a
seven-column board, and the failure would show up as bad play rather than as an error.
"""

import os
from typing import Any, Dict, Optional

import torch

from ai.zero.net import ZeroNet

FORMAT = 1


def save(
    net: ZeroNet,
    path: str,
    game: str,
    generation: int = 0,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Writes `net` and everything needed to rebuild and identify it."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)

    torch.save(
        {
            'format': FORMAT,
            'game': game,
            'generation': generation,
            'config': net.config,
            'weights': net.state_dict(),
            'metadata': metadata or {},
        },
        path,
    )


def load(path: str, game: Optional[str] = None) -> Dict[str, Any]:
    """
    Rebuilds a network from a checkpoint, refusing one that was trained on another game.

    `weights_only=False` because the payload is a config dict as well as tensors. That is only
    safe for files you produced yourself, which is the case here - a checkpoint is not a document
    format and should not be loaded from anywhere untrusted.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f'no checkpoint at {path}')

    blob = torch.load(path, map_location='cpu', weights_only=False)
    if blob.get('format') != FORMAT:
        raise ValueError(f'{path} is checkpoint format {blob.get("format")}, expected {FORMAT}')

    if game is not None and blob['game'] != game:
        raise ValueError(
            f'{path} was trained on {blob["game"]}, not {game} - the action spaces do not match'
        )

    net = ZeroNet(**blob['config'])
    net.load_state_dict(blob['weights'])
    net.eval()

    blob['net'] = net
    return blob
