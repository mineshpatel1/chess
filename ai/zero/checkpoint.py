"""
Saving a network so that something else can load it without being told what it is.

A bare `state_dict` is not enough - rebuilding the module needs the plane shape, the action count
and the layer widths - so a checkpoint carries its own configuration and `load` rebuilds from that.

The game name travels in order to refuse: a tic-tac-toe network loaded against Connect 4 would
produce nine plausible logits for a seven-column board and fail as bad play rather than as an error.

The optimiser's state travels so a run can be resumed rather than restarted. Adam keeps a running
mean and variance per parameter, and dropping them leaves the first steps after a resume
unmomented, which is worth the file being three times the size.

The replay buffer does not travel *inside* a checkpoint, since this is the file that gets committed
and pushed every generation. It goes in a git-ignored file beside it - see `ai/zero/replay.py`.
"""

import os
from typing import Any, Dict, Optional

import torch

from ai.zero.net import ZeroNet

FORMAT = 2


def save(
    net: ZeroNet,
    path: str,
    game: str,
    generation: int = 0,
    metadata: Optional[Dict[str, Any]] = None,
    optimiser: Optional[torch.optim.Optimizer] = None,
) -> None:
    """
    Writes `net` and everything needed to rebuild, identify and continue it.

    Written to a temporary file and renamed into place, which is atomic within a directory: a
    reader sees either the previous checkpoint or the new one, never a half-written file.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)

    blob = {
        'format': FORMAT,
        'game': game,
        'generation': generation,
        'config': net.config,
        'weights': net.state_dict(),
        'metadata': metadata or {},
        'optimiser': optimiser.state_dict() if optimiser is not None else None,
    }

    temporary = path + '.writing'
    torch.save(blob, temporary)
    os.replace(temporary, path)


def load(path: str, game: Optional[str] = None) -> Dict[str, Any]:
    """
    Rebuilds a network from a checkpoint, refusing one that was trained on another game.

    `weights_only=False` because the payload is a config dict as well as tensors, which is only
    safe for files you produced yourself. A checkpoint is not a document format.
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
    blob.setdefault('optimiser', None)
    return blob
