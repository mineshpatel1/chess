"""
Saving a network so that something else can load it without being told what it is.

A bare `state_dict` is not enough: rebuilding the module needs the plane shape, the action count
and the layer widths, and a file that carries weights but not its own shape is one refactor away
from being unloadable. So a checkpoint carries its configuration, and `load` rebuilds from that
rather than from whatever the caller happened to assume.

The game name travels too. It is not needed to rebuild the network, but it is needed to refuse:
a tic-tac-toe network loaded against Connect 4 would produce nine plausible-looking logits for a
seven-column board, and the failure would show up as bad play rather than as an error.

**The optimiser's state travels as well, so a run can be resumed rather than restarted.** Adam
keeps a running mean and variance per parameter; dropping them restarts the moment estimates from
zero and the first steps after a resume are effectively unmomented. On a run measured in hours,
where the reason for resuming is that the machine went away, that is worth the file being three
times the size.

What deliberately does *not* travel *inside* a checkpoint is the replay buffer. A checkpoint is the
file that gets committed and pushed as a run produces it, and putting the buffer in it would put
tens of megabytes of self-play into the history every generation to save the machine that already
has them a few minutes. It goes in a git-ignored file beside the checkpoint instead, which is
`ai/zero/replay.py` - the same resume point, split by what is worth keeping forever.
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

    Written to a temporary file and moved into place, because the thing most likely to interrupt a
    long run is also most likely to interrupt the write that was meant to survive it. A rename
    within a directory is atomic, so a reader either sees the previous checkpoint or the new one
    and never a half-written file.
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
    blob.setdefault('optimiser', None)
    return blob
