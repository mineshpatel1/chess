"""
The replay buffer on disk, so an interrupted run resumes with its data as well as its weights.

It lives beside the checkpoint as `<latest>.buffer` and is git-ignored, which is the whole split:
the checkpoint is small and is what `--commit-every` pushes to the branch, while the buffer is
large and worth nothing to any machine but this one. A resume on a fresh clone finds no buffer and
behaves as it would have without this file - a few generations of learning from less data than
usual, during which the score climbs back to where it already was.

Stored as three stacked tensors rather than as the examples themselves, which for a default
20,000-position Connect 4 buffer is 2.32MB and 0.09s against 7.64MB and 0.44s pickled whole.

Planes are int8, matching what encoders produce (0/1 or -1/0/+1). The policy is float32, which is
the precision the gradient step converts it to anyway.
"""

import os
from typing import Any, List, Sequence

import torch

import log
from ai.zero.selfplay import Example

FORMAT = 1

SUFFIX = '.buffer'


def path_for(checkpoint_path: str) -> str:
    """
    Where the buffer beside a checkpoint lives: `models/connect4-latest.pt` -> `...-latest.buffer`.

    Derived rather than configured: the buffer is part of the resume point, not a separate thing to
    remember to pass on a relaunch.
    """
    root, _ = os.path.splitext(checkpoint_path)
    return root + SUFFIX


def save(buffer: Sequence[Example], path: str, game: str, generation: int = 0) -> None:
    """
    Writes the buffer, temporary-file-and-rename as `ai.zero.checkpoint.save` does, so a reader
    never sees a half-written one.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)

    blob = {
        'format': FORMAT,
        'game': game,
        'generation': generation,
        'examples': len(buffer),
        'planes': torch.tensor([example.planes for example in buffer], dtype=torch.int8),
        'policy': torch.tensor([list(example.policy) for example in buffer], dtype=torch.float32),
        'value': torch.tensor([example.value for example in buffer], dtype=torch.float32),
    }

    temporary = path + '.writing'
    torch.save(blob, temporary)
    os.replace(temporary, path)


def load(path: str, game: str, encoder: Any) -> List[Example]:
    """
    The examples back, or none of them and a logged reason why.

    Nothing here raises: a bad buffer costs a few generations of refilling, and refusing to start
    would trade that for the outage the buffer exists to prevent. It warns rather than passing
    silently, since a buffer that stopped being restored looks like a run that got worse at
    learning.

    The shape check is the one that earns its keep. A buffer from another game is caught by its
    name, but one written before an encoder changed has the right name and planes that no longer
    mean what they say.
    """
    if not os.path.exists(path):
        return []

    try:
        blob = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as error:  # pragma: no cover - depends on how the write was interrupted
        log.warning(f'could not read the replay buffer at {path} ({error}); starting empty')
        return []

    if blob.get('format') != FORMAT:
        log.warning(f'{path} is replay format {blob.get("format")}, expected {FORMAT}; '
                    f'starting with an empty buffer')
        return []
    if blob.get('game') != game:
        log.warning(f'{path} holds {blob.get("game")} positions, not {game}; '
                    f'starting with an empty buffer')
        return []

    planes, policy, value = blob['planes'], blob['policy'], blob['value']
    shape = tuple(planes.shape[1:])
    if len(planes) and (shape != tuple(encoder.PLANE_SHAPE)
                        or policy.shape[1] != encoder.POLICY_SIZE):
        log.warning(f'{path} holds {shape} planes and {policy.shape[1]} actions, but this '
                    f'encoder produces {tuple(encoder.PLANE_SHAPE)} and {encoder.POLICY_SIZE}; '
                    f'starting with an empty buffer')
        return []

    return [
        Example(one_planes, one_policy, one_value)
        for one_planes, one_policy, one_value
        in zip(planes.tolist(), policy.tolist(), value.tolist())
    ]
