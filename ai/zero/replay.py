"""
The replay buffer on disk, so that an interrupted run resumes with its data as well as its weights.

A checkpoint carries the network and the optimiser, which is what it takes to *continue*; it does
not carry the buffer, and `ai/zero/checkpoint.py` explains why it should not. But a resume that
starts with an empty buffer trains its first generation on a fraction of the usual data and gets
measurably worse for it - about three generations of a Connect 4 run, during which the ladder score
climbs back to where it already was and looks exactly like progress. A run that is interrupted
often enough spends most of its life in that recovery.

So the buffer travels beside the checkpoint rather than inside it, as `<latest>.buffer`, and it is
git-ignored. That split is the whole idea: the checkpoint is small, is the thing worth committing
and pushing, and is what `--commit-every` copies to the branch; the buffer is large, is worth
nothing to anybody but this machine, and never goes near the history. A resume on a fresh clone
finds the checkpoint and no buffer and behaves exactly as it did before this file existed.

**Stored as three tensors rather than as the examples themselves.** An `Example` holds nested lists
of small ints, a policy and a float, and `torch.save` on a default 20,000-position Connect 4 buffer
is 7.64MB and 0.44s. Stacked into an int8 plane tensor and two float32 tensors it is **2.32MB and
0.09s**, because that is three contiguous buffers rather than nearly two million boxed Python
objects. This is written every generation beside a checkpoint that already costs more than either,
so neither number is a problem - but the tensor form is small enough to stop being a consideration
at all, which is what a buffer that scales with `--games` needs it to be.

The planes are int8 because that is what encoders produce: 0/1 for Connect 4's two binary planes
and -1/0/+1 for tic-tac-toe's one signed plane. The policy is float32, which is a rounding of the
visit-count fractions that were float64 - and is exactly the precision they are converted to in the
gradient step anyway, so nothing is lost that was ever used.
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

    Derived rather than configured, because the buffer is not a separate thing to keep track of -
    it is part of the resume point, and a flag for it would be one more thing to remember to pass
    on the relaunch, which is the moment you least want to be composing a new command line.
    """
    root, _ = os.path.splitext(checkpoint_path)
    return root + SUFFIX


def save(buffer: Sequence[Example], path: str, game: str, generation: int = 0) -> None:
    """
    Writes the buffer, temporary-file-and-rename as `ai.zero.checkpoint.save` does and for the
    same reason: the thing most likely to interrupt a long run is the write meant to survive it,
    and a half-written buffer that loads as garbage would be worse than no buffer at all.

    An empty buffer still writes a file. It means "this run has nothing yet" rather than "there is
    nothing here", and the difference shows up on a resume of a run killed in its first generation.
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
    The examples back, or none of them and a reason why.

    **Nothing here raises.** Every failure mode - no file, a file from another game, a file whose
    planes are the shape the encoder used to produce - costs the run a few generations of refilling
    and costs it nothing else, so refusing to start would trade an inconvenience for the outage the
    buffer exists to prevent. It says what it did on the way past, because a buffer that silently
    stopped being restored would look like a run that had quietly got worse at learning.

    The game and the shape are both checked, and the shape check is the one that earns its keep: a
    buffer of tic-tac-toe planes is rejected by the name, but a buffer written before an encoder
    changed has the right name and the wrong contents, and would train the network on positions
    that no longer mean what they say.
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
