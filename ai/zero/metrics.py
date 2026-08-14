"""
What a training run leaves behind, so it can be read afterwards rather than watched.

Every generation appends one JSON object to a file and `plot.py` turns the file into a page.
Appended and flushed per generation, because the run whose history is most worth having is the one
that died at hour three; one object per line rather than one document, because a killed file's
partial last line is one a reader can skip.

The fields are chosen so a flat curve can be diagnosed rather than merely observed. Each
distinguishes a failure the others cannot:

    optimal_rate        the headline: does the raw policy pick a best move
    first_rate,         the same split by seat, catching a player strong as one and hopeless
    second_rate         as the other
    value_mse           a value head confidently wrong while the policy looks healthy
    policy_loss,        which head has stalled; a total hides it
    value_loss
    target_entropy      how sharp the search's targets are. A policy loss flattening *at* this
                        is a network fitting bad targets perfectly, which looks identical to
                        one that cannot learn
    distinct_positions  self-play is on-policy and narrows as it improves
    draw_rate,          sanity on the games themselves, and nearly free to collect
    game_length
    seconds, and the    where the time actually goes, so a projection of the full run is
    split by phase      measured rather than guessed
"""

import json
import os
from typing import Any, Dict, Iterator, List, Optional, TextIO


class Recorder:
    """
    Appends one JSON object per generation to a file, flushing as it goes.

    A no-op when given no path, so a caller never has to branch on whether recording is on -
    `train` always has a recorder and sometimes it writes nowhere.
    """

    def __init__(self, path: Optional[str] = None, append: bool = False) -> None:
        """`append` keeps what is already in the file, which is what a resumed run wants."""
        self.path = path
        self._handle: Optional[TextIO] = None

        if path:
            directory = os.path.dirname(os.path.abspath(path))
            os.makedirs(directory, exist_ok=True)
            self._handle = open(path, 'a' if append else 'w')

    def write(self, record: Dict[str, Any]) -> None:
        if self._handle is None:
            return
        self._handle.write(json.dumps(record, sort_keys=True) + '\n')
        self._handle.flush()  # The point of the file is to survive the run that wrote it

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> 'Recorder':
        return self

    def __exit__(self, *_) -> None:
        self.close()


def read(path: str) -> List[Dict[str, Any]]:
    """
    Every generation recorded in a file.

    A trailing partial line is skipped rather than raised on, which a killed run leaves about as
    often as not.
    """
    records = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except ValueError:
                break  # A half-written last line; everything before it is still good
    return records


def truncate_after(path: str, generation: int) -> int:
    """
    Drops recorded generations past `generation`, returning how many went.

    A generation is recorded before its checkpoint is written, so a run killed in that window
    leaves the file one generation ahead of the weights and a resume would record the same
    generation twice.

    Rewritten rather than truncated in place, and the decision to rewrite is by content rather
    than by whether anything was dropped: a file ending in a half-written line has nothing past
    the checkpoint to drop, but appending to it would glue two records together.
    """
    if not os.path.exists(path):
        return 0

    records = read(path)
    kept = [record for record in records if record['generation'] <= generation]
    wanted = ''.join(json.dumps(record, sort_keys=True) + '\n' for record in kept)

    with open(path) as handle:
        if handle.read() == wanted:
            return 0

    with open(path, 'w') as handle:
        handle.write(wanted)
    return len(records) - len(kept)


def series(records: List[Dict[str, Any]], field: str) -> Iterator[Any]:
    """The values of one field, for the generations that recorded it."""
    for record in records:
        if record.get(field) is not None:
            yield record['generation'], record[field]
