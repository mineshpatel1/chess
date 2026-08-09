"""
What a training run leaves behind, so it can be read afterwards rather than watched.

A Connect 4 run is measured in hours. Watching a log scroll past is fine for tic-tac-toe, where a
generation is a second and a bad idea costs nothing to re-try; it is no way to run something that
will not finish before you want an answer about it. So every generation appends one JSON object to
a file, and `plot.py` turns the file into a page.

**Appended and flushed per generation, never held and written at the end.** A run that dies at hour
three - killed, out of memory, a bug in the last thing changed - is exactly the run whose history
is most worth having, and it is the one a write-at-the-end design loses.

**One object per line rather than one JSON document.** A partial line at the end of a killed file
is a line the reader can skip; a truncated JSON document is unreadable in its entirety.

The fields are chosen so that a flat curve can be diagnosed rather than merely observed, which is
the difference between "it is not learning" and knowing which part to fix. Each distinguishes a
failure the others cannot:

    optimal_rate        the headline: does the raw policy pick a best move
    first_rate,         the same split by seat. A player strong as one and hopeless as the
    second_rate         other is what the 2021 attempt in this repo actually was
    value_mse           a value head confidently wrong while the policy looks healthy
    policy_loss,        which head has stalled; a total hides it
    value_loss
    target_entropy      how sharp the search's targets are. Tic-tac-toe's c_puct fault showed up
                        as policy loss flattening *at* the entropy of its own targets - the
                        network fitting confidently wrong targets perfectly, which looks
                        identical to a network that cannot learn
    distinct_positions  self-play is on-policy and narrows as it improves; tic-tac-toe once
                        collapsed to 366 of the game's 4,520 decision positions
    draw_rate,          sanity on the games themselves, and nearly free to collect
    game_length
    seconds, and the    where the time actually goes, so a projection of the full run is measured
    split by phase      rather than guessed
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

    A trailing partial line is skipped rather than raised on. A killed run leaves one about as
    often as not, and refusing to read the other four hundred generations because the last write
    was interrupted would defeat the reason the file is flushed per line.
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

    What a resume needs so the file agrees with the checkpoint it is resuming from. A generation
    is recorded before its checkpoint is written - it has to be, since the record is what says the
    generation happened - so there is always a window in which the file is one generation ahead of
    the weights on disk. A run killed inside that window and then resumed would otherwise record
    the same generation number twice, and the file would describe a history that never happened.

    Rewritten rather than seeked and truncated in place, because the last line may itself be half
    written; `read` already knows how to stop at one, and this is only a few hundred lines.

    **The decision to rewrite is by content, not by whether anything was dropped.** A file killed
    mid-write ends in a partial line that `read` skips and no generation is past the checkpoint, so
    a count of dropped records is zero and the damage stays - and the resumed run then appends onto
    the half-written line, gluing two records together. Comparing what the file says against what
    it should say catches the repair and the truncation with one condition.
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
