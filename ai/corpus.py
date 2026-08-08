"""
Positions whose exact value is known, written down once and read back.

`ai.oracle` grades a player by asking it for a move in every position and comparing the answer with
the moves that hold the position's value. Tic-tac-toe can compute that on the spot - 5,478
positions, solved in a loop. Connect 4 has about 4.5e12 and cannot, so the answers are computed
ahead of time and committed, and this is what reads them back.

The file is plain text, one position per line:

    E 3435 -1 0 1 1 0 0 x       tier, moves played, then the value of each column

`tier` says how the position was chosen and is the reason it is on the line at all - a score over
positions from random play and a score over positions from real play are different measurements and
must not be averaged together. `moves` are columns, ours and zero-indexed, in the order played;
`-` is the empty board. The seven values that follow are what each column is worth **to the player
to move**, in the project's usual {-1, 0, 1}, with `x` for a column that cannot be played.

Every legal move gets a value, not just the best ones. `benchmark` scores a player by how much
value its move gave away, so a file holding only the optimal set would leave it unable to tell a
draw thrown away from a win thrown away.

Plain text rather than a pickle or a Python literal, and uncompressed at that. It is a few hundred
kilobytes either way, and this way a disagreement shows up as a diff of a few lines rather than as
a binary blob that changed - which matters, because the whole point of the file is to be the thing
that does not move.
"""

from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Sequence, Tuple

from games.base import GameState

# Where each game's corpus lives. Here rather than on the game class, because `games/` describes
# rules and should not know that `ai/` keeps files anywhere. A game absent from this is a game
# whose whole state space fits in a loop, and which therefore needs no corpus at all.
CORPORA = {
    'Connect4': 'ai/corpora/connect4.txt',
}

# The tiers, in the order a report should read them, with what each one actually measures.
#
# Never merged into one number. `R` reaches positions no sensible game visits; `P` never asks a
# player to recover from a bad one. A single figure over the two would hide whichever is worse,
# and which of them is worse is exactly the interesting part.
TIERS = (
    ('E', 'the opening — every position through six discs, enumerated'),
    ('R', 'random play — sampled, plies 7 to 34'),
    ('P', 'real play — sampled from games between alpha-beta players'),
)

# What a column is worth when it cannot be played at all.
ILLEGAL = 'x'

# The empty board has no moves to write down, and a blank field would make the line ambiguous.
EMPTY = '-'


class Entry(NamedTuple):
    """One solved position: how it was reached, how it was chosen, and what every move is worth."""

    tier: str
    moves: List[int]
    values: Dict[int, int]

    @property
    def ply(self) -> int:
        return len(self.moves)

    @property
    def value(self) -> int:
        """What the position is worth to the player to move: the best any move can do."""
        return max(self.values.values())

    @property
    def optimal(self) -> List[int]:
        """Every move that holds that value. Plural, because a tie-break is not a mistake."""
        best = self.value
        return sorted(move for move, value in self.values.items() if value == best)


def parse(line: str) -> Entry:
    """One line of the file. Raises ValueError on anything malformed, rather than guessing."""
    fields = line.split()
    if len(fields) < 3:
        raise ValueError(f'expected a tier, moves and at least one value: {line!r}')

    tier, moves = fields[0], fields[1]
    played = [] if moves == EMPTY else [int(character) for character in moves]

    values = {}
    for column, field in enumerate(fields[2:]):
        if field != ILLEGAL:
            values[column] = int(field)

    if not values:
        raise ValueError(f'no column is playable, so this position is over: {line!r}')
    return Entry(tier, played, values)


def format_entry(entry: Entry, columns: int) -> str:
    """An entry as a line, the inverse of `parse`."""
    moves = ''.join(str(column) for column in entry.moves) or EMPTY
    values = ' '.join(
        str(entry.values[column]) if column in entry.values else ILLEGAL
        for column in range(columns)
    )
    return f'{entry.tier} {moves} {values}'


def load(path: str, tiers: Optional[Sequence[str]] = None) -> List[Entry]:
    """
    Every entry in the file, optionally only those from the given tiers.

    Blank lines and `#` comments are skipped, so the file can say at the top of itself what it is
    and where it came from - which a corpus nobody can trace the provenance of should not be.
    """
    entries = []
    with open(path) as handle:
        for number, line in enumerate(handle, start=1):
            line = line.split('#', 1)[0].strip()
            if not line:
                continue
            try:
                entry = parse(line)
            except ValueError as error:
                raise ValueError(f'{path} line {number}: {error}') from None
            if tiers is None or entry.tier in tiers:
                entries.append(entry)
    return entries


def positions(
    entries: Sequence[Entry], game: Callable[[Sequence[int]], GameState],
) -> Iterator[Tuple[GameState, int]]:
    """
    The entries as (state, ply) pairs, which is what `ai.oracle.benchmark` walks.

    A fresh state per entry rather than one replayed in place. `enumerate_positions` yields a live
    state because it is walking a tree and unmaking as it goes; here the positions have nothing to
    do with each other, so there is no walk to share and no reason to be clever.
    """
    for entry in entries:
        yield game(entry.moves), entry.ply


def values(entries: Sequence[Entry]) -> Callable[[GameState], Dict[Any, int]]:
    """
    A lookup in the shape `benchmark` wants: a state to what each of its moves is worth.

    Keyed by `solver_key`, which for Connect 4 is an integer identifying the position outright -
    so this is an exact lookup and not a hash that could collide. Two entries reaching the same
    position by different move orders would collapse to one key, which is correct: it is one
    question and deserves one answer, not two votes.

    Raises rather than returning nothing for a position the corpus does not hold. A benchmark that
    silently skipped what it could not value would report a rate over a set nobody chose.
    """
    from games.connect4.board import Connect4  # Local: `games` must not depend on `ai`

    table: Dict[Any, Dict[Any, int]] = {}
    for entry in entries:
        table[Connect4(entry.moves).solver_key] = dict(entry.values)

    def lookup(state: GameState) -> Dict[Any, int]:
        try:
            return table[state.solver_key]
        except KeyError:
            raise KeyError(f'no solved answer for this position:\n{state}') from None

    return lookup
