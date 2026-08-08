"""
Exact play, and how far short of it a player falls.

Most games cannot be solved, which is why the rest of `ai/` is built on things that stand in for
correctness - perft counts, pinned search scores, match results. A game small enough to enumerate
needs none of those substitutes: the answer is computable, so a player can be graded against it
rather than against another player.

That grading is what this module is for, and it is a different question from "does it win". A
player that wins every game against a weak opponent has been shown very little. Walking *every*
position and asking "was that one of the best moves here" is a measurement with no opponent in
it at all, and no sampling either - it is the whole game, every time.

The benchmark reports the answer split by seat, and that split is the point. A player can be
excellent as the first player and hopeless as the second, and an aggregate score hides it behind
an average. That exact failure - strong first, weak second - is what a previous attempt at a
learned player in this project turned out to have, and it went unnoticed for months.

Only for games that declare `SOLVED_DEPTH`. Everything here enumerates the whole state space, so
asking it about chess would not return.
"""

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Tuple,
)

from games.base import GameState, has_moves

# Values are from the point of view of the player to move, the same convention `ai.search` uses.
WIN = 1
DRAW = 0
LOSS = -1


# How a stored value relates to the true one. A search that cut off did not learn the exact
# value, only that it was at least or at most what it returned, and a table that forgets which
# is which will hand a bound back as though it were the answer.
EXACT, LOWER, UPPER = 0, 1, 2


class Table:
    """
    What the solver remembers between positions.

    Two dictionaries, deliberately keyed differently, and the difference is the whole reason this
    is a class rather than a dict:

    *Values* are keyed by `canonical_key`, so a position and its mirror image share an entry. That
    is sound because a reflected board is worth exactly what the original is, and on Connect 4 it
    halves the table.

    *Move hints* are keyed by `solver_key`, and are never shared with a mirror. The best move in a
    reflected position is the *reflected* move, so a hint read from a mirrored entry would be a
    move for the wrong side of the board. Keeping them in separate dictionaries means that cannot
    happen by accident: there is no mirrored entry to read a move out of.
    """

    __slots__ = ('values', 'hints', 'hits', 'nodes')

    def __init__(self) -> None:
        self.values: Dict[Any, Tuple[int, int]] = {}
        self.hints: Dict[Any, Any] = {}
        self.hits = 0
        self.nodes = 0


def _value_of(outcome, turn) -> int:
    """A finished game's result, from the point of view of the player to move in it."""
    if outcome.winner is None:
        return DRAW
    return WIN if outcome.winner == turn else LOSS


def _finished_value(state: GameState) -> int:
    """The value of a position that is over. Raises if it is not."""
    result = state.result
    if result is None:
        raise ValueError(f'this game is not finished:\n{state}')
    return _value_of(result, state.turn)


def solve(state: GameState, _table: Optional[Table] = None) -> int:
    """
    The value of a position with best play by both sides, from the mover's point of view.

    Alpha-beta over a transposition table, searched on a full window - which is what keeps this an
    exact answer rather than a bound. Pruning inside a full window is sound: a cutoff only ever
    discards a line that provably cannot beat one already found, so the value that comes back is
    the true one. `move_values` relies on that, and searches every root child on a full window for
    the same reason.

    Values are in {-1, 0, 1}, which makes the window unusually tight and the pruning unusually
    good: proving a move wins is enough to stop looking at the rest.
    """
    table = _table if _table is not None else Table()
    return _search(state, LOSS, WIN, table)


def _search(state: GameState, alpha: int, beta: int, table: Table) -> int:
    """Negamax with alpha-beta, returning the value of `state` to the player to move in it."""
    table.nodes += 1

    outcome = state.outcome
    if outcome is not None:  # Somebody has already won; it cannot have been the player to move
        return _value_of(outcome, state.turn)

    moves = list(state.legal_moves)
    if not moves:
        return _value_of(state.outcome_without_moves, state.turn)

    # A move that wins right now ends the question, and collapses an entire subtree into one
    # test. In a game where most lines end in a forced win that is the single largest saving
    # available, which is why `winning_moves` is a hook a game can answer cheaply rather than a
    # loop over make/unmake here - Connect 4 answers it in bit operations, four times faster.
    #
    # Asked before the table is consulted, because it is cheaper than the lookup: on Connect 4 the
    # key involves mirroring the board, and there is no sense paying for that to learn something
    # a handful of shifts already knows.
    if has_moves(state.winning_moves):
        return WIN

    key = state.canonical_key
    stored = table.values.get(key)
    if stored is not None:
        value, flag = stored
        if flag == EXACT:
            table.hits += 1
            return value
        if flag == LOWER:
            alpha = max(alpha, value)
        else:
            beta = min(beta, value)
        if alpha >= beta:
            table.hits += 1
            return value

    original_alpha = alpha
    hint = table.hints.get(state.solver_key)
    if hint is not None and hint in moves:
        moves = [hint] + [move for move in moves if move != hint]

    best, best_move = LOSS - 1, None
    for move in moves:
        state.make_move(move)
        reply = -_search(state, -beta, -alpha, table)
        state.unmake_move()

        if reply > best:
            best, best_move = reply, move
        alpha = max(alpha, reply)
        if alpha >= beta:
            break  # This move is already good enough that the rest cannot matter

    flag = EXACT if original_alpha < best < beta else (LOWER if best >= beta else UPPER)
    table.values[key] = (best, flag)
    if best_move is not None:
        table.hints[state.solver_key] = best_move
    return best


def move_values(state: GameState, _table: Optional[Table] = None) -> Dict[Any, int]:
    """
    What each legal move is worth to the player to move. The whole answer, not just the best.

    Every child is searched on a full window, so every value that comes back is exact rather than
    a bound. That costs more than asking only for the best move and is the entire point: grading a
    player needs to know what the move it chose was worth, not merely that something beat it.
    """
    table = _table if _table is not None else Table()
    values = {}
    for move in state.legal_moves:
        state.make_move(move)
        values[move] = -_search(state, LOSS, WIN, table)
        state.unmake_move()
    return values


def optimal_moves(state: GameState, _table: Optional[Table] = None) -> List[Any]:
    """
    Every move that preserves the value of the position.

    Plural on purpose. Several moves are usually equally best, and a player that picks a different
    one from the solver has not made a mistake - grading against a single "correct" move would
    measure agreement with an arbitrary tie-break rather than quality of play.
    """
    values = move_values(state, _table)
    if not values:
        return []
    best = max(values.values())
    return [move for move, value in values.items() if value == best]


def enumerate_positions(game: Callable[[], GameState]) -> Iterator[Tuple[GameState, int]]:
    """
    Every position reachable from a new game, each yielded once, with the ply it occurs at.

    Yields the live state rather than a copy, so a caller keeping one must copy it. Deduplicated
    by `solver_key`: a position reachable by four different move orders is one position and is
    graded once, not four times weighted by how many ways there are to reach it.

    Only for a game small enough to walk. This is what `benchmark` is given for tic-tac-toe; a
    game that cannot be enumerated hands it a sample instead.
    """
    seen = set()
    state = game()

    def walk(ply: int) -> Iterator[Tuple[GameState, int]]:
        key = state.solver_key
        if key in seen:
            return
        seen.add(key)

        yield state, ply
        if state.is_game_over:
            return

        for move in list(state.legal_moves):
            state.make_move(move)
            yield from walk(ply + 1)
            state.unmake_move()

    yield from walk(0)


class Record(NamedTuple):
    """Every line a player could be taken down, counted."""

    wins: int
    draws: int
    losses: int

    @property
    def lines(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def unbeaten(self) -> bool:
        return self.losses == 0

    def __str__(self) -> str:
        verdict = 'unbeaten' if self.unbeaten else f'LOSES {self.losses}'
        return f'+{self.wins} ={self.draws} -{self.losses} over {self.lines} lines ({verdict})'


def play_every_line(
    player: Callable[[GameState], Any],
    game: Callable[[], GameState],
    seat: bool,
    opponent: Optional[Callable[[GameState], List[Any]]] = None,
) -> Record:
    """
    Plays `player` against every line its opponent has available, exhaustively.

    A different question from `benchmark`, and the one that decides whether a player is any good
    to play against. `benchmark` asks whether the player knows the whole game; this asks whether
    the game can be won against it. They come apart in exactly one direction: a player can be
    wrong in hundreds of positions and still be unbeatable, because a player that never blunders
    *on the path it actually walks* never reaches the positions it would get wrong.

    `opponent` returns the moves to branch on. The default tries everything, which is the
    strongest claim available - no sequence of moves beats this player. Passing `optimal_moves`
    asks the weaker and more usual question, whether it holds a draw against best play.

    Memoised on the position, so an opponent's 255,168 games collapse to the few thousand
    positions the player can actually be faced with.
    """
    branch = opponent or (lambda state: list(state.legal_moves))
    memo: Dict[Tuple[str, bool], Record] = {}
    state = game()

    def walk() -> Record:
        key = (state.signature, state.turn)
        if key in memo:
            return memo[key]

        result = state.result
        if result is not None:
            record = Record(
                wins=int(result.winner == seat),
                draws=int(result.winner is None),
                losses=int(result.winner == (not seat)),
            )
        elif state.turn == seat:
            state.make_move(player(state))
            record = walk()
            state.unmake_move()
        else:
            wins = draws = losses = 0
            for move in branch(state):
                state.make_move(move)
                below = walk()
                state.unmake_move()
                wins, draws, losses = wins + below.wins, draws + below.draws, losses + below.losses
            record = Record(wins, draws, losses)

        memo[key] = record
        return record

    return walk()


class Grade(NamedTuple):
    """How a player did over some set of positions."""

    positions: int
    optimal: int
    value_lost: int
    blunders: int  # Moves that gave away a win or a draw outright

    @property
    def rate(self) -> float:
        """The fraction of positions where the player chose one of the best moves."""
        return self.optimal / self.positions if self.positions else 0.0

    @property
    def mean_value_lost(self) -> float:
        return self.value_lost / self.positions if self.positions else 0.0

    def __str__(self) -> str:
        return (
            f'{self.rate:6.1%} optimal ({self.optimal}/{self.positions}), '
            f'{self.blunders} blunders, {self.mean_value_lost:.4f} mean value lost'
        )


VALUE_NAMES = {WIN: 'winning', DRAW: 'drawn  ', LOSS: 'losing '}


class Report(NamedTuple):
    """A player graded against perfect play, whole and split the ways that matter."""

    overall: Grade
    by_seat: Dict[bool, Grade]
    by_ply: Dict[int, Grade]
    worst: List[Tuple[str, Any, List[Any]]]  # (board, move played, moves that were best)
    value_error: Optional[float] = None  # Mean squared error of a value function, if given
    by_value: Dict[int, Grade] = {}  # Keyed by what the position is worth to the player to move

    def __str__(self) -> str:
        lines = [f'  overall      {self.overall}']
        for seat in (True, False):
            if seat in self.by_seat:
                name = 'first ' if seat else 'second'
                lines.append(f'  as {name}     {self.by_seat[seat]}')

        for value in (WIN, DRAW, LOSS):
            if value in self.by_value:
                lines.append(f'  in {VALUE_NAMES[value]}   {self.by_value[value]}')

        lines.append('  by ply       ' + '  '.join(
            f'{ply}:{grade.rate:.0%}' for ply, grade in sorted(self.by_ply.items())
        ))
        if self.value_error is not None:
            lines.append(f'  value head   {self.value_error:.4f} mean squared error vs truth')
        return '\n'.join(lines)


class _Tally:
    def __init__(self):
        self.positions = self.optimal = self.value_lost = self.blunders = 0

    def add(self, was_optimal: bool, lost: int):
        self.positions += 1
        self.optimal += int(was_optimal)
        self.value_lost += lost
        self.blunders += int(lost > 0)

    def grade(self) -> Grade:
        return Grade(self.positions, self.optimal, self.value_lost, self.blunders)


def benchmark(
    player: Callable[[GameState], Any],
    positions: Iterable[Tuple[GameState, int]],
    values: Callable[[GameState], Dict[Any, int]] = move_values,
    value_fn: Optional[Callable[[GameState], float]] = None,
    worst_examples: int = 5,
) -> Report:
    """
    Grades `player` over `positions` against perfect play.

    No opponent and no games: the player is asked for a move in each position, and each answer is
    compared with the set of moves that hold the position's value. A player is only as good as its
    worst reachable position, and playing matches will not find those - losing lines are exactly
    the ones a decent opponent never steers into.

    Both of what it needs are arguments, because how you get them differs by game and the report
    should not. Tic-tac-toe passes `enumerate_positions(TicTacToe)` and the live solver: the whole
    state space, valued on demand. Connect 4 cannot enumerate 4.5e12 positions, so it passes a
    sampled corpus and a lookup into pinned values solved once, ahead of time. Chess will pass
    whatever it can get. The grading is identical in all three, which is what makes the numbers
    comparable - and comparing them across games is most of what this is for.

    `values` maps a state to what each of its legal moves is worth, and defaults to solving on the
    spot. A pinned corpus passes `corpus.__getitem__` in effect, and never runs a search at all.

    `value_fn` is optional and grades a *position evaluator* rather than a move chooser - for a
    learned player, its value head against the true game-theoretic value. Worth separating,
    because a player can pick good moves with a badly calibrated evaluation and vice versa. It is
    graded against `values` too, taking the best move's value as the position's, so that it needs
    no second source of truth.

    The by-value split exists because it is the main way a score over *sampled* positions can
    flatter. A position that is already winning usually has several moves that keep it winning, so
    a set full of them is easy; the drawn positions, where exactly one move holds and the rest
    lose, are where a player is really tested. An aggregate over a set that happens to be mostly
    decided says more about the sampling than about the player.
    """
    overall = _Tally()
    by_seat: Dict[bool, _Tally] = {}
    by_ply: Dict[int, _Tally] = {}
    by_value: Dict[int, _Tally] = {}
    worst: List[Tuple[str, Any, List[Any]]] = []
    squared_error = 0.0
    evaluated = 0

    for state, ply in positions:
        if state.is_game_over:
            if value_fn is not None:
                squared_error += (value_fn(state) - _finished_value(state)) ** 2
                evaluated += 1
            continue

        worths = values(state)
        best = max(worths.values())

        if value_fn is not None:
            squared_error += (value_fn(state) - best) ** 2
            evaluated += 1

        move = player(state)
        if move not in worths:
            raise ValueError(f'{player} returned {move!r}, which is not legal here:\n{state}')

        lost = best - worths[move]
        was_optimal = lost == 0

        overall.add(was_optimal, lost)
        by_seat.setdefault(state.turn, _Tally()).add(was_optimal, lost)
        by_ply.setdefault(ply, _Tally()).add(was_optimal, lost)
        by_value.setdefault(best, _Tally()).add(was_optimal, lost)

        if lost > 0 and len(worst) < worst_examples:
            worst.append((str(state), move, [m for m, v in worths.items() if v == best]))

    return Report(
        overall=overall.grade(),
        by_seat={seat: tally.grade() for seat, tally in by_seat.items()},
        by_ply={ply: tally.grade() for ply, tally in by_ply.items()},
        worst=worst,
        value_error=(squared_error / evaluated) if evaluated else None,
        by_value={value: tally.grade() for value, tally in by_value.items()},
    )
