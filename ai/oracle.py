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

from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Tuple

from games.base import GameState

# Values are from the point of view of the player to move, the same convention `ai.search` uses.
WIN = 1
DRAW = 0
LOSS = -1


def _key(state: GameState) -> str:
    """
    What makes two positions the same question.

    `signature` is the contract's own answer to that, and it is stricter than the printed board:
    two chess positions that look alike can differ in castling rights and play differently. The
    turn is folded in because a signature need not carry it, and a position with the other player
    to move is a different question with a different answer.
    """
    return f'{state.signature}|{int(state.turn)}'


def solve(state: GameState, _memo: Optional[Dict[str, int]] = None) -> int:
    """
    The value of a position with best play by both sides, from the mover's point of view.

    Plain memoised negamax, with no alpha-beta. Pruning would be faster and would make this a
    worse instrument: a pruned search returns a bound for the moves it cut off, and this has to
    be able to say what *every* move is worth, not just which one is best.

    Memoised on `_key`, so the 255,168 games of tic-tac-toe collapse to its 5,478 positions.
    """
    memo = _memo if _memo is not None else {}
    key = _key(state)
    if key in memo:
        return memo[key]

    result = state.result
    if result is not None:
        value = DRAW if result.winner is None else (WIN if result.winner == state.turn else LOSS)
    else:
        best = LOSS - 1
        for move in state.legal_moves:
            state.make_move(move)
            reply = -solve(state, memo)
            state.unmake_move()
            best = max(best, reply)
        value = best

    memo[key] = value
    return value


def move_values(state: GameState, _memo: Optional[Dict[str, int]] = None) -> Dict[Any, int]:
    """What each legal move is worth to the player to move. The whole answer, not just the best."""
    memo = _memo if _memo is not None else {}
    values = {}
    for move in state.legal_moves:
        state.make_move(move)
        values[move] = -solve(state, memo)
        state.unmake_move()
    return values


def optimal_moves(state: GameState, _memo: Optional[Dict[str, int]] = None) -> List[Any]:
    """
    Every move that preserves the value of the position.

    Plural on purpose. Several moves are usually equally best, and a player that picks a different
    one from the solver has not made a mistake - grading against a single "correct" move would
    measure agreement with an arbitrary tie-break rather than quality of play.
    """
    values = move_values(state, _memo)
    if not values:
        return []
    best = max(values.values())
    return [move for move, value in values.items() if value == best]


def enumerate_positions(game: Callable[[], GameState]) -> Iterator[Tuple[GameState, int]]:
    """
    Every position reachable from a new game, each yielded once, with the ply it occurs at.

    Yields the live state rather than a copy, so a caller keeping one must copy it. Deduplicated
    by `_key`: a position reachable by four different move orders is one position and is graded
    once, not four times weighted by how many ways there are to reach it.
    """
    seen = set()
    state = game()

    def walk(ply: int) -> Iterator[Tuple[GameState, int]]:
        key = _key(state)
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


class Report(NamedTuple):
    """A player graded against perfect play, whole and split the ways that matter."""

    overall: Grade
    by_seat: Dict[bool, Grade]
    by_ply: Dict[int, Grade]
    worst: List[Tuple[str, Any, List[Any]]]  # (board, move played, moves that were best)
    value_error: Optional[float] = None  # Mean squared error of a value function, if given

    def __str__(self) -> str:
        lines = [f'  overall      {self.overall}']
        for seat in (True, False):
            if seat in self.by_seat:
                name = 'first ' if seat else 'second'
                lines.append(f'  as {name}     {self.by_seat[seat]}')

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
    game: Callable[[], GameState],
    value_fn: Optional[Callable[[GameState], float]] = None,
    worst_examples: int = 5,
) -> Report:
    """
    Grades `player` in every position of `game` against perfect play.

    No opponent and no games: the player is asked for a move in every position that can arise,
    including ones no sensible game would reach, and each answer is compared with the set of
    moves that hold the position's value. A player is only as good as its worst reachable
    position, and playing matches will not find those - losing lines are exactly the ones a
    decent opponent never steers into.

    `value_fn` is optional and grades a *position evaluator* rather than a move chooser - for a
    learned player, its value head against the true game-theoretic value. Worth separating,
    because a player can pick good moves with a badly calibrated evaluation and vice versa.
    """
    memo: Dict[str, int] = {}
    overall = _Tally()
    by_seat: Dict[bool, _Tally] = {}
    by_ply: Dict[int, _Tally] = {}
    worst: List[Tuple[str, Any, List[Any]]] = []
    squared_error = 0.0
    evaluated = 0

    for state, ply in enumerate_positions(game):
        if value_fn is not None:
            squared_error += (value_fn(state) - solve(state, memo)) ** 2
            evaluated += 1

        if state.is_game_over:
            continue

        values = move_values(state, memo)
        best = max(values.values())
        move = player(state)

        if move not in values:
            raise ValueError(f'{player} returned {move!r}, which is not legal here:\n{state}')

        lost = best - values[move]
        was_optimal = lost == 0

        overall.add(was_optimal, lost)
        by_seat.setdefault(state.turn, _Tally()).add(was_optimal, lost)
        by_ply.setdefault(ply, _Tally()).add(was_optimal, lost)

        if lost > 0 and len(worst) < worst_examples:
            worst.append((str(state), move, [m for m, v in values.items() if v == best]))

    return Report(
        overall=overall.grade(),
        by_seat={seat: tally.grade() for seat, tally in by_seat.items()},
        by_ply={ply: tally.grade() for ply, tally in by_ply.items()},
        worst=worst,
        value_error=(squared_error / evaluated) if evaluated else None,
    )
