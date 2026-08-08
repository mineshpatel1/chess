"""
The contract every game in this project implements, and the only thing the search in `ai/`
knows about a game.

The search never inspects a move. It asks for the legal ones, plays them, unplays them, and
hands back the one it liked, so a move can be whatever suits the game: a from/to pair in
chess, a cell in tic-tac-toe, a column in connect-4. All it needs is to be hashable and
printable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, Iterator, List, NamedTuple, Optional, Tuple, Type

# True is the first player, the one a positive evaluation favours. Chess already works this
# way: game.constants defines Colour = bool with WHITE = True.
Player = bool


class Outcome(NamedTuple):
    """How a finished game finished. A winner of None is a draw."""
    winner: Optional[Player]

    def __str__(self) -> str:
        if self.winner is None:
            return 'Draw'
        return f'{"First" if self.winner else "Second"} player wins'


DRAW = Outcome(None)


def win(player: Player) -> Outcome:
    return Outcome(player)


def has_moves(moves: Iterable[Any]) -> bool:
    """
    Whether there are any moves at all, said carefully.

    `any(moves)` is the obvious way to write this and is wrong, because it asks whether any move
    is *truthy* rather than whether any move exists. A move is whatever suits the game, and the
    two small games here number theirs from zero - a Connect 4 column, a tic-tac-toe cell - so
    move 0 is a perfectly legal move that is also falsy. A position whose only continuation is
    move 0 would be reported as finished while it is still being played: Connect 4 called a draw
    with four cells free in column 0, and tic-tac-toe with the top-left corner still empty, which
    happens in about one game in nine.

    Chess escapes it only because a Move object has no __bool__ and so is always truthy. That is
    not a property the contract asks for, which is exactly why this is a function rather than a
    convention to remember at each call site.
    """
    return any(True for _ in moves)


Planes = List[List[List[int]]]
Policy = List[float]


class Encoder(ABC):
    """
    How a game is shown to a neural network, for the learned player in `ai.zero`.

    Optional: a game without one is still a perfectly good game, it just cannot be learned. It is
    kept separate from GameState because it answers a different question - GameState says what
    the rules are, this says what a network sees - and because only some games will ever want it.

    Everything here is plain Python: planes are nested lists of ints and moves stay whatever the
    game already uses. Tensors are built in `ai.zero`, which is what lets `games/` keep its
    promise of no third-party dependencies.

    Two rules matter more than the rest, and both are here because a previous attempt at a learned
    player in this project got them wrong in ways that took months to find.

    **Planes are relative to the player to move**, never to a fixed player: plane 0 is always
    "mine" and plane 1 always "theirs". A network then only has to learn one player's problem,
    and every position teaches it something about both seats.

    **The action space is single and shared.** POLICY_SIZE covers the moves of *the player to
    move*, whoever that is - not one block of outputs per player. Splitting it halves the data
    each half sees, and invites the two halves to disagree about which block is which.
    """

    # (planes, rows, columns) of what `planes` returns.
    PLANE_SHAPE: Tuple[int, int, int]

    # How many distinct actions exist, shared by both players.
    POLICY_SIZE: int

    @staticmethod
    @abstractmethod
    def planes(state: 'GameState') -> Planes:
        """The position as the player to move sees it."""

    @staticmethod
    @abstractmethod
    def action_index(move: Any) -> int:
        """Which policy output corresponds to `move`."""

    @staticmethod
    @abstractmethod
    def action_move(index: int) -> Any:
        """The move a policy output corresponds to. The inverse of `action_index`."""

    @staticmethod
    def symmetries(planes: Planes, policy: Policy) -> Iterator[Tuple[Planes, Policy]]:
        """
        Equivalent (position, policy) pairs, for turning one training example into several.

        A board with symmetries gets them for free: a rotated position is the same position, so
        the rotated policy is the same policy, and the value does not change at all. The default
        claims no symmetry and yields the example unaltered, which is always correct and never
        useful.

        Whatever a game yields here has to keep planes and policy in step. A transform applied to
        one and not the other teaches the network to play the mirror image of what it is shown.
        """
        yield planes, policy


class GameState(ABC):
    """
    A position in a two-player, perfect-information, zero-sum game.

    Implementations are mutable and are expected to make and unmake moves in place: the search
    walks a single state up and down the tree rather than copying it at every node, so
    `unmake_move` must restore the previous position exactly.
    """

    # Whether splitting the root moves across a process pool pays for itself. Chess sets this;
    # games with a branching factor in single figures search faster in-process than they can
    # start a pool, so the default is off.
    PARALLEL_ROOT = False

    # The evaluation the search reaches for when it is not given one.
    DEFAULT_EVAL: Callable[['GameState'], int]

    # The depth at which a search sees the whole game, for a game small enough to be solved.
    #
    # A search this deep reaches only finished positions, so it never consults an evaluation and
    # never guesses: it is playing perfectly. `play.py` offers this as its default depth, which
    # is how tic-tac-toe comes out of the box unbeatable rather than merely good.
    #
    # None for a game whose tree does not end inside any depth worth searching, which is most of
    # them - chess and Connect 4 both leave it alone.
    SOLVED_DEPTH: Optional[int] = None

    # How this game is shown to a neural network, for `ai.zero`. None means the game cannot be
    # learned, which is the default and costs a game that does not care precisely nothing.
    ENCODER: Optional[Type[Encoder]] = None

    # Which player is to move.
    turn: Player

    @property
    @abstractmethod
    def legal_moves(self) -> Iterable[Any]:
        """Every move the player to move may legally play, in any order."""

    @abstractmethod
    def make_move(self, move: Any) -> None:
        """Plays `move`, handing the turn to the other player."""

    @abstractmethod
    def unmake_move(self) -> None:
        """Undoes the last move, restoring the position exactly."""

    @abstractmethod
    def copy(self) -> 'GameState':
        """
        An independent state at the same position. Used to hand a position to a worker
        process, so it need not carry history that only undo would want.
        """

    @property
    @abstractmethod
    def outcome_without_moves(self) -> Outcome:
        """
        The result when the player to move has no legal move at all.

        Chess: a loss for the player to move if it is in check, otherwise a draw.
        Tic-tac-toe and connect-4: a draw, the board being full.
        """

    @property
    def outcome(self) -> Optional[Outcome]:
        """
        The result if this position is already decided by a *win condition* — one the game can
        test cheaply, without generating moves. None while the game is still running.

        The search checks this at every node, so it must stay cheap. It defaults to None
        because most games, chess among them, end by a player running out of moves rather than
        by a condition on the position, and those games are served by
        `outcome_without_moves` alone. Games won while moves remain — a line of three, a line
        of four — override it.
        """
        return None

    @property
    def winning_moves(self) -> Iterable[Any]:
        """
        Every legal move that wins the game on the spot.

        An optimisation hook, and it is allowed to under-report: `ai.oracle` uses it only to stop
        early when a win is available, so missing one costs a search it need not have done and
        finding one that does not win would be a wrong answer. Yield conservatively.

        The default plays each move and looks, which is correct for any game that decides a win
        with `outcome` and finds nothing at all in a game - chess among them - that ends by the
        loser running out of moves. That is the right default for chess: a mate search here would
        cost more at every node than the shortcut could ever save.

        Games override it when they can answer the question without playing anything. Connect 4
        can, in bit operations, and it is worth four times the default because the solver asks at
        every node it visits.

        Playing and unplaying `self` while a caller iterates is safe as written - each move is
        undone before its result is yielded - but a caller that keeps the generator across its own
        moves will get nonsense. Consume it before doing anything else.
        """
        for move in self.legal_moves:
            self.make_move(move)
            outcome = self.outcome
            self.unmake_move()
            if outcome is not None and outcome.winner is not None:
                yield move

    @property
    def solver_key(self) -> Any:
        """
        What makes two positions the same question, as cheaply as the game can say it.

        Used to key the exact solver's transposition table in `ai.oracle`, which touches it once
        per node - so on a game with millions of nodes this is a hot path and the default is a
        poor one. `signature` is a string, and building and hashing a string per node costs more
        than the rest of a node put together.

        The turn is folded in because a signature need not carry it, and a position with the
        other player to move is a different question with a different answer.

        Games override with something the machine can hash in one instruction. Connect 4 has an
        integer that identifies a position outright, so it does.
        """
        return self.signature, self.turn

    @property
    def canonical_key(self) -> Any:
        """
        A key that a position and its mirror image share, for a game with a symmetry.

        Only sound for *values*: a position and its reflection are worth exactly the same, so a
        table of values can be shared between them and halve its size. It is **not** sound for
        moves - the winning move in a mirrored position is the mirrored move - and `ai.oracle`
        therefore never reads a move out of an entry keyed by this.

        The default claims no symmetry, which is always correct and never useful. Chess has none
        worth having (castling rights are not mirror-symmetric); Connect 4 has a left-right flip.
        """
        return self.solver_key

    def parse_move(self, text: str) -> Any:
        """
        The move a person meant by typing `text`, for a game loop reading from a terminal.

        The search never needs this - it only ever plays moves it was handed - so it is not
        abstract, and the default is enough for a game whose moves print the way they are
        typed: it matches against `str(move)` over the legal moves. Chess moves print as UCI
        and connect-4 moves as a column number, so both already work.

        Games override it to say something more useful than "no such move" when the input is
        wrong, which is most of what a person needs from a prompt. Raise ValueError with a
        readable message; the caller shows it and asks again.
        """
        for move in self.legal_moves:
            if str(move) == text.strip():
                return move
        raise ValueError(f'{text.strip()!r} is not a legal move here')

    @property
    def signature(self) -> str:
        """
        Everything that distinguishes this position from another one, as a string.

        Two states with the same signature must be the same position to play from. That is
        stricter than looking the same: chess positions that print identically can still
        differ in castling rights or in which pawn may be taken en passant, and a state
        restored without them is a different game wearing the same picture.

        Defaults to the printed board, which is right for games where the board is the whole
        of the state. Games carrying anything else override it - chess with its FEN.
        """
        return str(self)

    @property
    def result(self) -> Optional[Outcome]:
        """
        How the game finished, or None if it has not.

        This is the whole question in one place, and it is what a game loop wants. The search
        deliberately does not use it: it asks the two halves separately so that generating
        moves, much the more expensive half, happens once per node rather than twice.

        Games with ways to finish that neither half covers — chess draws by repetition or by
        the fifty move rule — override this and keep `is_game_over` agreeing with it for free.
        """
        outcome = self.outcome
        if outcome is not None:
            return outcome
        if not has_moves(self.legal_moves):
            return self.outcome_without_moves
        return None

    @property
    def is_game_over(self) -> bool:
        return self.result is not None
