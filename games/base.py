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
from typing import Any, Callable, Iterable, NamedTuple, Optional

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
        if not any(self.legal_moves):
            return self.outcome_without_moves
        return None

    @property
    def is_game_over(self) -> bool:
        return self.result is not None
