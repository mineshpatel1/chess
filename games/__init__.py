"""
Every game the project knows about.

Kept as a plain tuple of state classes rather than anything cleverer: the search reaches a
game through the GameState contract, not through here, so this exists only for the things that
genuinely want to enumerate games - a harness comparing them, a CLI choosing one.

Adding a game means adding its class here and nowhere else.
"""

from typing import Tuple, Type

from games.base import GameState
from games.chess.board import ChessBoard
from games.connect4.board import Connect4
from games.tictactoe.board import TicTacToe

GAMES: Tuple[Type[GameState], ...] = (
    ChessBoard,
    Connect4,
    TicTacToe,
)
