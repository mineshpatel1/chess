"""
Positions to test against, generated once and written down.

A fixture that regenerates on every run is not a fixture: it moves when the code under it moves,
which is exactly what a fixture is for stopping. So the games below are literals, and the comment
records how the drawn one was found rather than the search being left in to run again.

`reachable_positions` is the exception, and is not a corpus at all. Chess and Connect 4 sample a
reproducible spread of positions because their trees cannot be walked; tic-tac-toe's can, so the
tests that want a range of positions get *every* position instead of a sample of them, and a
claim proved over it is proved outright.
"""

from typing import Iterator, List

from games.tictactoe.board import TicTacToe

# Games decided one way each, one per direction of line. Five plies is the earliest a game can
# be won, so each of these is also a minimal example.
#
#     0 1 2
#     3 4 5
#     6 7 8
#
ROW_WIN: List[int] = [0, 3, 1, 4, 2]  # Crosses take the top row
COLUMN_WIN: List[int] = [0, 1, 3, 2, 6]  # Crosses take the left column
DIAGONAL_WIN: List[int] = [0, 1, 4, 2, 8]  # Crosses take 0-4-8
ANTI_DIAGONAL_WIN: List[int] = [2, 0, 4, 1, 6]  # Crosses take 2-4-6

# The second player winning, so that nothing is only ever tested from Crosses' side. Noughts
# cannot win before ply 6, this being their third mark.
NOUGHT_WIN: List[int] = [0, 4, 1, 3, 8, 5]

# A full board that nobody won, found by playing seeded random games until one drew - seed 0,
# first time. Random tic-tac-toe draws about one game in nine, where random Connect 4 took 1,168
# tries. Every cell is taken and no line is complete, so this is the only way
# `outcome_without_moves` gets reached.
DRAWN_GAME: List[int] = [3, 5, 4, 6, 7, 1, 2, 8, 0]

# The position that `GameState.result` used to get wrong: eight cells taken, nobody has won, and
# the only move left is cell 0 - which is falsy, so `any(legal_moves)` called it a finished game
# and the drawn game above could never be reached at all.
#
# It is DRAWN_GAME without its last ply, which is not a contrivance but the point: the bug sat on
# the ordinary route to an ordinary draw, and the reason it survived Connect 4 being added is
# that a walk stopping at `is_game_over` stops one move before it.
LAST_CELL_IS_ZERO: List[int] = DRAWN_GAME[:-1]


def reachable_positions() -> Iterator[TicTacToe]:
    """
    Every position reachable from an empty board, each yielded once.

    Yields the live state rather than a copy, so a caller that wants to keep one must copy it.
    There are 5,478 of them, terminal positions included, which is the published count and is
    what tests/tictactoe/test_permutations.py checks this against.
    """
    seen = set()
    state = TicTacToe()

    def walk() -> Iterator[TicTacToe]:
        if state.signature in seen:
            return
        seen.add(state.signature)

        yield state
        if state.is_game_over:
            return

        for move in list(state.legal_moves):
            state.make_move(move)
            yield from walk()
            state.unmake_move()

    yield from walk()
