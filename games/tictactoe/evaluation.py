"""
How good a tic-tac-toe position is, in the terms the search wants it.

`value` is absolute - positive for Crosses, negative for Noughts - because that is the natural
way to read a position. The search is negamax, where every node speaks for whoever is to move in
it, so `weighted_eval` flips the sign for Noughts. Getting that flip wrong gives an engine that
plays well at even depths and badly at odd ones, which is a hard thing to notice and an easy
thing to avoid, so the flip lives in exactly one place.

What is counted is **open twos, and nothing else**: a line holding two of a player's marks and an
empty third, which is one move from winning. Eight lines, one `&` each, counted for both players
and subtracted.

## What this is worth, and when

The unusual thing about this evaluation is that the game it evaluates is solved. TicTacToe
declares SOLVED_DEPTH, so `play.py` searches all nine plies by default, every leaf it reaches is
a finished game scored by `ai.search.terminal_score`, and **this file is never consulted at all**
in a game played at the default depth. It matters only when somebody asks for a shallower search.

So it was measured where it can matter, with `ai.match` against this same search evaluating every
position as zero, 400 games per row:

    depth 1    +329 =0   -71    0.823 +/- 0.019    significant
    depth 2    +117 =234 -49    0.585 +/- 0.016    significant
    depth 3    +156 =167 -77    0.599 +/- 0.018    significant
    depth 4    +77  =246 -77    0.500 +/- 0.016    not significant
    depth 9    +66  =68  -66    0.500 +/- 0.029    not significant

and against a fixed third-party opponent - `ai.search.random_move`, 400 games, the check that
catches a ladder of pairwise wins going nowhere:

    depth 2    open twos 0.930 +/- 0.010     zero 0.922 +/- 0.010
    depth 3    open twos 0.934 +/- 0.010     zero 0.915 +/- 0.011

The shape of that is the whole story. The term is worth a great deal at depth 1, where it is
almost the entire policy; it is worth a real but shrinking amount at 2 and 3; and by depth 4 the
search can see far enough that being told about threats adds nothing it was not about to find.
The last row is not a null result but a confirmation: at SOLVED_DEPTH both players are perfect,
and two perfect players over `ai.match`'s paired openings produce exactly as many wins as losses,
which is what +66 =68 -66 is. The wins are the openings that are already lost when they start.

## Why this game keeps its term and Connect 4 threw its terms away

games/connect4/evaluation.py records several hundred games' worth of evidence that terms which
feel obviously right make an engine worse, and that returning zero is a real answer because it
preserves the centre-first tie-break in `legal_moves`. The same tie-break exists here, and it is
good - `legal_moves` offers the centre, then the corners, then the edges, so a zero evaluation
already plays the opening properly, which is why the zero column above is respectable rather
than hopeless.

The difference is that tic-tac-toe's threats are not heuristic. A line of two with an empty third
*is* a win next move unless it is answered, with no positional judgement in it anywhere, and at
shallow depth that is information the search genuinely does not have. Connect 4's open threes are
the same shape of claim and are not the same quality of information: they can be answered from
below, blocked for free by the opponent's own plan, or floating four rows above the stack and
irrelevant for the rest of the game. Counting a thing is worth doing when the count is a fact.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from games.tictactoe.constants import CROSS, LINE, NOUGHT, WIN_MASKS

if TYPE_CHECKING:
    from games.tictactoe.board import TicTacToe

# Per line that is one mark from winning. The only term, so its size is arbitrary and only its
# sign carries meaning - it is 10 rather than 1 so that a second term could be weighted against
# it without either becoming fractional.
TWO = 10

# Every score has to stay well inside ai.search.MATE, or a merely good position becomes
# indistinguishable from a forced win and the engine stops trying to actually finish games. There
# are eight lines, so `value` cannot exceed 8 * TWO in either direction; this is the round number
# above that, and tests/tictactoe/test_evaluation.py checks the bound holds over every reachable
# position rather than trusting the arithmetic.
MAX_EVAL = 1_000


def bit_count(bits: int) -> int:
    """
    The number of cells in a bitboard.

    `int.bit_count` would do this, but it arrived in Python 3.10 and this project supports 3.7,
    which is why games/chess/bitboard.py and games/connect4/bitboard.py both hand-roll it too.
    """
    return bin(bits).count('1')


def open_twos(marks: int, opponent: int) -> int:
    """
    Lines holding two of `marks` and an empty third: the positions one move from winning.

    A line the opponent has any mark in is dead and is skipped before it is counted, which is the
    difference between counting threats and counting pairs.
    """
    return sum(
        1
        for line in WIN_MASKS
        if not line & opponent and bit_count(line & marks) == LINE - 1
    )


def value(board: 'TicTacToe') -> int:
    """
    How much better Crosses' position is than Noughts'. Positive favours Crosses.

    Exactly zero in a position where neither side has an open two, which keeps the centre-first
    tie-break in `legal_moves` intact wherever this has nothing to say. Says nothing about a
    position that is already won either; that is the search's business, and `outcome` is checked
    before this is reached.
    """
    crosses, noughts = board.marks[CROSS], board.marks[NOUGHT]
    return TWO * (open_twos(crosses, noughts) - open_twos(noughts, crosses))


def weighted_eval(board: 'TicTacToe') -> int:
    """The evaluation the search uses, read from the point of view of the player to move."""
    return value(board) if board.turn else -value(board)
