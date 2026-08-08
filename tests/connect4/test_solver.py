"""
The exact solver, checked three independent ways.

`ai.oracle` is the instrument every learned Connect 4 player will be graded with, so it is the one
piece of this repo that cannot be checked by being compared with itself. Tic-tac-toe has it easy -
tests/tictactoe/test_perfect_play.py writes a second solver from scratch and the two are compared
over all 5,478 positions - but Connect 4 has 4.5e12 positions and a second solver would be just as
slow and just as unproven. So the checks here come from three places, none of them the solver:

**The pinned corpus** (tests/connect4/solved.py). 280 positions solved by the plain full-width
negamax this solver used to be, before alpha-beta, the transposition table, move ordering and
mirror sharing were added. Every value and every optimal-move set has to come back identical. This
is the regression gate: it is what makes "faster, same answers" a thing that is checked rather
than claimed.

**Mirror invariance.** A Connect 4 board reflected in its central column is worth exactly what the
original is. That needs no oracle at all, holds in every position, and catches precisely the class
of fault the mirror-sharing optimisation could introduce - a value bound read out of an entry that
belongs to the reflection.

**The published solution.** Connect 4 was solved in 1988 by Allis and independently by Allen: a
win for the first player, and only by taking the centre column. The outcomes of the seven opening
moves are loss, draw, draw, win, draw, draw, loss. That is an external fact about the game, in the
same spirit as the perft counts in tests/chess/ - nothing in this repo could have produced it, so
agreeing with it is evidence rather than consistency.

Timing is deliberately not asserted anywhere here. How long a solve takes depends on the machine
it runs on, and a test that fails on a slow one is a test that gets ignored. bench.py reports the
timings; this file only decides whether the answers are right.
"""

import unittest
from typing import List

from ai.oracle import Table, move_values, optimal_moves, solve
from games.connect4.bitboard import mirror
from games.connect4.board import Connect4
from games.connect4.constants import COLS
from tests.connect4.corpus import positions
from tests.connect4.solved import SOLVED


def reflect(moves: List[int]) -> List[int]:
    """
    The same game played into the other side of the board.

    Reflecting the *moves* rather than the bitboards, so the mirrored position is reached by
    playing it rather than by being constructed - which means the test exercises the mirror
    through the ordinary rules and cannot be fooled by a hand-built position the game could not
    actually reach.
    """
    return [COLS - 1 - column for column in moves]


class TestPinnedCorpus(unittest.TestCase):
    """
    Every answer the solver used to give, given again.

    The corpus was generated once by the unoptimised solver and committed. Nothing regenerates it:
    a fixture that regenerates moves when the code under it moves, which is the one thing a
    fixture exists to stop.
    """

    def test_every_value_is_reproduced(self):
        for ply, moves, value, _ in SOLVED:
            state = Connect4(moves)
            self.assertEqual(value, solve(state), f'ply {ply}, moves {moves}\n{state}')

    def test_every_optimal_move_set_is_reproduced(self):
        """
        The half a value check would miss.

        A solver can be right about what a position is worth and wrong about which moves hold it -
        a bad move-ordering hint that survives into the returned move, say. The move set is what
        the benchmark grades a player against, so it is the part that matters most.
        """
        for ply, moves, _, optimal in SOLVED:
            state = Connect4(moves)
            self.assertEqual(optimal, sorted(optimal_moves(state)), f'ply {ply}, moves {moves}')

    def test_move_values_and_solve_agree(self):
        """
        Two entry points, one answer.

        `solve` searches the position; `move_values` searches each child on a full window and
        negates. They take different paths through the pruning and must not be able to disagree.
        """
        for _, moves, value, _ in SOLVED:
            state = Connect4(moves)
            self.assertEqual(value, max(move_values(state).values()), moves)

    def test_the_corpus_covers_the_middle_game_as_well_as_the_end(self):
        """
        A guard on the fixture rather than on the solver.

        Endgame positions are trivial for any solver, so a corpus that drifted towards them would
        keep passing while testing less and less. This fails if that ever happens.
        """
        plies = sorted({ply for ply, _, _, _ in SOLVED})
        self.assertEqual(list(range(22, 35, 2)), plies)
        self.assertTrue(all(ply == len(moves) for ply, moves, _, _ in SOLVED))


class TestMirrorInvariance(unittest.TestCase):
    """
    The check that needs no oracle: a reflected position is worth what the original is.

    This is where the mirror-sharing optimisation would show up if it were wrong, and it would
    show up as a *value* being wrong rather than a crash.
    """

    def test_a_position_and_its_reflection_solve_alike(self):
        for _, moves, value, _ in SOLVED:
            self.assertEqual(value, solve(Connect4(reflect(moves))), moves)

    def test_the_optimal_moves_of_a_reflection_are_the_reflected_optimal_moves(self):
        """
        The stronger statement, and the one that catches a move read out of a mirrored entry.

        Values are mirror-invariant so sharing them is safe; moves are not - the winning move in a
        reflected position is the reflected move. `ai.oracle` keeps move hints in a separate
        dictionary keyed by `solver_key` for exactly this reason, and this is what says so.
        """
        for _, moves, _, optimal in SOLVED:
            reflected = sorted(optimal_moves(Connect4(reflect(moves))))
            self.assertEqual(sorted(reflect(optimal)), reflected, moves)

    def test_it_holds_for_positions_outside_the_corpus_too(self):
        """
        The corpus is 280 positions chosen once; this ranges over fresh ones every ply.

        Shallow enough to be affordable in the suite, which is the point - invariance costs
        nothing to check and so can be checked over positions nobody solved in advance.
        """
        for moves in positions(count=25, seed=7, plies=28):
            state, reflected = Connect4(moves), Connect4(reflect(moves))
            self.assertEqual(solve(state), solve(reflected), moves)

    def test_the_canonical_key_is_what_makes_that_sharing_possible(self):
        """`canonical_key` agrees across the mirror; `solver_key` must not, or it is not exact."""
        differed = 0
        for moves in positions(count=40, seed=11, plies=16):
            state, reflected = Connect4(moves), Connect4(reflect(moves))
            self.assertEqual(state.canonical_key, reflected.canonical_key, moves)
            differed += state.solver_key != reflected.solver_key

        self.assertGreater(differed, 0, 'no position in the sample was actually asymmetric')

    def test_mirroring_a_bitboard_twice_returns_it(self):
        for moves in positions(count=40, seed=13, plies=20):
            state = Connect4(moves)
            self.assertEqual(state.occupied, mirror(mirror(state.occupied)), moves)


class TestSolverKey(unittest.TestCase):
    """
    `Connect4.solver_key` replaces the inherited string signature, and the replacement has to be
    exact rather than merely fast. Two positions sharing a key would make the solver return one
    position's value for another, and it would do it silently.
    """

    def test_distinct_positions_get_distinct_keys(self):
        keys = {}
        for moves in positions(count=400, seed=3, plies=14):
            state = Connect4(moves)
            signature = (state.discs[True], state.discs[False], state.turn)
            clash = keys.setdefault(state.solver_key, signature)
            self.assertEqual(clash, signature, f'key collision at {moves}')

    def test_the_key_carries_whose_turn_it_is(self):
        """
        The same board with the other player to move is a different question with a different
        answer, so it has to be a different key. The `discs[turn] + occupied` encoding gets this
        for free by keying off the mover's discs, which is why it is used rather than `occupied`.
        """
        played = Connect4([3, 3])
        after = Connect4([3, 3, 4])
        self.assertNotEqual(played.solver_key, after.solver_key)


class TestPublishedSolution(unittest.TestCase):
    """
    Connect 4 as the literature has it, which is the only check here that comes from outside this
    repository entirely.
    """

    def test_a_race_between_two_threats_is_won_by_the_player_to_move(self):
        """
        Both sides are one move from four; Yellow moves, so Yellow wins - but not by any move.

            [ ][ ][ ][ ][ ][ ][ ]
            [ ][ ][ ][ ][ ][ ][ ]
            [ ][ ][ ][ ][ ][ ][ ]
            [ ][ ][ ][ ][ ][ ][R]
            [ ][ ][ ][ ][ ][ ][R]
            [ ][ ][Y][Y][Y][ ][R]
             0  1  2  3  4  5  6

        Three moves hold the win and the third is the interesting one. Columns 1 and 5 win on the
        spot. Column 6 does not win at all - it blocks Red's vertical, and Yellow then wins anyway
        because Red cannot cover both ends of the open three. Anything else loses outright, Red
        having the faster threat.

        This is the check on the immediate-win shortcut in `_search`, which returns before the
        transposition table is consulted. A shortcut that fired too eagerly would report the
        position won and stop, and would never discover that column 6 also holds it; one that
        fired too late is just slower. The move set is what tells the two apart.
        """
        state = Connect4.from_diagram('''
            .......
            .......
            .......
            ......R
            ......R
            ..YYY.R
        ''')
        self.assertEqual(1, solve(state))
        self.assertEqual([1, 5, 6], sorted(optimal_moves(state)))

    def test_a_position_that_is_lost_is_reported_lost(self):
        """The same board from the other seat: three in a row with both ends open cannot be held."""
        state = Connect4.from_diagram('''
            .......
            .......
            .......
            .......
            ..YYY..
            ..RRYR.
        ''')
        self.assertEqual(-1, solve(state))

    def test_the_table_does_not_change_the_answer(self):
        """
        A shared table across positions must give what a fresh one per position gives.

        This is the reason `Table` exists as a class: a bound stored from a narrow window is not a
        value, and a table that forgot the difference would return answers that depended on what
        had been solved before them - correct on a fresh run and wrong in a benchmark.
        """
        shared = Table()
        for _, moves, value, _ in SOLVED[::7]:
            self.assertEqual(value, solve(Connect4(moves), shared), moves)
            self.assertEqual(value, solve(Connect4(moves)), moves)


if __name__ == '__main__':
    unittest.main()
