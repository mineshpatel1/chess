"""
Perft: every distinct sequence of legal moves of a given length, counted.

Connect 4 had no published reference table and had to derive its counts. Tic-tac-toe is small
enough to have been counted exhaustively long ago, so this is the one game besides chess with a
genuine external oracle - the game has 255,168 playable games, of which the first player wins
131,184, the second wins 77,904 and 46,080 are drawn, over 5,478 reachable positions. Those are
the numbers checked here, and none of them can be produced by a subtly wrong set of rules.

The perft counts themselves argue for part of their own correctness:

  * A win needs three marks, so the first player's third mark at ply 5 is the earliest a game can
    end. Nothing can be decided before then and no cell can be exhausted, so every position up to
    ply 5 offers every empty cell and perft(d) is the falling factorial 9!/(9-d)!.

  * A game won at exactly ply 5 is still a leaf and still counts, so wins do not reduce the count
    until ply 6 - which is why depth 6 is the first count that is not a falling factorial, and so
    the first that says anything the depths above it did not.
"""

import unittest

from ai.perft import divide, traverse_moves
from games.base import DRAW, win
from games.tictactoe.board import TicTacToe
from games.tictactoe.constants import CELLS, CROSS, LINE, NOUGHT
from tests.tictactoe.corpus import DRAWN_GAME, ROW_WIN, reachable_positions

# Sequences of exactly n plies from an empty board, wins counted as leaves and not played on
# through. Depths 1 to 5 are re-derived below rather than trusted; 6 to 9 are pinned.
PERFT = [9, 72, 504, 3024, 15120, 54720, 148176, 200448, 127872]

# How the 255,168 playable games finish. The published census, and the strongest single check on
# the rules there is: any rule implemented too loosely or too tightly moves these.
FIRST_PLAYER_WINS = 131_184
SECOND_PLAYER_WINS = 77_904
DRAWS = 46_080
GAMES = FIRST_PLAYER_WINS + SECOND_PLAYER_WINS + DRAWS

REACHABLE_POSITIONS = 5_478

# The first player's third mark, and so the earliest ply a game can be decided on.
EARLIEST_WIN = LINE * 2 - 1


class TestPerft(unittest.TestCase):
    def test_the_tree_is_unconstrained_for_the_first_five_plies(self):
        """
        Nothing can be won inside five moves, so every node offers every empty cell and the count
        is the falling factorial. Anything generating a move too few or too many shows up here,
        and compounds with depth.
        """
        expected = 1
        for depth in range(1, EARLIEST_WIN + 1):
            expected *= CELLS - depth + 1
            self.assertEqual(expected, traverse_moves(TicTacToe(), depth, False), f'perft({depth})')

    def test_the_first_wins_show_up_at_depth_six(self):
        """
        The first depth whose count is not a falling factorial. 60,480 sequences of six moves
        exist on an empty grid; 5,760 of them are games somebody had already won at ply 5, and
        those are not played on through.
        """
        unconstrained = 9 * 8 * 7 * 6 * 5 * 4
        self.assertEqual(60_480, unconstrained)
        self.assertEqual(54_720, traverse_moves(TicTacToe(), 6, False))
        self.assertEqual(5_760, unconstrained - PERFT[5])

    def test_the_pinned_counts(self):
        for depth, expected in enumerate(PERFT, start=1):
            self.assertEqual(expected, traverse_moves(TicTacToe(), depth, False), f'perft({depth})')

    def test_no_game_can_be_won_before_the_fifth_ply(self):
        """
        The claim the falling factorials rest on. If anything were decided earlier, the
        decided-position rule in ai/perft.py would be truncating lines and those counts would
        not hold.
        """
        self.assertEqual(0, self._decided_positions(TicTacToe(), EARLIEST_WIN - 1))

    def test_the_fifth_ply_does_decide_some_games(self):
        """The complement, so the bound above is exactly where it should be, not merely above it."""
        earliest = TicTacToe(ROW_WIN)
        self.assertEqual(EARLIEST_WIN, len(earliest.move_stack))
        self.assertIsNotNone(earliest.outcome)

    def _decided_positions(self, state: TicTacToe, depth: int) -> int:
        """Distinct move sequences of `depth` plies that finish the game."""
        if state.outcome is not None:
            return 1 if depth == 0 else 0
        if depth == 0:
            return 0

        total = 0
        for move in state.legal_moves:
            state.make_move(move)
            total += self._decided_positions(state, depth - 1)
            state.unmake_move()
        return total

    def test_a_decided_position_has_no_continuations(self):
        """
        What the decided check in ai/perft.py is for. Crosses have taken the top row with four
        cells still empty, so a perft that did not ask would count the marks made afterwards.
        """
        won = TicTacToe(ROW_WIN)
        self.assertIsNotNone(won.outcome)
        self.assertTrue(list(won.legal_moves), 'the point of this test is that moves remain')

        for depth in range(1, 4):
            self.assertEqual(0, traverse_moves(won, depth, False), f'depth {depth}')
        self.assertEqual({}, divide(won, 2))

    def test_a_position_decided_at_the_horizon_is_still_a_leaf(self):
        """
        The ordering half of the same rule. A game finishing at exactly `depth` plies is one
        sequence and counts as one, which is what chess gets free from a mate generating no moves.
        """
        one_from_won = TicTacToe(ROW_WIN[:-1])
        self.assertIsNone(one_from_won.outcome)
        self.assertEqual(CELLS - len(ROW_WIN) + 1, traverse_moves(one_from_won, 1, False))

    def test_divide_accounts_for_the_whole_tree(self):
        counts = divide(TicTacToe(), 4)
        self.assertEqual(CELLS, len(counts))
        self.assertEqual(PERFT[3], sum(counts.values()))


class TestTheCensus(unittest.TestCase):
    """
    Every game played out and its result recorded, against the published totals. Slower than
    perft and worth it: perft counts sequences and cannot tell a won game from a drawn one, so
    the rules could generate exactly the right moves and still decide games wrongly.
    """

    @classmethod
    def setUpClass(cls):
        cls.census = cls._walk(TicTacToe(), {CROSS: 0, NOUGHT: 0, None: 0})

    @staticmethod
    def _walk(state, tally):
        result = state.result
        if result is not None:
            tally[result.winner] += 1
            return tally

        for move in list(state.legal_moves):
            state.make_move(move)
            TestTheCensus._walk(state, tally)
            state.unmake_move()
        return tally

    def test_the_first_player_wins_the_published_number_of_games(self):
        self.assertEqual(FIRST_PLAYER_WINS, self.census[CROSS])

    def test_the_second_player_wins_the_published_number_of_games(self):
        self.assertEqual(SECOND_PLAYER_WINS, self.census[NOUGHT])

    def test_the_published_number_of_games_are_drawn(self):
        self.assertEqual(DRAWS, self.census[None])

    def test_there_are_the_published_number_of_games(self):
        """
        Also the check that every game finishes: the walk only tallies positions that have a
        result, so a line ending in neither a win nor a full board would go uncounted rather than
        fail, and the total would come up short.
        """
        self.assertEqual(GAMES, sum(self.census.values()))
        self.assertEqual(255_168, GAMES)

    def test_there_are_the_published_number_of_reachable_positions(self):
        self.assertEqual(REACHABLE_POSITIONS, sum(1 for _ in reachable_positions()))


class TestOutcomes(unittest.TestCase):
    def test_a_won_game_reports_the_player_who_won_it(self):
        self.assertEqual(win(CROSS), TicTacToe(ROW_WIN).result)

    def test_a_full_board_nobody_won_is_a_draw(self):
        state = TicTacToe(DRAWN_GAME)
        self.assertEqual(DRAW, state.result)
        self.assertEqual([], list(state.legal_moves))


def main():
    unittest.main()


if __name__ == '__main__':
    main()
