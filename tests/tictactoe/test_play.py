"""
The terminal front end, driven by a script instead of a person.

play.py is generic over the registry, so most of what is tested here is really the contract -
that a game can be rendered, prompted for and finished without the loop knowing which game it is.
Tic-tac-toe is a good vehicle because a whole game is nine lines of input.

The exception is `default_depth`, which is genuinely new: it is how a solved game arrives at the
prompt already unbeatable, and it is the one place outside games/tictactoe/ that this game
changed.
"""

import logging
import unittest
from typing import List
from unittest import mock

import log
import play
from games.base import DRAW, win
from games.chess.board import ChessBoard
from games.connect4.board import Connect4
from games.tictactoe.board import TicTacToe
from games.tictactoe.constants import CELLS, CROSS
from tests.tictactoe.corpus import DRAWN_GAME, ROW_WIN


def scripted(lines: List[str]):
    """Stands in for `input`, handing back the next line each time it is called."""
    return mock.patch.object(play, 'input', create=True, side_effect=[str(line) for line in lines])


class TestDefaultDepth(unittest.TestCase):
    """
    The one addition to the contract. A game that can be solved says so, and the front end
    believes it rather than carrying a table of special cases.
    """

    def setUp(self):
        log.setLevel(logging.ERROR)  # `choose_players` prints the menus it is asking about

    def tearDown(self):
        log.setLevel(logging.INFO)

    def test_a_solved_game_defaults_to_searching_all_of_itself(self):
        self.assertEqual(CELLS, TicTacToe.SOLVED_DEPTH)
        self.assertEqual(CELLS, play.default_depth(TicTacToe))

    def test_an_unsolved_game_defaults_to_the_ordinary_depth(self):
        self.assertIsNone(ChessBoard.SOLVED_DEPTH)
        self.assertIsNone(Connect4.SOLVED_DEPTH)
        self.assertEqual(play.DEFAULT_DEPTH, play.default_depth(ChessBoard))
        self.assertEqual(play.DEFAULT_DEPTH, play.default_depth(Connect4))

    def test_the_default_is_offered_at_the_prompt(self):
        """An empty answer takes the default, which for a solved game is perfect play."""
        with scripted(['1', '1', '']):  # computer, computer, accept the depth
            first, second = play.choose_players(TicTacToe)

        state = TicTacToe()
        self.assertEqual(4, first(state), 'the default depth should open in the centre')

    def test_a_typed_depth_overrides_the_default(self):
        with scripted(['1', '1', '1']):
            first, _ = play.choose_players(TicTacToe)

        # Depth 1 is not perfect, but it must still be a legal move rather than a crash.
        self.assertIn(first(TicTacToe()), list(TicTacToe().legal_moves))


class TestPlayLoop(unittest.TestCase):
    """
    The loop renders the board before every move, which is the point of it and is also a screenful
    of noise in a test run, so the log is turned down for the duration.
    """

    def setUp(self):
        log.setLevel(logging.ERROR)

    def tearDown(self):
        log.setLevel(logging.INFO)

    def test_a_game_can_be_played_to_a_win(self):
        state = TicTacToe()
        with scripted(ROW_WIN):
            outcome = play.play(state, play.human_player, play.human_player)

        self.assertEqual(win(CROSS), outcome)
        self.assertEqual(ROW_WIN, state.move_stack)

    def test_a_game_can_be_played_to_a_draw(self):
        """
        The only route to `outcome_without_moves`, and the game that used to be unreachable: its
        last move is cell 0, which `result` treated as no move at all.
        """
        state = TicTacToe()
        with scripted(DRAWN_GAME):
            outcome = play.play(state, play.human_player, play.human_player)

        self.assertEqual(DRAW, outcome)
        self.assertEqual(CELLS, len(state.move_stack))

    def test_the_loop_stops_the_moment_the_game_is_won(self):
        """Input is left over on purpose: the loop must not read it."""
        state = TicTacToe()
        with scripted(ROW_WIN + [5, 6, 7]):
            play.play(state, play.human_player, play.human_player)

        self.assertEqual(len(ROW_WIN), len(state.move_stack))

    def test_a_bad_move_is_refused_and_asked_for_again(self):
        state = TicTacToe()
        with scripted(['what', '9', '3']):
            move = play.human_player(state)

        self.assertEqual(3, move)
        self.assertEqual([], state.move_stack, 'nothing should have been played yet')

    def test_a_taken_cell_is_refused(self):
        state = TicTacToe([4])
        with scripted(['4', '0']):
            self.assertEqual(0, play.human_player(state))

    def test_a_game_between_two_computers_finishes(self):
        """No input at all, so this is the loop and the rules on their own."""
        state = TicTacToe()
        outcome = play.play(state, play.computer_player(1), play.random_move)

        self.assertIsNotNone(outcome)
        self.assertEqual(state.result, outcome)
        self.assertTrue(state.is_game_over)

    def test_two_perfect_players_draw(self):
        """
        What a person accepting every default actually gets. Both sides search the whole game, so
        the result is the game's value, which is a draw.
        """
        state = TicTacToe()
        depth = play.default_depth(TicTacToe)
        outcome = play.play(state, play.computer_player(depth), play.computer_player(depth))

        self.assertEqual(DRAW, outcome)
        self.assertEqual(CELLS, len(state.move_stack))


class TestParseMove(unittest.TestCase):
    def test_the_default_matches_a_move_against_how_it_prints(self):
        """Cells print as their number, so the inherited default already works."""
        state = TicTacToe([4, 0])
        for move in list(state.legal_moves):
            parsed = super(TicTacToe, state).parse_move(str(move))
            self.assertEqual(str(move), str(parsed))

    def test_it_takes_a_cell_number(self):
        state = TicTacToe()
        self.assertEqual(3, state.parse_move(' 3 '))

    def test_a_rejected_move_says_which_kind_of_wrong_it_was(self):
        state = TicTacToe([4])

        with self.assertRaises(ValueError) as unparseable:
            state.parse_move('middle')
        with self.assertRaises(ValueError) as taken:
            state.parse_move('4')
        with self.assertRaises(ValueError) as off_board:
            state.parse_move('9')

        messages = {
            str(unparseable.exception),
            str(taken.exception),
            str(off_board.exception),
        }
        self.assertEqual(3, len(messages), f'the three refusals should differ: {messages}')
        self.assertIn('taken', str(taken.exception))


def main():
    unittest.main()


if __name__ == '__main__':
    main()
