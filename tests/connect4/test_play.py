"""
The terminal front end, driven by a script instead of a person.

play.py is generic over the registry, so most of what is tested here is really the contract:
that a game can be rendered, prompted for and finished without the loop knowing which game it
is. Connect 4 is the vehicle because a whole game of it fits in a test.
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
from games.connect4.constants import FULL_BOARD, YELLOW
from tests.connect4.corpus import DRAWN_GAME, HORIZONTAL_WIN


def scripted(lines: List[str]):
    """Stands in for `input`, handing back the next line each time it is called."""
    return mock.patch.object(play, 'input', create=True, side_effect=[str(line) for line in lines])


class TestPlayLoop(unittest.TestCase):
    """
    The loop renders the board before every move, which is the point of it and is also forty
    boards of noise in a test run, so the log is turned down for the duration.
    """

    def setUp(self):
        log.setLevel(logging.ERROR)

    def tearDown(self):
        log.setLevel(logging.INFO)

    def test_a_game_can_be_played_to_a_win(self):
        state = Connect4()
        with scripted(HORIZONTAL_WIN):
            outcome = play.play(state, play.human_player, play.human_player)

        self.assertEqual(win(YELLOW), outcome)
        self.assertEqual(HORIZONTAL_WIN, state.columns_played)

    def test_a_game_can_be_played_to_a_draw(self):
        """
        The only route to `outcome_without_moves`. A win ends the game with columns to spare,
        so without a full board this branch of the rules is never reached at all.
        """
        state = Connect4()
        with scripted(DRAWN_GAME):
            outcome = play.play(state, play.human_player, play.human_player)

        self.assertEqual(DRAW, outcome)
        self.assertEqual(FULL_BOARD, state.occupied)

    def test_the_loop_stops_the_moment_the_game_is_won(self):
        """Input is left over on purpose: the loop must not read it."""
        state = Connect4()
        with scripted(HORIZONTAL_WIN + [0, 0, 0]):
            play.play(state, play.human_player, play.human_player)

        self.assertEqual(len(HORIZONTAL_WIN), len(state.move_stack))

    def test_a_bad_move_is_refused_and_asked_for_again(self):
        state = Connect4()
        with scripted(['what', '9', '3']):
            move = play.human_player(state)

        self.assertEqual(3, move)
        self.assertEqual([], state.move_stack, 'nothing should have been played yet')

    def test_a_full_column_is_refused(self):
        state = Connect4([0, 0, 0, 0, 0, 0])
        with scripted(['0', '1']):
            self.assertEqual(1, play.human_player(state))

    def test_a_game_between_two_computers_finishes(self):
        """No input at all, so this is the loop and the rules on their own."""
        state = Connect4()
        outcome = play.play(state, play.computer_player(1), play.random_move)

        self.assertIsNotNone(outcome)
        self.assertEqual(state.result, outcome)
        self.assertTrue(state.is_game_over)

    def test_the_computer_answers_with_a_legal_move(self):
        state = Connect4([3, 3, 2])
        self.assertIn(play.computer_player(2)(state), list(state.legal_moves))

    def test_a_random_player_is_depth_zero(self):
        self.assertIs(play.random_move, play.computer_player(0))


class TestParseMove(unittest.TestCase):
    """
    The one addition to the GameState contract. It has a working default, so what matters is
    that both games still round-trip through whatever they print.
    """

    def test_the_default_matches_a_move_against_how_it_prints(self):
        for state in (Connect4([3, 3, 2]), ChessBoard()):
            for move in list(state.legal_moves):
                # The default, reached explicitly, rather than either game's override
                parsed = super(type(state), state).parse_move(str(move))
                self.assertEqual(str(move), str(parsed))

    def test_connect_four_takes_a_column_number(self):
        state = Connect4()
        self.assertEqual(3, state.parse_move(' 3 '))

        with self.assertRaises(ValueError):
            state.parse_move('seven')
        with self.assertRaises(ValueError):
            state.parse_move('7')
        with self.assertRaises(ValueError):
            Connect4([2, 2, 2, 2, 2, 2]).parse_move('2')

    def test_chess_takes_uci(self):
        board = ChessBoard()
        self.assertEqual('e2e4', board.parse_move('E2E4').uci)

        with self.assertRaises(ValueError):
            board.parse_move('hello')
        with self.assertRaises(ValueError):
            board.parse_move('e2e5')

    def test_a_rejected_move_says_which_kind_of_wrong_it_was(self):
        board = ChessBoard()
        with self.assertRaises(ValueError) as unparseable:
            board.parse_move('zz')
        with self.assertRaises(ValueError) as illegal:
            board.parse_move('e2e5')

        self.assertNotEqual(str(unparseable.exception), str(illegal.exception))


def main():
    unittest.main()


if __name__ == '__main__':
    main()
