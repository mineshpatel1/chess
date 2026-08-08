"""
The tests every game in this project must pass.

A game that satisfies GameState's signatures can still be wrong in ways the search will
quietly build on: an unmake that restores almost everything, a copy that shares state with its
original, an `outcome` that disagrees with `result`. Those failures do not look like failures.
They look like an engine that plays slightly badly.

So a game subclasses GameConformanceTests, fills in the three hooks below, and gets cover for
all of it. To add a game:

    class TestTicTacToe(GameConformanceTests, unittest.TestCase):
        def new_game(self):
            return TicTacToe()

        def decided_games(self):
            return [(TicTacToe(ROW_WIN), win(CROSS)), (TicTacToe(DRAWN_GAME), DRAW)]

        def forced_win_in_one(self):
            return TicTacToe([0, 3, 1, 4]), CROSS

That sketch was written before tic-tac-toe existed, as an illustration. It exists now, and
tests/tictactoe/test_conformance.py is the real thing.

`unittest` collects tests from the mixin only where it is combined with TestCase, so the
shared methods run once per game rather than once on their own.
"""

import random
from typing import List, Optional, Tuple

from ai.search import alpha_beta, random_move
from ai.simulate import simulate_game
from games.base import GameState, Outcome, Player, has_moves


class GameConformanceTests:
    """
    Mix into a TestCase alongside the three hooks below.

    PLAYOUTS and PLAYOUT_PLIES trade coverage against suite time; a game with a small state
    space can afford to raise them.
    """

    PLAYOUTS = 12
    PLAYOUT_PLIES = 24
    SEARCH_DEPTH = 2

    # ---- hooks a game fills in -------------------------------------------------------

    def new_game(self) -> GameState:
        """A fresh state at the starting position."""
        raise NotImplementedError

    def decided_games(self) -> List[Tuple[GameState, Outcome]]:
        """Finished positions and the result each should report."""
        raise NotImplementedError

    def forced_win_in_one(self) -> Optional[Tuple[GameState, Player]]:
        """
        A position where the player to move wins immediately, and who that is. Return None if
        the game has no such position that is easy to write down.
        """
        return None

    # ---- helpers ---------------------------------------------------------------------

    def _walk(self, seed: int = 0):
        """Yields states along a seeded random playout, stopping when the game ends."""
        rng = random.Random(seed)
        state = self.new_game()
        for _ in range(self.PLAYOUT_PLIES):
            if state.is_game_over:
                return
            yield state
            state.make_move(rng.choice(list(state.legal_moves)))

    @staticmethod
    def _identity(state: GameState):
        """
        Everything about a state that a move should change and an unmake should put back.

        Built on `signature` rather than on the printed board. An undo that drops the en
        passant square leaves a board that prints identically and plays differently, and
        comparing pictures cannot tell the two apart - which is exactly the bug this is here
        to catch.
        """
        return state.signature, state.turn, sorted(str(m) for m in state.legal_moves)

    # ---- the tests -------------------------------------------------------------------

    def test_a_new_game_is_not_over(self):
        state = self.new_game()
        self.assertIsNone(state.result)
        self.assertFalse(state.is_game_over)
        self.assertTrue(has_moves(state.legal_moves))

    def test_a_position_with_moves_left_is_not_finished(self):
        """
        Whether a game is over is a question about how many moves there are, never about what
        the moves are worth - and `any(moves)` quietly conflates the two. Both small games here
        number their moves from zero, so move 0 is legal and falsy, and `GameState.result` used
        to call a position finished when move 0 was the only one left: Connect 4 drawn with four
        cells free in a column, tic-tac-toe drawn with a corner still empty.

        The walk drives itself by `legal_moves` and `outcome` rather than by `is_game_over`,
        because `is_game_over` is the thing under test - a walk that trusted it would stop at
        the very position that breaks it, which is how this survived Connect 4 being added.
        """
        for seed in range(self.PLAYOUTS):
            rng = random.Random(seed)
            state = self.new_game()

            for _ in range(self.PLAYOUT_PLIES):
                if state.outcome is not None:
                    break
                moves = list(state.legal_moves)
                if not moves:
                    break

                self.assertIsNone(state.result, f'{state} still has {len(moves)} moves')
                self.assertFalse(state.is_game_over, str(state))
                state.make_move(rng.choice(moves))

    def test_unmake_restores_the_position_exactly(self):
        """
        The search walks one state up and down the tree instead of copying at every node, so
        an unmake that restores nearly everything corrupts every sibling branch after it.
        """
        for state in self._walk():
            before = self._identity(state)
            for move in list(state.legal_moves):
                state.make_move(move)
                state.unmake_move()
                self.assertEqual(before, self._identity(state), f'after {move} on {state}')

    def test_moves_can_be_unmade_all_the_way_back(self):
        """Undo has to survive a whole game, not just one move at a time."""
        rng = random.Random(7)
        state = self.new_game()
        history = []

        while not state.is_game_over and len(history) < self.PLAYOUT_PLIES:
            history.append(self._identity(state))
            state.make_move(rng.choice(list(state.legal_moves)))

        for expected in reversed(history):
            state.unmake_move()
            self.assertEqual(expected, self._identity(state))

    def test_every_generated_move_is_playable(self):
        for state in self._walk(seed=1):
            for move in list(state.legal_moves):
                state.make_move(move)
                state.unmake_move()

    def test_a_move_passes_the_turn(self):
        for state in self._walk(seed=2):
            turn = state.turn
            for move in list(state.legal_moves):
                state.make_move(move)
                self.assertNotEqual(turn, state.turn, f'{move} did not pass the turn')
                state.unmake_move()

    def test_copy_is_equal_and_independent(self):
        """
        Root moves are searched on copies, in separate processes for some games. A copy that
        shares mutable state with its original corrupts the position it was taken from.
        """
        for state in self._walk(seed=3):
            clone = state.copy()
            self.assertEqual(self._identity(state), self._identity(clone))

            before = self._identity(state)
            for _ in range(3):  # Each move re-read from the clone, which the last one changed
                moves = list(clone.legal_moves)
                if not moves:
                    break
                clone.make_move(moves[0])
            self.assertNotEqual(before, self._identity(clone), 'the copy did not actually move')
            self.assertEqual(before, self._identity(state), 'mutating a copy moved the original')

    def test_a_copy_is_interchangeable_with_its_original(self):
        """
        Stronger than the test above, and deliberately not built on `_identity`.

        `_identity` is `signature` plus the move list, so a copy that drops something
        `signature` does not mention is *equal* to its original and does not *behave* like it.
        Chess did exactly that: `copy` rebuilt the board from its FEN, which carries castling,
        en passant and the clocks but not the repetition history, so a copy of a drawn game had
        the same signature and was still running.

        This compares what a caller can actually observe instead - the moves, both halves of the
        result, and whose turn it is. Anything a copy is allowed to leave behind must be
        invisible here, which is the real definition of what `copy` may drop.
        """
        for state in self._walk(seed=5):
            clone = state.copy()
            self.assertEqual(
                self._observable(state), self._observable(clone), f'the copy differs\n{state}',
            )

    @staticmethod
    def _observable(state: GameState):
        """What a caller can see. Two states equal here have to be interchangeable."""
        return (
            sorted(str(move) for move in state.legal_moves),
            str(state.outcome),
            str(state.result),
            state.turn,
        )

    def test_decided_games_report_their_result(self):
        for state, expected in self.decided_games():
            self.assertEqual(expected, state.result, str(state))
            self.assertTrue(state.is_game_over, str(state))

    def test_outcome_never_contradicts_result(self):
        """
        `outcome` is the cheap half the search trusts at every node. If it can claim a win the
        full `result` does not agree with, the search is scoring positions that are not over.
        """
        for seed in range(self.PLAYOUTS):
            for state in self._walk(seed=seed):
                outcome = state.outcome
                if outcome is not None:
                    self.assertEqual(outcome, state.result, str(state))

    def test_a_running_game_is_never_reported_as_won(self):
        for seed in range(self.PLAYOUTS):
            for state in self._walk(seed=seed):
                self.assertIsNone(state.result, f'{state} ended but the walk continued')

    def test_the_search_returns_a_legal_move(self):
        for state in self._walk(seed=4):
            move = alpha_beta(state, depth=self.SEARCH_DEPTH)
            self.assertIn(str(move), [str(m) for m in state.legal_moves], str(state))

    def test_the_search_takes_a_win_it_can_see(self):
        position = self.forced_win_in_one()
        if position is None:
            self.skipTest('no forced win written down for this game')

        state, winner = position
        self.assertEqual(winner, state.turn)

        move = alpha_beta(state, depth=self.SEARCH_DEPTH)
        state.make_move(move)
        self.assertEqual(winner, state.result.winner, f'{move} did not finish it')

    def test_a_game_between_random_players_finishes(self):
        """
        The loop, the rules and the terminal conditions together. If a game can reach a state
        that is neither playable nor over, this is what hangs on it.
        """
        for seed in range(3):
            state = self.new_game()
            random.seed(seed)
            outcome = simulate_game(state, random_move, random_move, print_summary=False)
            self.assertIsNotNone(outcome)
            self.assertEqual(outcome, state.result)
