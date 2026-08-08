"""
The claim that tic-tac-toe is here to make: the engine plays it perfectly.

Every other game in this project is tested against things that stand in for correctness - perft
counts, pinned search scores, match results - because none of them can be solved. This one can,
so it is tested against the answer.

The oracle is a plain memoised minimax written below, deriving the value of every position from
scratch. It shares nothing with `ai.search` but the rules: no alpha-beta, no move ordering, no
depth counting, no negamax sign flipping. That independence is the whole point, and it is the
same discipline tests/connect4/test_bitboard.py applies when it re-derives the masks rather than
importing them - a test that reuses the machinery under test can only prove it is consistent
with itself.

Three claims, in increasing order of how much they mean:

1. `alpha_beta` at SOLVED_DEPTH returns an optimal move in all 5,478 reachable positions.
2. Its scores agree with the oracle's everywhere, which is the pruning being sound: alpha-beta
   is only worth anything if it returns what an unpruned search would have.
3. The engine cannot be beaten, shown by playing it against *every* possible opponent rather
   than by inferring it from 1 and 2.
"""

import unittest
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from ai.search import alpha_beta, _negamax_ab, LOW_BOUND, HIGH_BOUND
from games.tictactoe.board import TicTacToe, is_win
from games.tictactoe.constants import CELLS, CROSS, FULL_BOARD, NOUGHT
from tests.tictactoe.corpus import reachable_positions

# Values from the point of view of Crosses, who moves first.
CROSS_WINS = 1
NOUGHT_WINS = -1
DRAWN = 0


@lru_cache(maxsize=None)
def solve(crosses: int, noughts: int, crosses_to_move: bool) -> int:
    """
    The value of a position with best play from both sides, as plain minimax over bare ints.

    Deliberately naive. It takes the two mark masks rather than a board so that it cannot
    accidentally depend on TicTacToe, it maximises for Crosses and minimises for Noughts rather
    than negating anything, and it has no notion of depth - so it says nothing about *how fast*
    a win comes, only whether there is one. It is memoised because that turns 255,168 games into
    5,478 positions, and for no other reason.
    """
    if is_win(crosses):
        return CROSS_WINS
    if is_win(noughts):
        return NOUGHT_WINS

    occupied = crosses | noughts
    if occupied == FULL_BOARD:
        return DRAWN

    values = [
        solve(crosses | 1 << cell, noughts, False)
        if crosses_to_move
        else solve(crosses, noughts | 1 << cell, True)
        for cell in range(CELLS)
        if not occupied >> cell & 1
    ]
    return max(values) if crosses_to_move else min(values)


def optimal_moves(state: TicTacToe) -> List[int]:
    """Every move that preserves the value of the position, by the oracle's reckoning."""
    crosses, noughts = state.marks[CROSS], state.marks[NOUGHT]
    best = solve(crosses, noughts, state.turn)

    moves = []
    for cell in state.legal_moves:
        bit = 1 << cell
        if state.turn:
            value = solve(crosses | bit, noughts, False)
        else:
            value = solve(crosses, noughts | bit, True)
        if value == best:
            moves.append(cell)
    return moves


class TestTheOracle(unittest.TestCase):
    """
    The oracle is only worth anything if it is right, and nothing else in the suite checks it,
    so it is checked against the published facts about the game before it is used to judge.
    """

    def test_the_game_is_a_draw(self):
        """The thing everybody knows about tic-tac-toe, and the premise of every test below."""
        self.assertEqual(DRAWN, solve(0, 0, CROSS))

    def test_the_first_player_can_win_if_the_second_blunders(self):
        """A sanity check that the oracle can say anything other than 'draw'."""
        state = TicTacToe([4, 0])  # Crosses centre, Noughts a corner: still drawn
        self.assertEqual(DRAWN, solve(state.marks[CROSS], state.marks[NOUGHT], state.turn))

        state = TicTacToe([4, 1])  # Noughts take an edge instead, which loses
        self.assertEqual(CROSS_WINS, solve(state.marks[CROSS], state.marks[NOUGHT], state.turn))

    def test_every_reachable_position_is_valued(self):
        """5,478 is the published number of reachable positions, terminal ones included."""
        self.assertEqual(5478, sum(1 for _ in reachable_positions()))


class TestPerfectPlay(unittest.TestCase):
    DEPTH = TicTacToe.SOLVED_DEPTH

    def test_the_search_plays_an_optimal_move_everywhere(self):
        """
        The claim, over the whole game rather than a sample of it: in every position that can be
        reached, the move the engine returns is one the oracle is willing to play.

        Not a check that it returns *the* best move - several are usually equally best, and which
        one comes back is a matter of move ordering rather than of correctness.
        """
        for state in reachable_positions():
            if state.is_game_over:
                continue

            move = alpha_beta(state, depth=self.DEPTH)
            self.assertIn(move, optimal_moves(state), f'{state}\nchose {move}')

    def test_alpha_beta_agrees_with_unpruned_minimax(self):
        """
        The pruning is sound: cutting off a branch never changed the answer.

        Compared by sign rather than by number, because the two searches are answering slightly
        different questions - `ai.search` prefers a mate in one to a mate in five and scores them
        differently, while the oracle only knows won, lost and drawn. Sign is the whole of what
        they both claim, and it is what the move choice rests on.

        Connect 4 gives this its own test_search_equivalence.py over a sampled corpus. Here it
        runs over every position there is.
        """
        for state in reachable_positions():
            if state.is_game_over:
                continue

            searched = _negamax_ab(
                state, self.DEPTH, LOW_BOUND, HIGH_BOUND, state.DEFAULT_EVAL,
            )
            # The search speaks for the player to move; the oracle speaks for Crosses.
            expected = solve(state.marks[CROSS], state.marks[NOUGHT], state.turn)
            if not state.turn:
                expected = -expected

            self.assertEqual(
                expected, _sign(searched), f'{state}\nsearch said {searched}',
            )

    def test_the_engine_cannot_be_beaten_as_either_player(self):
        """
        Perfect play, demonstrated rather than inferred: the engine is played against every line
        its opponent has available, exhaustively, first as Crosses and then as Noughts.

        This is the test a person actually cares about - the other two are about the search being
        correct, and this one is about the game being unloseable. It subsumes them and is kept
        alongside them because when it fails they say why.

        Memoised on the position, so the opponent's 255,168 games collapse to the few thousand
        distinct positions the engine can be faced with.
        """
        for engine_plays in (CROSS, NOUGHT):
            state = TicTacToe()
            losses = self._play_every_line(state, engine_plays, {})
            self.assertEqual(
                0, losses, f'the engine lost {losses} lines playing {"first" if engine_plays else "second"}',
            )

    def _play_every_line(
        self, state: TicTacToe, engine_plays: bool, memo: Dict[Tuple[int, int], int],
    ) -> int:
        """Lines below this position that the engine loses, the opponent trying everything."""
        key = (state.marks[CROSS], state.marks[NOUGHT])
        if key in memo:
            return memo[key]

        result = state.result
        if result is not None:
            return 1 if result.winner == (not engine_plays) else 0

        if state.turn == engine_plays:
            state.make_move(alpha_beta(state, depth=self.DEPTH))
            losses = self._play_every_line(state, engine_plays, memo)
            state.unmake_move()
        else:
            losses = 0
            for move in list(state.legal_moves):
                state.make_move(move)
                losses += self._play_every_line(state, engine_plays, memo)
                state.unmake_move()

        memo[key] = losses
        return losses

    def test_the_engine_opens_in_the_centre(self):
        """
        Not a rule of the game, but the consequence of one: every opening move is drawn, so the
        search has nothing to choose between them and takes the first `legal_moves` offers. That
        is the centre, because CENTRE_FIRST orders cells by how many lines run through them.
        """
        self.assertEqual(4, alpha_beta(TicTacToe(), depth=self.DEPTH))

    def test_a_win_is_taken_at_once_rather_than_deferred(self):
        """
        `ai.search.terminal_score` scores a nearer win higher, so an engine with two winning
        lines takes the shorter. Without that a solved engine can shuffle in a won position
        forever, every move being equally 'winning'.
        """
        state = TicTacToe([0, 3, 1, 4])  # Crosses at 0 and 1, Noughts at 3 and 4
        self.assertEqual(2, alpha_beta(state, depth=self.DEPTH), 'cell 2 finishes it now')


def _sign(score: int) -> int:
    return (score > 0) - (score < 0)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
