"""
The Rust ladder driver against the Python paths it replaces.

Two claims, proven two different ways:

* **The game.** `play_one` is a hand-rolled reference: the challenger's MCTS against
  `ai.search.alpha_beta`, move for move, over the hashed evaluator `tests/zero/test_fast.py`
  already uses - no network and no torch needed. `write_fixture` pins its answers, and
  `rust/crates/c4-core/tests/ladder.rs::challenger_matches_the_pinned_python_answers` replays them
  with the same evaluator ported to Rust, against `LadderMatch`'s own unbatched reference - which
  that file separately proves plays identically to the batched driver. Nothing here needs to drive
  `zero_rs.LadderMatch` directly to make that claim; the transitivity is the point.
* **The integration.** `ai.zero.train._climb` - the actual call a training run makes - with a real
  network, `engine='python'` against `engine='rust'`, both pinned to the CPU. A GPU batched
  forward pass is not guaranteed bit-identical to a CPU single-position one, so this checks what
  can be promised on one device and leaves the GPU's speed to `bench.py`.

Skipped rather than failed when the extension or torch is absent, exactly as `test_fast.py` does.

`python3 -m tests.zero.test_ladder_fast --write-fixture` regenerates the pinned answers
`rust/crates/c4-core/tests/ladder.rs` checks against without a Python interpreter.
"""

import unittest

try:
    import zero_rs
    FAST = hasattr(zero_rs, 'LadderMatch')
except ImportError:  # pragma: no cover - depends on the environment, not the code
    FAST = False

try:
    import torch
    TORCH = True
except ImportError:  # pragma: no cover - depends on the environment, not the code
    TORCH = False

from ai.search import alpha_beta
from ai.zero.mcts import MCTS
from games.connect4.board import Connect4
from games.connect4.encoding import Connect4Encoder

needs_fast = unittest.skipUnless(
    FAST, 'the Rust ladder driver is not built (see rust/README.md)')
needs_torch = unittest.skipUnless(
    TORCH, 'PyTorch is not installed (pip install -r requirements-zero.txt)')

MASK64 = (1 << 64) - 1


def _mix(value: int) -> int:
    """SplitMix64, and the same in `rust/crates/c4-core/tests/ladder.rs::mix`."""
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return value ^ (value >> 31)


def hashed_bitboards(mine, theirs):
    """A stand-in for the network, the same one `tests/zero/test_fast.py` pins its fixture with."""
    base = _mix(mine ^ _mix(theirs))
    priors = [(_mix(base + column) >> 11) / float(1 << 53) for column in range(7)]
    return priors, (_mix(base + 97) >> 11) / float(1 << 53) * 2.0 - 1.0


def hashed(state):
    """The same evaluator against a `Connect4`, which is what `ai.zero.mcts` asks for."""
    return hashed_bitboards(state.discs[state.turn], state.discs[not state.turn])


def play_one(columns, depth, simulations, exploration, challenger_is_first):
    """
    One ladder game against the hashed evaluator, played straight through: the challenger's MCTS
    where `state.turn == challenger_is_first`, `ai.search.alpha_beta` otherwise. The Python
    reference `rust/crates/c4-core/tests/ladder.rs`'s pinned-fixture test replays.
    """
    state = Connect4(columns)
    moves = []

    if not state.is_game_over and state.turn != challenger_is_first:
        move = alpha_beta(state, depth=depth)
        state.make_move(move)
        moves.append(move)

    while not state.is_game_over:
        # No noise, and the most-visited move rather than a sample: a player being measured gives
        # its actual opinion, exactly `ai.zero.player.model_player`'s ladder chooser.
        search = MCTS(hashed, Connect4Encoder, simulations=simulations, exploration=exploration)
        move = search.search(state, noise=False).move
        state.make_move(move)
        moves.append(move)

        if state.is_game_over:
            break
        move = alpha_beta(state, depth=depth)
        state.make_move(move)
        moves.append(move)

    return moves, state.result.winner, challenger_is_first


# A handful of positions, a couple of depths and simulation counts, both colours - enough to
# exercise the branches a batched driver could get wrong: an opponent-first opening, a challenger
# search that ends the game outright, and one that hands back to the opponent.
CORPUS_COUNT = 6
CORPUS_SEED = 7
CORPUS_PLIES = 6
CONFIGS = [(2, 24, 1.5), (3, 12, 2.0)]


@needs_fast
@needs_torch
class TestTheChallengerIsTheSameChallenger(unittest.TestCase):
    """
    `ai.zero.train._climb` end to end, `engine='python'` against `engine='rust'` - the actual call
    a training run makes, not just the driver underneath it.
    """

    def setUp(self):
        from ai.zero.net import ZeroNet
        from ai.zero.train import _climb, architecture

        torch.manual_seed(0)
        self.climb = _climb
        self.encoder = Connect4Encoder
        self.net = ZeroNet(
            self.encoder.PLANE_SHAPE, self.encoder.POLICY_SIZE, **architecture('Connect4'))
        self.net.eval()

    def test_the_standing_is_identical_either_way(self):
        kwargs = dict(rungs=['minimax:2'], games=8, seed=3, simulations=16)
        python_standing = self.climb(self.net, self.encoder, Connect4, engine='python', **kwargs)
        rust_standing = self.climb(self.net, self.encoder, Connect4, engine='rust', **kwargs)

        self.assertEqual(len(python_standing.rungs), 1)
        self.assertEqual(python_standing.rungs[0].spec, rust_standing.rungs[0].spec)
        self.assertEqual(python_standing.rungs[0].result, rust_standing.rungs[0].result)


def write_fixture(path='rust/crates/c4-core/tests/fixtures/ladder_fixture.rs'):
    """Regenerates the pinned answers `rust/crates/c4-core/tests/ladder.rs` checks without torch."""
    from tests.connect4.corpus import positions

    lines = [
        "// The Python reference's answers, generated once and pinned. Regenerate with",
        '// `python3 -m tests.zero.test_ladder_fast --write-fixture`; see tests/ladder.rs for',
        '// what they are compared against.',
        '',
        '/// An opening, the rung played against it, and how the game came out.',
        "pub type Pinned = (&'static [u8], i32, f64, u32, &'static [u8], Option<bool>, bool);",
        '',
        'pub const FIXTURE: &[Pinned] = &[',
    ]
    for columns in positions(CORPUS_COUNT, seed=CORPUS_SEED, plies=CORPUS_PLIES):
        for depth, simulations, exploration in CONFIGS:
            for challenger_is_first in (True, False):
                moves, winner, first = play_one(
                    columns, depth, simulations, exploration, challenger_is_first)
                played_columns = ', '.join(str(column) for column in columns)
                played_moves = ', '.join(str(move) for move in moves)
                winner_rs = 'None' if winner is None else f'Some({str(winner).lower()})'
                lines.append(
                    f'    (&[{played_columns}], {depth}, {exploration}, {simulations}, '
                    f'&[{played_moves}], {winner_rs}, {str(first).lower()}),'
                )
    lines.append('];')

    with open(path, 'w') as fixture:
        fixture.write('\n'.join(lines) + '\n')
    return path


if __name__ == '__main__':  # pragma: no cover - a maintenance command, not a test
    import sys

    if '--write-fixture' in sys.argv:
        print(f'wrote {write_fixture()}')
    else:
        unittest.main()
