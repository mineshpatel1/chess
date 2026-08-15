"""
The Rust self-play engine against the Python one it copies.

The claim `ai/zero/fast.py` rests on is that the two engines are the same player at different
speeds, and this is where it is checked rather than asserted. Two tests carry it:

* **The tree.** Both searches are handed the same deterministic evaluator - a hash of the
  position, so no network is involved - and must produce identical visit counts. That covers the
  board, the encoder, every float in PUCT and every tie-break, and it needs no PyTorch.
* **The games.** Both engines play a generation with the same network and the same seed and must
  produce identical training examples. That additionally covers the Dirichlet noise and the move
  sampling, because the Rust engine reimplements CPython's Mersenne Twister rather than choosing
  its own random numbers.

Skipped rather than failed when the extension is not built, exactly as the torch tests skip when
torch is absent: a clean checkout runs `tests.test_all` either way.

`python3 -m tests.zero.test_fast --write-fixture` regenerates the pinned answers that
`rust/crates/c4-core/tests/search.rs` checks against without a Python interpreter.
"""

import unittest

try:
    import zero_rs
    FAST = True
except ImportError:  # pragma: no cover - depends on the environment, not the code
    FAST = False

try:
    import numpy
    import torch
    TORCH = True
except ImportError:  # pragma: no cover - depends on the environment, not the code
    TORCH = False

from ai.zero.mcts import MCTS, drive
from games.connect4.board import Connect4
from games.connect4.encoding import Connect4Encoder

needs_fast = unittest.skipUnless(
    FAST, 'the Rust engine is not built (see rust/README.md)')
needs_torch = unittest.skipUnless(
    TORCH, 'PyTorch is not installed (pip install -r requirements-zero.txt)')

MASK64 = (1 << 64) - 1


def _mix(value: int) -> int:
    """SplitMix64, and the same in `rust/crates/c4-core/tests/search.rs`."""
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return value ^ (value >> 31)


def hashed_bitboards(mine, theirs):
    """
    A stand-in for a network: priors and a value that depend only on the position.

    Both engines are handed *this* function rather than each implementing it, so nothing about
    the comparison is left for a second copy to get wrong. Dividing by 2^53 keeps every number a
    float64 with no rounding to disagree about.
    """
    base = _mix(mine ^ _mix(theirs))
    priors = [(_mix(base + column) >> 11) / float(1 << 53) for column in range(7)]
    return priors, (_mix(base + 97) >> 11) / float(1 << 53) * 2.0 - 1.0


def hashed(state):
    """The same evaluator against a `Connect4`, which is what `ai.zero.mcts` asks for."""
    return hashed_bitboards(state.discs[state.turn], state.discs[not state.turn])


# Positions chosen to reach every branch the two implementations could disagree on: the empty
# board, a full column, a game already won before the search starts, and a board one move from
# filling. The searches vary c_puct and the simulation count because both scale the PUCT terms.
POSITIONS = [
    [],
    [3],
    [3, 3],
    [0],
    [3, 3, 3, 3, 3, 3],
    [0, 1, 0, 1, 0],
    [3, 2, 4, 1, 5, 0, 6, 3, 2, 4],
    [3, 3, 4, 4, 5],
    [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
    [1, 2, 1, 2, 1, 2, 5, 5, 5, 5, 5, 5],
    [6, 6, 6, 6, 6, 6, 0, 0, 0, 0, 0, 0],
    [3, 4, 3, 4, 2, 5, 1],
]

SEARCHES = [(5.0, 50), (2.0, 600), (1.5, 200)]


def python_search(columns, exploration, simulations):
    """The Python search's visit counts over the whole action space, and the move it chose."""
    state = Connect4(columns)
    search = MCTS(hashed, Connect4Encoder(), simulations=simulations, exploration=exploration)
    result = drive(search.steps(state, noise=False), hashed)
    return [result.visits.get(column, 0) for column in range(7)], result.move



@needs_fast
class TestTheTreeIsTheSameTree(unittest.TestCase):
    """
    The same evaluator into both searches must give the same tree.

    Nothing here is a tolerance. A PUCT term computed in a different order, children generated in
    a different order, or a tie broken the other way all show up as a different visit count.
    """

    def test_visit_counts_match_position_for_position(self):
        for columns in POSITIONS:
            for exploration, simulations in SEARCHES:
                with self.subTest(columns=columns, c_puct=exploration, simulations=simulations):
                    visits, move = zero_rs.search(
                        columns, exploration, simulations, hashed_bitboards)
                    self.assertEqual(
                        python_search(columns, exploration, simulations), (visits, move))

    def test_the_encoder_sees_the_same_board(self):
        for columns in POSITIONS:
            with self.subTest(columns=columns):
                planes, legal = zero_rs.encode(columns)
                state = Connect4(columns)
                self.assertEqual(Connect4Encoder.planes(state), planes)
                self.assertEqual([column in state.legal_moves for column in range(7)], legal)


@needs_fast
@needs_torch
class TestTheGamesAreTheSameGames(unittest.TestCase):
    """
    The whole engine against the whole Python one, with a real network in between.

    This is the test that covers the parts a fixed evaluator cannot: the Dirichlet noise drawn at
    every root and the move sampled from the visit counts. Both come out of a Mersenne Twister
    seeded per game, and the Rust engine reproduces CPython's rather than choosing its own.

    The generation is played once for the whole class, playing it being the expensive part and
    each test below a different question about the same pair of answers.
    """

    GAMES = 6
    SIMULATIONS = 60
    EXPLORATION = 2.0

    @classmethod
    def setUpClass(cls):
        from ai.zero.fast import play_games as fast_play_games
        from ai.zero.net import ZeroNet, evaluate_batch
        from ai.zero.selfplay import play_games
        from ai.zero.train import architecture

        torch.manual_seed(0)
        cls.encoder = Connect4Encoder()
        cls.net = ZeroNet(
            cls.encoder.PLANE_SHAPE, cls.encoder.POLICY_SIZE, **architecture('Connect4'))
        cls.net.eval()

        cls.played = play_games(
            lambda states: evaluate_batch(cls.net, states, cls.encoder),
            cls.encoder, Connect4, cls.GAMES, cls.SIMULATIONS,
            batch_size=cls.GAMES, seed=1, exploration=cls.EXPLORATION)
        cls.generation = fast_play_games(
            cls.net, cls.GAMES, cls.SIMULATIONS, exploration=cls.EXPLORATION, seed=1)

    def test_the_examples_are_identical(self):
        expected = [example for examples, _ in self.played for example in examples]
        planes, policy, value = self.generation.examples

        self.assertEqual(len(expected), len(planes))
        for index, example in enumerate(expected):
            with self.subTest(position=index):
                self.assertEqual(example.planes, planes[index].tolist())
                self.assertEqual(
                    numpy.array(example.policy, dtype=numpy.float32).tolist(),
                    policy[index].tolist())
                self.assertEqual(numpy.float32(example.value), value[index])

    def test_the_games_run_the_same_length(self):
        self.assertEqual(
            [len(examples) for examples, _ in self.played], self.generation.lengths)

    def test_the_same_games_were_drawn(self):
        drawn = sum(1 for _, finished in self.played if finished.result.winner is None)
        self.assertEqual(drawn, self.generation.drawn)


@needs_fast
class TestChoosingAnEngine(unittest.TestCase):
    """
    `--engine` picks between them, and asking for one by name and not getting it is an error.

    A run started for the speed that quietly takes four hours instead is worse than one that
    refuses, which is the whole difference between `auto` and `rust`.
    """

    def setUp(self):
        from ai.zero.train import choose_engine, games_in_flight_for

        self.choose_engine = choose_engine
        self.games_in_flight_for = games_in_flight_for

    def test_auto_takes_the_rust_engine_where_it_speaks_for_the_game(self):
        from games.tictactoe.board import TicTacToe

        self.assertEqual('rust', self.choose_engine('auto', Connect4))
        self.assertEqual('rust', self.choose_engine(None, Connect4))
        self.assertEqual('python', self.choose_engine('auto', TicTacToe))

    def test_python_is_always_available(self):
        self.assertEqual('python', self.choose_engine('python', Connect4))

    def test_asking_for_rust_and_not_getting_it_raises(self):
        from games.tictactoe.board import TicTacToe

        with self.assertRaises(ValueError):
            self.choose_engine('rust', TicTacToe)

    def test_the_rust_engine_holds_the_whole_generation_in_flight(self):
        """The Python cap is how many trees it can afford to walk; the Rust engine has no cap."""
        self.assertEqual(400, self.games_in_flight_for('rust', None, 400))
        self.assertEqual(32, self.games_in_flight_for('python', None, 400))
        self.assertEqual(64, self.games_in_flight_for('rust', 64, 400))


@needs_fast
@needs_torch
class TestSwappingEnginesChangesNothingButSpeed(unittest.TestCase):
    """
    The integration point, from the caller `train` actually uses.

    `TestTheGamesAreTheSameGames` compares the two drivers at length; this only has to cover the
    layer above them - the seed string, the opening plies and what reaches the replay buffer - so
    it plays a couple of short games rather than a generation.
    """

    def test_a_generation_comes_out_the_same_either_way(self):
        from ai.zero.net import ZeroNet
        from ai.zero.train import _self_play, architecture

        torch.manual_seed(0)
        encoder = Connect4Encoder()
        net = ZeroNet(encoder.PLANE_SHAPE, encoder.POLICY_SIZE, **architecture('Connect4'))

        def generation(engine):
            return _self_play(
                net, encoder, Connect4, count=2, simulations=12, opening_plies=0,
                temperature_moves=30, final_temperature=0.0, exploration=2.0,
                dirichlet_epsilon=0.25, batch_size=2, seed='5:1', report_every=0, engine=engine)

        expected, drawn, lengths = generation('python')
        examples, fast_drawn, fast_lengths = generation('rust')

        self.assertEqual(lengths, fast_lengths)
        self.assertEqual(drawn, fast_drawn)
        self.assertEqual(len(expected), len(examples))
        for index, example in enumerate(expected):
            with self.subTest(position=index):
                self.assertEqual(example.planes, examples[index].planes)
                self.assertEqual(
                    numpy.array(example.policy, dtype=numpy.float32).tolist(),
                    examples[index].policy)
                self.assertEqual(numpy.float32(example.value), examples[index].value)


def write_fixture(path='rust/crates/c4-core/tests/fixtures/search_fixture.rs'):
    """Regenerates the pinned answers the Rust suite checks against without an interpreter."""
    lines = [
        "// The Python search's answers, generated once and pinned. Regenerate with",
        '// `python3 -m tests.zero.test_fast --write-fixture`; see tests/search.rs for what they',
        '// are compared against.',
        '',
        '/// A position, the search that ran on it, and the visit counts and move it produced.',
        "pub type Pinned = (&'static [u8], f64, u32, [u32; 7], u8);",
        '',
        'pub const FIXTURE: &[Pinned] = &[',
    ]
    for columns in POSITIONS:
        for exploration, simulations in SEARCHES:
            visits, move = python_search(columns, exploration, simulations)
            played = ', '.join(str(column) for column in columns)
            lines.append(f'    (&[{played}], {exploration}, {simulations}, {visits}, {move}),')
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
