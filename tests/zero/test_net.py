"""
The network, its checkpoints, and the training loop actually learning something.

The only tests here that need PyTorch, which is why they are separated from the search and
self-play tests rather than mixed in with them: the parts of `ai.zero` most likely to be wrong -
the tree and the training targets - are checked with nothing installed, and only the parts that
genuinely are a neural network need the dependency.

Skipped rather than failed when torch is absent, so `tests.test_all` stays runnable on a clean
checkout with no install.
"""

import os
import tempfile
import unittest

try:
    import torch
    TORCH = True
except ImportError:  # pragma: no cover - depends on the environment, not the code
    TORCH = False

from games.tictactoe.board import TicTacToe
from games.tictactoe.encoding import TicTacToeEncoder as Encoder

needs_torch = unittest.skipUnless(TORCH, 'PyTorch is not installed (pip install -r requirements-zero.txt)')


@needs_torch
class TestNetwork(unittest.TestCase):
    def setUp(self):
        from ai.zero.net import ZeroNet
        self.net = ZeroNet(Encoder.PLANE_SHAPE, Encoder.POLICY_SIZE)

    def test_it_is_the_shape_it_was_asked_for(self):
        """Input, two hidden layers of 64, and two heads."""
        self.assertEqual((64, 64), self.net.hidden)
        self.assertEqual(Encoder.POLICY_SIZE, self.net.policy.out_features)
        self.assertEqual(1, self.net.value.out_features)

    def test_a_forward_pass_gives_a_logit_per_action_and_one_value(self):
        from ai.zero.net import to_tensor

        planes = to_tensor([Encoder.planes(TicTacToe()), Encoder.planes(TicTacToe([4]))])
        logits, value = self.net(planes)

        self.assertEqual((2, Encoder.POLICY_SIZE), tuple(logits.shape))
        self.assertEqual((2,), tuple(value.shape))

    def test_the_value_is_bounded(self):
        """tanh, so a value can never wander outside what a game result can be."""
        from ai.zero.net import to_tensor

        _, value = self.net(to_tensor([Encoder.planes(TicTacToe([4, 0]))]))
        self.assertLessEqual(abs(float(value[0])), 1.0)

    def test_it_is_small(self):
        """
        A few thousand parameters. Not a style preference: MCTS calls this once per simulation,
        so a training run makes millions of forward passes and latency is the whole cost.
        """
        self.assertLess(sum(p.numel() for p in self.net.parameters()), 20_000)


@needs_torch
class TestMasking(unittest.TestCase):
    def test_illegal_moves_get_no_probability(self):
        """
        Masked before the softmax, not zeroed after it. Renormalising a distribution that already
        spent mass on moves that do not exist is not the same distribution.
        """
        from ai.zero.net import evaluate, ZeroNet

        state = TicTacToe([4, 0, 8])
        priors, _ = evaluate(ZeroNet(Encoder.PLANE_SHAPE, Encoder.POLICY_SIZE), state, Encoder)

        legal = {Encoder.action_index(move) for move in state.legal_moves}
        for action, prior in enumerate(priors):
            if action not in legal:
                self.assertAlmostEqual(0.0, prior, places=6, msg=f'action {action} is taken')

    def test_the_priors_sum_to_one(self):
        from ai.zero.net import evaluate, ZeroNet

        priors, _ = evaluate(ZeroNet(Encoder.PLANE_SHAPE, Encoder.POLICY_SIZE),
                             TicTacToe([4, 0]), Encoder)
        self.assertAlmostEqual(1.0, sum(priors), places=5)


@needs_torch
class TestCheckpoints(unittest.TestCase):
    def _save(self, directory, game='TicTacToe'):
        from ai.zero.checkpoint import save
        from ai.zero.net import ZeroNet

        net = ZeroNet(Encoder.PLANE_SHAPE, Encoder.POLICY_SIZE)
        path = os.path.join(directory, 'net.pt')
        save(net, path, game=game, generation=7)
        return net, path

    def test_a_checkpoint_rebuilds_the_same_network(self):
        from ai.zero.checkpoint import load
        from ai.zero.net import evaluate

        with tempfile.TemporaryDirectory() as directory:
            net, path = self._save(directory)
            loaded = load(path)['net']

            state = TicTacToe([4, 0])
            self.assertEqual(evaluate(net, state, Encoder), evaluate(loaded, state, Encoder))

    def test_it_carries_what_is_needed_to_rebuild_it(self):
        """A file with weights but not its own shape is one refactor from being unloadable."""
        from ai.zero.checkpoint import load

        with tempfile.TemporaryDirectory() as directory:
            _, path = self._save(directory)
            blob = load(path)

            self.assertEqual('TicTacToe', blob['game'])
            self.assertEqual(7, blob['generation'])
            self.assertEqual(Encoder.POLICY_SIZE, blob['config']['policy_size'])

    def test_a_checkpoint_for_another_game_is_refused(self):
        """
        Nine logits would look perfectly plausible for a seven-column board, and the failure
        would show up as bad play rather than as an error.
        """
        from ai.zero.checkpoint import load

        with tempfile.TemporaryDirectory() as directory:
            _, path = self._save(directory, game='Connect4')
            with self.assertRaises(ValueError):
                load(path, game='TicTacToe')

    def test_a_missing_checkpoint_says_so(self):
        from ai.zero.checkpoint import load
        with self.assertRaises(FileNotFoundError):
            load('/nonexistent/model.pt')


@needs_torch
class TestPlayer(unittest.TestCase):
    def test_a_model_player_returns_legal_moves_with_and_without_search(self):
        from ai.zero.checkpoint import save
        from ai.zero.net import ZeroNet
        from ai.zero.player import model_player

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'net.pt')
            save(ZeroNet(Encoder.PLANE_SHAPE, Encoder.POLICY_SIZE), path, game='TicTacToe')

            for simulations in (0, 5):
                chooser = model_player(path, simulations=simulations)
                for state in (TicTacToe(), TicTacToe([4, 0]), TicTacToe([4, 0, 8, 2])):
                    self.assertIn(chooser(state), list(state.legal_moves),
                                  f'{simulations} simulations')

    def test_the_raw_player_does_not_search(self):
        """Simulations of 0 is the network's own opinion, which is the point of offering it."""
        from ai.zero.checkpoint import save
        from ai.zero.net import ZeroNet
        from ai.zero.player import model_player

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'net.pt')
            save(ZeroNet(Encoder.PLANE_SHAPE, Encoder.POLICY_SIZE), path, game='TicTacToe')
            self.assertEqual(0, model_player(path, 0).simulations)


@needs_torch
class TestTrainingLearns(unittest.TestCase):
    def test_a_short_run_beats_an_untrained_network(self):
        """
        The end-to-end claim, kept small enough for a test suite: a handful of generations must
        move the needle against the oracle. It does not have to reach perfection here - the point
        is that the loop is wired up and pointing the right way, which is exactly what could not
        be said of the 2021 implementation.
        """
        from ai.oracle import benchmark
        from ai.zero.net import ZeroNet, evaluate
        from ai.zero.train import train

        def rate(net):
            def raw(state):
                priors, _ = evaluate(net, state, Encoder)
                return max(state.legal_moves,
                           key=lambda move: priors[Encoder.action_index(move)])
            return benchmark(raw, TicTacToe).overall.rate

        torch.manual_seed(0)
        before = rate(ZeroNet(Encoder.PLANE_SHAPE, Encoder.POLICY_SIZE))
        after = rate(train(TicTacToe, generations=8, games_per_generation=15,
                           simulations=25, seed=0))

        self.assertGreater(after, before + 0.05, f'{before:.1%} -> {after:.1%}')


def main():
    unittest.main()


if __name__ == '__main__':
    main()
