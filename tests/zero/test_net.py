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
class TestFlushingDenormals(unittest.TestCase):
    """
    The one-line fix that was worth six times the speed of the training loop.

    Weight decay parks unused weights around 1e-40, which float32 can only hold as a subnormal,
    and x86 runs subnormal arithmetic in microcode. By generation 8 of a Connect 4 run 11% of the
    weights were denormal and the network was 6x slower than the same architecture freshly built.

    What these assert is that it is a *speed* change: the count is right, normal weights are left
    alone, and the network's outputs do not move.
    """

    def setUp(self):
        from ai.zero.net import ZeroNet
        self.net = ZeroNet(Encoder.PLANE_SHAPE, Encoder.POLICY_SIZE)

    def _poison(self, count=5, value=1e-40):
        """Puts denormals where weight decay would have left them."""
        with torch.no_grad():
            flat = next(self.net.parameters()).view(-1)
            flat[:count] = value
        return count

    def test_it_zeroes_them_and_says_how_many(self):
        from ai.zero.net import flush_denormals

        self._poison(count=5)
        self.assertEqual(5, flush_denormals(self.net))
        self.assertEqual(0, flush_denormals(self.net), 'a second pass has nothing left to do')

    def test_a_clean_network_reports_none(self):
        """Zero weights are not denormal, and counting them would report thousands from nowhere."""
        from ai.zero.net import flush_denormals

        with torch.no_grad():
            next(self.net.parameters()).view(-1)[:20] = 0.0
        self.assertEqual(0, flush_denormals(self.net))

    def test_the_weights_that_mattered_are_untouched(self):
        from ai.zero.net import flush_denormals

        self._poison()
        before = [p.clone() for p in self.net.parameters()]
        flush_denormals(self.net)

        for was, now in zip(before, self.net.parameters()):
            normal = was.abs() >= 1.18e-38
            self.assertTrue(torch.equal(was[normal], now[normal]))

    def test_the_network_computes_the_same_thing_afterwards(self):
        """
        The claim the whole fix rests on. A weight of 1e-40 in a network whose real weights are
        around 1e-2 contributes nothing a float32 sum can represent, so removing it is free.
        """
        from ai.zero.net import evaluate, flush_denormals

        self._poison(count=200)
        state = TicTacToe([4, 0])
        before = evaluate(self.net, state, Encoder)
        flush_denormals(self.net)
        after = evaluate(self.net, state, Encoder)

        self.assertEqual(before[1], after[1])
        for one, two in zip(before[0], after[0]):
            self.assertEqual(one, two)

    def test_it_survives_a_network_of_every_sign(self):
        """Denormals arrive from both directions; only the magnitude is the test."""
        from ai.zero.net import flush_denormals

        with torch.no_grad():
            flat = next(self.net.parameters()).view(-1)
            flat[0], flat[1] = 1e-40, -1e-40
        self.assertEqual(2, flush_denormals(self.net))


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

    def test_the_optimiser_travels_with_the_weights(self):
        """
        Adam keeps a running mean and variance per parameter. A resume that dropped them would
        restart the moment estimates from zero, so the first steps back would be unmomented -
        which is a strange thing to do to a run precisely when it is being rescued.
        """
        from ai.zero.checkpoint import load, save
        from ai.zero.net import ZeroNet

        with tempfile.TemporaryDirectory() as directory:
            net = ZeroNet(Encoder.PLANE_SHAPE, Encoder.POLICY_SIZE)
            optimiser = torch.optim.Adam(net.parameters(), lr=1e-3)

            logits, value = net(torch.zeros(1, *Encoder.PLANE_SHAPE))
            (logits.sum() + value.sum()).backward()
            optimiser.step()

            path = os.path.join(directory, 'net.pt')
            save(net, path, game='TicTacToe', generation=3, optimiser=optimiser)

            restored = torch.optim.Adam(
                load(path)['net'].parameters(), lr=1e-3)
            restored.load_state_dict(load(path)['optimiser'])
            self.assertEqual(1, list(restored.state.values())[0]['step'].item())

    def test_a_checkpoint_saved_without_one_says_so_rather_than_lying(self):
        from ai.zero.checkpoint import load

        with tempfile.TemporaryDirectory() as directory:
            _, path = self._save(directory)
            self.assertIsNone(load(path)['optimiser'])

    def test_a_half_written_checkpoint_never_replaces_a_good_one(self):
        """
        The thing most likely to interrupt a long run is also most likely to interrupt the write
        meant to survive it, so the write goes somewhere else and is renamed into place.
        """
        from ai.zero.checkpoint import save
        from ai.zero.net import ZeroNet

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'net.pt')
            save(ZeroNet(Encoder.PLANE_SHAPE, Encoder.POLICY_SIZE), path, game='TicTacToe')

            saved = torch.save
            torch.save = lambda *_, **__: (_ for _ in ()).throw(RuntimeError('disk full'))
            try:
                with self.assertRaises(RuntimeError):
                    save(ZeroNet(Encoder.PLANE_SHAPE, Encoder.POLICY_SIZE), path, game='TicTacToe')
            finally:
                torch.save = saved

            from ai.zero.checkpoint import load
            self.assertEqual('TicTacToe', load(path)['game'], 'the good checkpoint was clobbered')


@needs_torch
class TestResuming(unittest.TestCase):
    """
    Picking a killed run back up where it stopped, which is the only thing that makes a run
    measured in hours safe to start on a machine that can go away.
    """

    def _train(self, directory, generations, resume=False):
        from ai.zero.train import train

        return train(
            TicTacToe,
            generations=generations,
            games_per_generation=4,
            simulations=5,
            steps=2,
            benchmark_every=1000,  # Grading is the slow part and this test is not about grading
            latest_path=os.path.join(directory, 'latest.pt'),
            metrics_path=os.path.join(directory, 'run.jsonl'),
            resume_from=os.path.join(directory, 'latest.pt') if resume else None,
            seed=0,
        )

    def test_a_resumed_run_continues_rather_than_replaying(self):
        """
        The bug this exists to catch: resuming from the *best* checkpoint rather than the latest
        one re-runs every generation since the last improvement, and the metrics file grows a
        repeated generation number where the seam is.
        """
        from ai.zero.metrics import read

        with tempfile.TemporaryDirectory() as directory:
            self._train(directory, generations=3)
            self._train(directory, generations=6, resume=True)

            recorded = [entry['generation'] for entry in read(os.path.join(directory, 'run.jsonl'))]
            self.assertEqual([1, 2, 3, 4, 5, 6], recorded)

    def test_it_carries_on_from_the_weights_it_stopped_with(self):
        from ai.zero.checkpoint import load
        from ai.zero.net import evaluate

        with tempfile.TemporaryDirectory() as directory:
            self._train(directory, generations=2)
            stopped = load(os.path.join(directory, 'latest.pt'))
            self.assertEqual(2, stopped['generation'])

            state = TicTacToe([4, 0])
            before = evaluate(stopped['net'], state, Encoder)
            resumed = self._train(directory, generations=3, resume=True)
            self.assertNotEqual(before, evaluate(resumed, state, Encoder),
                                'the resumed run should have moved the weights on')

    def test_the_hook_sees_a_checkpoint_that_matches_the_generation_it_is_told_about(self):
        """
        The pair a hook copies has to be consistent, since `--commit-every` copies it somewhere
        a resume will read it from. Called before the save, the hook gets generation N's metrics
        line and generation N-1's weights, and resuming from that pair replays a generation and
        writes its number into the metrics file twice.
        """
        from ai.zero.checkpoint import load
        from ai.zero.train import train

        seen = []
        with tempfile.TemporaryDirectory() as directory:
            latest = os.path.join(directory, 'latest.pt')
            train(
                TicTacToe, generations=3, games_per_generation=4, simulations=5, steps=2,
                benchmark_every=1000, latest_path=latest, seed=0,
                on_generation=lambda progress: seen.append(
                    (progress.generation, load(latest)['generation'])),
            )

        self.assertEqual([(1, 1), (2, 2), (3, 3)], seen)

    def test_resuming_a_finished_run_does_nothing_rather_than_starting_over(self):
        """Relaunching a run that already finished should be a no-op, not four more hours."""
        from ai.zero.metrics import read

        with tempfile.TemporaryDirectory() as directory:
            self._train(directory, generations=2)
            self._train(directory, generations=2, resume=True)

            recorded = [entry['generation'] for entry in read(os.path.join(directory, 'run.jsonl'))]
            self.assertEqual([1, 2], recorded)


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
        from ai.oracle import benchmark, enumerate_positions
        from ai.zero.net import ZeroNet, evaluate
        from ai.zero.train import train

        def rate(net):
            def raw(state):
                priors, _ = evaluate(net, state, Encoder)
                return max(state.legal_moves,
                           key=lambda move: priors[Encoder.action_index(move)])
            return benchmark(raw, enumerate_positions(TicTacToe)).overall.rate

        torch.manual_seed(0)
        before = rate(ZeroNet(Encoder.PLANE_SHAPE, Encoder.POLICY_SIZE))
        after = rate(train(TicTacToe, generations=8, games_per_generation=15,
                           simulations=25, seed=0))

        self.assertGreater(after, before + 0.05, f'{before:.1%} -> {after:.1%}')


def main():
    unittest.main()


if __name__ == '__main__':
    main()
