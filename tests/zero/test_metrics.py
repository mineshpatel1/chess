"""
The record a training run leaves behind, and the page made from it.

Neither needs PyTorch: a metrics file is JSON lines and the plotter is string formatting, so the
part of the instrumentation most likely to be reached for after a run has died is also the part
that can be checked on a clean checkout.

The case that matters most is the **half-written file**. A run killed at hour three is exactly the
one whose history is worth reading, and it is the one that ends mid-line - so being able to read
everything before that point is the whole reason the file is JSON lines rather than a document.
"""

import json
import os
import tempfile
import unittest

import plot
from ai.zero.metrics import Recorder, read, series


def record(generation, **fields):
    base = {
        'generation': generation,
        'optimal_rate': 0.5 + generation / 100,
        'value_mse': 1.0 / generation,
        'policy_loss': 2.0 - generation / 50,
        'value_loss': 0.8,
        'draw_rate': 0.25,
        'seconds': 3.0,
        'target_entropy': 1.2,
        'distinct_positions': 100 * generation,
        'game_length': 8.0,
        'first_rate': 0.6,
        'second_rate': 0.55,
    }
    base.update(fields)
    return base


class TestRecording(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, 'run.jsonl')

    def test_a_run_round_trips(self):
        with Recorder(self.path) as recorder:
            for generation in (1, 2, 3):
                recorder.write(record(generation))

        got = read(self.path)
        self.assertEqual([1, 2, 3], [entry['generation'] for entry in got])
        self.assertEqual(record(2), got[1])

    def test_each_generation_is_on_disk_before_the_next_one_starts(self):
        """
        The reason for flushing per line. A run that dies is the one whose history is most worth
        having, and holding it in memory until the end is how that gets lost.
        """
        recorder = Recorder(self.path)
        recorder.write(record(1))
        self.assertEqual(1, len(read(self.path)), 'nothing was flushed')
        recorder.write(record(2))
        self.assertEqual(2, len(read(self.path)))
        recorder.close()

    def test_a_recorder_with_no_path_writes_nowhere_and_does_not_complain(self):
        """So `train` never has to branch on whether recording was asked for."""
        recorder = Recorder(None)
        recorder.write(record(1))
        recorder.close()

    def test_a_half_written_last_line_does_not_lose_the_rest(self):
        with open(self.path, 'w') as handle:
            handle.write(json.dumps(record(1)) + '\n')
            handle.write(json.dumps(record(2)) + '\n')
            handle.write('{"generation": 3, "optimal_ra')  # killed mid-write

        got = read(self.path)
        self.assertEqual([1, 2], [entry['generation'] for entry in got])

    def test_blank_lines_are_skipped(self):
        with open(self.path, 'w') as handle:
            handle.write(json.dumps(record(1)) + '\n\n' + json.dumps(record(2)) + '\n')
        self.assertEqual(2, len(read(self.path)))

    def test_a_series_skips_generations_that_did_not_record_it(self):
        with Recorder(self.path) as recorder:
            recorder.write(record(1, optimal_rate=None))
            recorder.write(record(2))

        self.assertEqual([(2, 0.52)], list(series(read(self.path), 'optimal_rate')))

    def test_a_resumed_run_adds_to_the_file_rather_than_starting_it_again(self):
        """
        The point of recording a long run is the whole curve, and a resume that truncated would
        leave only the part after the failure - the least interesting part.
        """
        with Recorder(self.path) as recorder:
            for generation in (1, 2, 3):
                recorder.write(record(generation))

        with Recorder(self.path, append=True) as recorder:
            recorder.write(record(4))

        self.assertEqual([1, 2, 3, 4], [entry['generation'] for entry in read(self.path)])

    def test_a_fresh_run_does_not_inherit_the_previous_one(self):
        with Recorder(self.path) as recorder:
            recorder.write(record(1))
        with Recorder(self.path) as recorder:
            recorder.write(record(1))

        self.assertEqual(1, len(read(self.path)))

    def test_the_directory_is_created(self):
        nested = os.path.join(self.directory, 'runs', 'deep', 'run.jsonl')
        with Recorder(nested) as recorder:
            recorder.write(record(1))
        self.assertTrue(os.path.exists(nested))


class TestThePlotter(unittest.TestCase):
    """
    A smoke test with teeth: the page has to contain a chart per recorded field and no reference
    to anything it would have to fetch.
    """

    def setUp(self):
        self.records = [record(generation) for generation in range(1, 21)]

    def test_it_renders_a_chart_for_every_field_it_has_data_for(self):
        page = plot.render(self.records, 'Connect4', 'run.jsonl')
        drawn = sum(1 for field, _, _ in plot.CHARTS if plot._points(self.records, field))
        self.assertEqual(drawn, page.count('<svg'))
        self.assertGreater(drawn, 5, 'the fixture should exercise most of the charts')

    def test_the_page_needs_nothing_from_the_network(self):
        """
        No matplotlib, and no CDN either. A page that fetches anything is a page that stops
        rendering the moment it is opened somewhere without a connection.
        """
        page = plot.render(self.records, 'Connect4', 'run.jsonl')
        for fetching in ('http://', 'https://', '<script', '<img'):
            self.assertNotIn(fetching, page)

    def test_the_reference_lines_are_drawn_for_a_game_that_has_them(self):
        page = plot.render(self.records, 'Connect4', 'run.jsonl')
        self.assertIn('minimax:4', page)
        self.assertIn('random', page)

    def test_a_game_without_references_still_renders(self):
        page = plot.render(self.records, 'Chess', 'run.jsonl')
        self.assertIn('<svg', page)

    def test_one_generation_does_not_divide_by_zero(self):
        """A run inspected after its first generation is the most likely time to look at it."""
        page = plot.render([record(1)], 'Connect4', 'run.jsonl')
        self.assertIn('<svg', page)

    def test_a_flat_series_does_not_divide_by_zero(self):
        flat = [record(generation, value_mse=0.5) for generation in range(1, 5)]
        self.assertIn('<svg', plot.render(flat, 'Connect4', 'run.jsonl'))

    def test_the_game_is_guessed_from_the_filename(self):
        self.assertEqual('Connect4', plot._guess_game('runs/connect4-first-try.jsonl'))
        self.assertEqual('TicTacToe', plot._guess_game('/tmp/tic_tac_toe.jsonl'))
        self.assertIsNone(plot._guess_game('runs/experiment.jsonl'))

    def test_the_title_and_source_are_escaped(self):
        """The source is a path from the command line, so it is not to be trusted into HTML."""
        page = plot.render(self.records, 'Connect4', '<script>alert(1)</script>')
        self.assertNotIn('<script>', page)


if __name__ == '__main__':
    unittest.main()
