"""
The parts of `zero.py train` that keep a long run alive across a machine going away.

Neither needs PyTorch - resuming is a decision about a path, and committing is a subprocess - so
they are checked here rather than alongside the training loop.

The committer is worth a real repository rather than a mock. What it has to get right is exactly
what a mock would assume: that `git add` of a path under `.gitignore` quietly stages nothing, that
committing with nothing staged is an error rather than a no-op, and that the resume point actually
lands in a commit. A four-hour run whose insurance silently does nothing is worse than one with no
insurance at all, because it is not being watched.
"""

import os
import subprocess
import tempfile
import unittest
from typing import NamedTuple

import zero


class Fake(NamedTuple):
    """Enough of a `Progress` for the committer, which reads it only to write a commit message."""

    generation: int
    optimal_rate: float = 0.5
    value_mse: float = 0.9
    ladder_score: float = 0.4
    highest_rung: str = ''


def git(*arguments, cwd):
    return subprocess.run(['git', *arguments], cwd=cwd, capture_output=True, text=True)


class TestResumeFrom(unittest.TestCase):
    class Args(NamedTuple):
        resume: bool
        latest: str

    def test_it_reads_the_latest_checkpoint_when_asked_to_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'latest.pt')
            open(path, 'w').close()
            self.assertEqual(path, zero._resume_from(self.Args(True, path)))

    def test_a_run_that_was_not_asked_to_resume_starts_over(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'latest.pt')
            open(path, 'w').close()
            self.assertIsNone(zero._resume_from(self.Args(False, path)))

    def test_resuming_with_nothing_to_resume_from_starts_at_the_beginning(self):
        """
        A run killed in its first few minutes should restart on the same command line, not need a
        different one composed at the least convenient moment.
        """
        self.assertIsNone(zero._resume_from(self.Args(True, '/nonexistent/latest.pt')))


class TestCommitter(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.repo = os.path.join(self.directory, 'work')
        remote = os.path.join(self.directory, 'remote.git')

        git('init', '--bare', remote, cwd=self.directory)
        git('init', self.repo, cwd=self.directory)
        for setting, value in (('user.email', 'a@b.c'), ('user.name', 'Test'), ('commit.gpgsign', 'false')):
            git('config', setting, value, cwd=self.repo)
        git('remote', 'add', 'origin', remote, cwd=self.repo)

        self.latest = os.path.join(self.repo, 'latest.pt')
        self.metrics = os.path.join(self.repo, 'run.jsonl')

        self.previous = os.getcwd()
        os.chdir(self.repo)

    def tearDown(self):
        os.chdir(self.previous)

    def _write(self, text):
        for path in (self.latest, self.metrics):
            with open(path, 'w') as handle:
                handle.write(text)

    def _subjects(self):
        finished = git('log', '--format=%s', cwd=self.repo)
        return [line for line in finished.stdout.splitlines() if line]

    def test_it_commits_the_run_on_its_schedule_and_not_between(self):
        commit = zero._committer([self.latest, self.metrics], every=3)

        for generation in (1, 2, 3, 4, 5, 6):
            self._write(f'generation {generation}')
            commit(Fake(generation))

        self.assertEqual(2, len(self._subjects()), self._subjects())
        self.assertIn('generation 6', self._subjects()[0])

    def test_what_lands_in_the_commit_is_what_a_resume_needs(self):
        commit = zero._committer([self.latest, self.metrics], every=1)
        self._write('anything')
        commit(Fake(1))

        listed = git('show', '--name-only', '--format=', 'HEAD', cwd=self.repo).stdout.split()
        self.assertEqual(['latest.pt', 'run.jsonl'], sorted(listed))

    def test_a_generation_that_changed_nothing_does_not_make_an_empty_commit(self):
        commit = zero._committer([self.latest, self.metrics], every=1)
        self._write('unchanged')
        commit(Fake(1))
        commit(Fake(2))

        self.assertEqual(1, len(self._subjects()))

    def test_a_file_that_does_not_exist_yet_is_skipped_rather_than_fatal(self):
        """`--metrics` is optional, and the first commit can land before a file is written."""
        commit = zero._committer([self.latest, self.metrics], every=1)
        with open(self.latest, 'w') as handle:
            handle.write('only this one')

        commit(Fake(1))
        self.assertEqual(['latest.pt'],
                         git('show', '--name-only', '--format=', 'HEAD', cwd=self.repo).stdout.split())

    def test_the_run_survives_git_failing(self):
        """
        Losing four hours of self-play to a failed `git push` would be a strange way to make a
        run more robust, so nothing here raises.
        """
        git('remote', 'set-url', 'origin', '/nonexistent/remote.git', cwd=self.repo)
        commit = zero._committer([self.latest, self.metrics], every=1)
        self._write('anything')

        commit(Fake(1))  # Must not raise
        self.assertEqual(1, len(self._subjects()), 'the commit should still have been made')

    def test_committing_is_off_unless_asked_for(self):
        self.assertIsNone(zero._committer([self.latest], every=0))
        self.assertIsNone(zero._committer([None, None], every=5))


if __name__ == '__main__':
    unittest.main()
