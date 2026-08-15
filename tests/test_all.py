"""
The whole suite.

    python3 -m unittest tests.test_all

Every `tests/**/test_*.py` is found rather than listed, so a new test file, or a new game's
`tests/<name>/` package, joins the suite by existing. Tests live beside the game whose positions
they use: `tests/chess/` for everything driven by a chess board, including the tests of `ai/` that
use chess as their vehicle. Only what is shared between games stays here - `tests/conformance.py`,
the contract every game must satisfy.

A test that needs PyTorch or the Rust extension skips itself when it is not there, so this runs
on whatever the machine has.
"""

import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_discovering = False


def load_tests(loader, tests, pattern):
    """
    Discovery, guarded against recursing through this file.

    This module matches the pattern it asks discovery for, so `discover` imports it and calls
    this function again. The second call returns nothing rather than starting a third.
    """
    global _discovering

    if _discovering:
        return unittest.TestSuite()

    _discovering = True
    try:
        return loader.discover(os.path.join(ROOT, 'tests'), top_level_dir=ROOT)
    finally:
        _discovering = False


def main():
    unittest.main()


if __name__ == '__main__':
    main()
