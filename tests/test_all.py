"""
The whole suite.

Tests live beside the game whose positions they use: `tests/chess/` for everything driven by a
chess board, including the tests of `ai/` that use chess as their vehicle. Only what is shared
between games stays here - `tests/conformance.py`, the contract every game must satisfy.

Adding a game means adding a `tests/<name>/` package and one import block below.
"""

import unittest

from tests.chess.test_board import TestBitboard
from tests.chess.test_moves import TestMoves
from tests.chess.test_undo import TestUndo
from tests.chess.test_permutations import TestPermutations
from tests.chess.test_search import TestSearch, TestTerminalValue, TestUciSearchOutput
from tests.chess.test_search_equivalence import TestHarness, TestDecisivePositions
from tests.chess.test_simulate import TestResult, TestSimulateGame
from tests.chess.test_conformance import TestChessConformance

from tests.connect4.test_bitboard import TestConstants, TestIndexing, TestDrops, TestConnectLength
from tests.connect4.test_board import (
    TestInvariants,
    TestMoveGeneration,
    TestCopy,
    TestSignature,
    TestDiagrams,
    TestRendering,
)
from tests.connect4.test_wins import (
    TestRuns,
    TestExhaustiveWins,
    TestSentinelBoundaries,
    TestOutcome,
)
from tests.connect4.test_permutations import TestPermutations as TestConnect4Permutations
from tests.connect4.test_play import TestPlayLoop, TestParseMove
from tests.connect4.test_search_equivalence import (
    TestEquivalence,
    TestPruningPays,
    TestTerminalScores,
)
from tests.connect4.test_conformance import TestConnect4Conformance


def main():
    unittest.main()


if __name__ == '__main__':
    main()
