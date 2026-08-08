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
from tests.connect4.test_evaluation import (
    TestSymmetry,
    TestBounds,
    TestThreatCells,
    TestThreatValue,
)
from tests.connect4.test_match import TestMatchResult, TestPlayMatch
from tests.connect4.test_conformance import TestConnect4Conformance

from tests.tictactoe.test_board import (
    TestGeometry,
    TestIsWin,
    TestMoveGeneration as TestTicTacToeMoveGeneration,
    TestPlayingAndUndoing,
    TestCopy as TestTicTacToeCopy,
    TestSignature as TestTicTacToeSignature,
    TestDiagrams as TestTicTacToeDiagrams,
    TestRendering as TestTicTacToeRendering,
)
from tests.tictactoe.test_permutations import (
    TestPerft,
    TestTheCensus,
    TestOutcomes,
)
from tests.tictactoe.test_evaluation import (
    TestBitCount,
    TestOpenTwos,
    TestValue,
    TestWeightedEval,
)
from tests.tictactoe.test_perfect_play import TestTheOracle, TestPerfectPlay
from tests.tictactoe.test_play import (
    TestDefaultDepth,
    TestPlayLoop as TestTicTacToePlayLoop,
    TestParseMove as TestTicTacToeParseMove,
)
from tests.tictactoe.test_conformance import TestTicTacToeConformance
from tests.tictactoe.test_encoding import (
    TestShape,
    TestPlanes,
    TestActions,
    TestSymmetries,
)

# The learned player. Only tests/zero/test_net.py needs PyTorch, and it skips without it - the
# tree and the training targets, which are the parts most likely to be wrong, are checked with
# nothing installed at all.
from tests.test_oracle import (
    TestSolver,
    TestEnumeration,
    TestBenchmarkCalibration,
    TestGrade,
)
from tests.zero.test_mcts import (
    TestTerminalValue as TestZeroTerminalValue,
    TestSearchWithPerfectKnowledge,
    TestSearchWithNoKnowledge,
    TestTheTreeIsPaths,
    TestPolicyOutput,
)
from tests.zero.test_selfplay import (
    TestValueTargets,
    TestPlayGame,
    TestRandomOpenings,
    TestAugmentation,
)
from tests.zero.test_net import (
    TestNetwork,
    TestMasking,
    TestCheckpoints,
    TestPlayer,
    TestTrainingLearns,
)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
