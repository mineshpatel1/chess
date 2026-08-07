import unittest

from tests.test_board import TestBitboard
from tests.test_moves import TestMoves
from tests.test_undo import TestUndo
from tests.test_permutations import TestPermutations
from tests.test_search import TestSearch, TestTerminalValue, TestUciSearchOutput
from tests.test_search_equivalence import TestHarness, TestDecisivePositions
from tests.test_simulate import TestResult, TestSimulateGame


def main():
    unittest.main()


if __name__ == '__main__':
    main()
