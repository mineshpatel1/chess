import unittest

from connect_4.game import *
from connect_4.ai import minimax, alpha_beta


class TestMinimax(unittest.TestCase):
    def test_minimax(self):
        for mhn, move in (
            ('333303030', 0),
            ('30303330', 0),
        ):
            state = Connect4(mhn)
            prediction = minimax(state, 4)
            self.assertEqual(prediction % 7, move)
            prediction = alpha_beta(state, 4)
            self.assertEqual(prediction % 7, move)
