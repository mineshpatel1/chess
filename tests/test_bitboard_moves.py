import unittest

from engine.bitboard import *


class TestMoves(unittest.TestCase):
    def test_precalc_moves(self):
        match = {
            'e1d1', 'd3c4', 'h4h5', 'f3f4', 'd3d8', 'e2e3', 'b4a3', 'd3d7', 'd3d6', 'g2g4', 'g3e5', 'd3d5', 'g3f2',
            'b4e7', 'd3c3', 'g3f4', 'g3h2', 'e4g4', 'c2c4', 'e1f1', 'd2d4', 'e4d4', 'b4d6', 'b1a3', 'b4c3', 'e1f2',
            'b1c3', 'e4e5', 'd3d4', 'a2a3', 'g3d6', 'd3e3', 'g3c7', 'c2c3', 'e4c4', 'b4c5', 'e4f4', 'a2a4', 'g1h3',
            'e4e6', 'b3c4', 'e4e7', 'b4a5', 'e4e3'
        }
        bb = Board('rnbqkbnr/ppp1pppp/8/8/1Bp1R2P/1P1Q1PB1/P1PPP1P1/RN2K1N1 w Qkq - 0 1')
        white_moves = {m.uci for m in bb._pseudo_legal_moves(WHITE)}
        self.assertEqual(white_moves, match)

    def test_check(self):
        for test in (
            ('rnbqkb2/ppppp1p1/5p2/2n1r2p/3KP3/3P4/PPP2PPP/RNBQ1BNR w', False),
            # ('rnb1kbnr/pppp1ppp/4p3/8/7q/5P2/PPPPP1PP/RNBQKBNR w', True),
            # ('rnb1kbnr/pppp1ppp/4p3/7q/8/BP3P2/P1PPP1PP/RN1QKBNR b', False),
            # ('rnb2bnr/ppppkppp/4p3/7q/8/BP3P2/P1PPP1PP/RN1QKBNR b', True),
        ):
            _board = Board(fen=test[0])
            self.assertEqual(_board.is_in_check, test[1])


def main():
    unittest.main()


if __name__ == '__main__':
    main()
