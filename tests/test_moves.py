import unittest

from game.board import *


class TestMoves(unittest.TestCase):
    def test_precalc_moves(self):
        match = {
            'h4h5', 'e4e7', 'e4e6', 'e4e5', 'e4g4', 'e4f4', 'e4d4', 'e4c4', 'e4e3', 'b4e7', 'b4d6', 'b4c5', 'b4a5',
            'b4c3', 'b4a3', 'g3c7', 'g3d6', 'g3e5', 'g3f4', 'g3h2', 'g3f2', 'f3f4', 'd3d8', 'd3d7', 'd3d6', 'd3d5',
            'd3d4', 'd3c4', 'd3e3', 'd3c3', 'b3c4', 'e2e3', 'c2c3', 'a2a4', 'a2a3', 'g1h3', 'e1f2', 'b1a3', 'b1c3',
            'e1f1', 'e1d1',
        }
        bb = Board('rnbqkbnr/ppp1pppp/8/8/1Bp1R2P/1P1Q1PB1/P1PPP1P1/RN2K1N1 w Qkq - 0 1')
        white_moves = {m.uci for m in bb._pseudo_legal_moves(WHITE)}
        self.assertEqual(white_moves, match)

    def test_check(self):
        for fen, match in (
            ('rnbqkb2/ppppp1p1/5p2/2n1r2p/3KP3/3P4/PPP2PPP/RNBQ1BNR w', False),
            ('rnb1kbnr/pppp1ppp/4p3/8/7q/5P2/PPPPP1PP/RNBQKBNR w', True),
            ('rnb1kbnr/pppp1ppp/4p3/7q/8/BP3P2/P1PPP1PP/RN1QKBNR b', False),
            ('rnb2bnr/ppppkppp/4p3/7q/8/BP3P2/P1PPP1PP/RN1QKBNR b', True),
        ):
            _board = Board(fen=fen)
            self.assertEqual(_board.is_in_check, match)

    def test_safe_moves(self):
        for fen, match in (
            (STARTING_STATE, {
                'b1a3', 'h2h3', 'h2h4', 'f2f3', 'g2g3', 'e2e4', 'g2g4', 'f2f4', 'e2e3', 'd2d4', 'd2d3', 'c2c4', 'c2c3',
                'b2b4', 'b2b3', 'a2a4', 'a2a3', 'g1h3', 'g1f3', 'b1c3'
            }),
            ('rnb1kbnr/pppp1ppp/4p3/2K1B2q/8/BP3P2/P1PPP1PP/RN1Q2NR w - - 0 1', {'c5b5', 'c5d4', 'c5c4'}),
            ('rnb2bnr/ppppNppp/2B1p3/1k5q/3K4/BPQ2P2/P1PPP1PP/R5NR b - - 0 1', {
                'b8c6', 'd7c6', 'b7c6', 'b5b6', 'b5a6',
            }),
            ('rnb1k3/pppp1p2/4p1p1/2P1qPN1/1P2Q1P1/4P3/P5B1/R1B1K2r w Qq - 2 25', {
                'e1f2', 'e1e2', 'e1d2', 'g2h1', 'g2f1',
            }),
            ('1rb2k2/pp1p1p1p/n7/2p5/P1Pp4/1P6/8/RN1KQ2q w - - 1 28', {
                'a4a5', 'b3b4', 'e1h1', 'e1g1', 'e1f1', 'd1e2', 'd1d2', 'd1c2', 'd1c1', 'b1c3', 'b1a3', 'b1d2', 'a1a3',
                'a1a2',
            }),
            ('3rr3/1ppb1pp1/pbnnpk1p/8/1P1P1P2/B1P2NPP/2N5/1q1QKBR1 w - - 0 38', {
                'f3g5', 'f3e5', 'f3h4', 'f3h2', 'f3d2', 'a3b2', 'a3c1', 'c2e3', 'c2a1', 'g1g2', 'g1h1', 'f1a6', 'f1b5',
                'f1c4', 'f1d3', 'f1g2', 'f1e2', 'e1f2', 'e1e2', 'e1d2', 'd1c1', 'd1b1', 'f4f5', 'd4d5', 'b4b5', 'h3h4',
                'g3g4', 'c3c4',
            }),
        ):
            _board = Board(fen=fen)
            _moves = {m.uci for m in _board.legal_moves}
            self.assertEqual(_moves, match)

    def test_make_basic_moves(self):
        bb = Board()
        bb.make_move(Move.from_uci('c2c3'))
        with self.assertRaises(IllegalMove):
            bb.make_move(Move.from_uci('b2b3'))
        bb.make_move(Move.from_uci('G8F6'))
        self.assertEqual(bb.fen, 'rnbqkb1r/pppppppp/5n2/8/8/2P5/PP1PPPPP/RNBQKBNR w KQkq - 1 2')

    def test_legal_castling(self):
        for fen, match in (
            ('rnbqkbnr/pppppppp/8/8/8/3BPN2/PPPP1PPP/RNBQK2R w KQkq - 0 1', {'e1f1', 'e1g1', 'e1e2'}),
            (
                'rnbqkbnr/pppppppp/8/8/N1B5/BP1PPN2/P1PQ1PPP/R3K2R w KQkq - 0 1',
                {'e1f1', 'e1e2', 'e1g1', 'e1c1', 'e1d1'}
            ),
            ('rnb1kbnr/pppppppp/8/1P6/N1B5/BPqPPN2/P2Q1PPP/R3K2R w KQkq - 0 1', {'e1g1', 'e1d1', 'e1e2', 'e1f1'}),
            ('rnb1kbn1/pppppppp/4q3/1P6/N1B5/BP1PPN2/P2QrPPP/R3K2R w KQq - 0 1', {'e1f1', 'e1d1', 'e1e2'}),
        ):
            _board = Board(fen=fen)
            _moves = {m.uci for m in _board.legal_moves if m.from_square == E1}
            self.assertEqual(_moves, match)

    def test_castling_moves(self):
        bb = Board()
        for move in (
            'g1f3',
            'b8c6',
            'h2h4',
            'a7a5',
            'g2g3',
            'b7b6',
            'f1g2',
            'c8b7',
            'd2d3',
            'd7d5',
            'a2a3',
            'd8d6',
        ):
            m = Move.from_uci(move)
            self.assertTrue(m in bb.legal_moves)
            bb.make_move(m)

        self.assertEqual(bb.fen, 'r3kbnr/1bp1pppp/1pnq4/p2p4/7P/P2P1NP1/1PP1PPB1/RNBQK2R w KQkq - 1 7')
        self.assertEqual(bb.castle_flags, 'KQkq')

        for move in bb.legal_moves:
            if move.is_castling:
                self.assertEqual(move.uci, 'e1g1')
                bb.make_move(move)
        self.assertEqual(bb.fen, 'r3kbnr/1bp1pppp/1pnq4/p2p4/7P/P2P1NP1/1PP1PPB1/RNBQ1RK1 b kq - 2 7')
        self.assertEqual(bb.castle_flags, 'kq')

        for move in bb.legal_moves:
            if move.is_castling:
                self.assertEqual(move.uci, 'e8c8')
                bb.make_move(move)
        self.assertEqual(bb.fen, '2kr1bnr/1bp1pppp/1pnq4/p2p4/7P/P2P1NP1/1PP1PPB1/RNBQ1RK1 w - - 3 8')
        self.assertEqual(bb.castle_flags, '-')

    def test_en_passant(self):
        bb = Board()
        for m in (
                'a2a3',
                'g7g5',
                'a3a4',
                'g5g4',
                'f2f4',
                'g4f3',
        ):
            move = Move.from_uci(m)
            self.assertTrue(move in bb.legal_moves)
            bb.make_move(move)
        self.assertEqual(bb.fen, 'rnbqkbnr/pppppp1p/8/8/P7/5p2/1PPPP1PP/RNBQKBNR w KQkq - 0 4')

    def test_promotion(self):
        b = Board('rnbqr3/pppp2P1/3k1n1p/2p1p3/3b4/8/PPPPPP1P/RNBQKBNR w KQ - 0 1')
        promotion_moves = set()
        for m in b.legal_moves:
            if m.from_square == G7:
                promotion_moves.add(m.uci)
        self.assertEqual(promotion_moves, {f'g7g8{p}' for p in ['q', 'r', 'b', 'n']})

        b.make_move(Move.from_uci('g7g8r'))
        self.assertEqual(b.fen, 'rnbqr1R1/pppp4/3k1n1p/2p1p3/3b4/8/PPPPPP1P/RNBQKBNR b KQ - 0 1')
        b.unmake_move()
        b.make_move(Move.from_uci('g7g8q'))
        self.assertEqual(b.fen, 'rnbqr1Q1/pppp4/3k1n1p/2p1p3/3b4/8/PPPPPP1P/RNBQKBNR b KQ - 0 1')


    def test_checkmate(self):
        for fen, result in (
            ('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w', False),
            ('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b', False),
            ('rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w', True),
            ('rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR b', False),
            ('3q1bRk/5p2/5N1p/8/8/8/2r2PPP/6K1 w', False),
            ('3q1bRk/5p2/5N1p/8/8/8/2r2PPP/6K1 b', True),
            ('R7/3pkppr/5P1p/2p5/8/4P3/3P2PP/1NBQKBNR b K - 0 19', False),
        ):
            _board = Board(fen=fen)
            self.assertEqual(_board.is_checkmate, result)

    def test_stalemate(self):
        for fen, result in (
            ('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w', False),
            ('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b', False),
            ('rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w', False),
            ('rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR b', False),
            ('5k2/5P2/5K2/8/8/8/8/8 w - - 0 1 w', False),
            ('5k2/5P2/5K2/8/8/8/8/8 b - - 0 1 b', True),
        ):
            _board = Board(fen=fen)
            self.assertEqual(_board.is_stalemate, result)

    def test_insufficient_material(self):
        for fen, result in (
            ('5k2/5P2/5K2/8/8/8/8/8 b - - 0 1', False),
            ('rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w', False),
            ('8/8/3K4/8/1k6/8/8/8 w - - 0 1', True),
            ('8/8/3K4/8/1k6/8/3b4/8 w - - 0 1', True),
            ('8/8/3n4/8/1k6/8/3K4/8 b - - 0 1', True),
            ('8/8/3bb3/8/1k6/8/3K4/8 b - - 0 1', False),
            ('8/8/3b4/8/1k6/4B3/3K4/8 b - - 0 1', True),
        ):
            _board = Board(fen)
            self.assertEqual(_board.has_insufficient_material, result)

    def test_threefold_repetition(self):
        _board = Board(track_repetitions=True)
        for move in (
            'B2B3',
            'C7C6',
            'B3B4',
            'C6C5',
            'B4C5',
            'B8C6',
            'C2C4',
            'A8B8',
            'D1B3',
            'B8A8',
            'B3D3',
            'A8B8',
            'D3B3',
            'B8A8',
            'B3D3',
            'A8B8',
            'D3B3',
        ):
            m = Move.from_uci(move)
            self.assertTrue(m in _board.legal_moves)
            _board.make_move(m)

        self.assertTrue(_board.has_threefold_repetition)
        _board.unmake_move()
        _board.make_move(Move.from_uci('h2h3'))
        self.assertFalse(_board.has_threefold_repetition)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
