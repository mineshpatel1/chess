import unittest

from games.chess.board import *


class TestMoves(unittest.TestCase):
    def test_precalc_moves(self):
        match = {
            'h4h5', 'e4e7', 'e4e6', 'e4e5', 'e4g4', 'e4f4', 'e4d4', 'e4c4', 'e4e3', 'b4e7', 'b4d6', 'b4c5', 'b4a5',
            'b4c3', 'b4a3', 'g3c7', 'g3d6', 'g3e5', 'g3f4', 'g3h2', 'g3f2', 'f3f4', 'd3d8', 'd3d7', 'd3d6', 'd3d5',
            'd3d4', 'd3c4', 'd3e3', 'd3c3', 'b3c4', 'e2e3', 'c2c3', 'a2a4', 'a2a3', 'g1h3', 'e1f2', 'b1a3', 'b1c3',
            'e1f1', 'e1d1',
        }
        bb = ChessBoard('rnbqkbnr/ppp1pppp/8/8/1Bp1R2P/1P1Q1PB1/P1PPP1P1/RN2K1N1 w Qkq - 0 1')
        white_moves = {m.uci for m in bb._pseudo_legal_moves(WHITE)}
        self.assertEqual(white_moves, match)

    def test_check(self):
        for fen, match in (
            ('rnbqkb2/ppppp1p1/5p2/2n1r2p/3KP3/3P4/PPP2PPP/RNBQ1BNR w', False),
            ('rnb1kbnr/pppp1ppp/4p3/8/7q/5P2/PPPPP1PP/RNBQKBNR w', True),
            ('rnb1kbnr/pppp1ppp/4p3/7q/8/BP3P2/P1PPP1PP/RN1QKBNR b', False),
            ('rnb2bnr/ppppkppp/4p3/7q/8/BP3P2/P1PPP1PP/RN1QKBNR b', True),
        ):
            _board = ChessBoard(fen=fen)
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
            _board = ChessBoard(fen=fen)
            _moves = {m.uci for m in _board.legal_moves}
            self.assertEqual(_moves, match)

    def test_pinned_piece_moves_along_the_pin_ray(self):
        """
        A pinned piece is not frozen: it may move anywhere along the ray it is pinned on, the
        pinner's own square included. Only leaving the ray is illegal.
        """
        for fen, from_square, match in (
            # Bishop e3 pinned to the King on f2 by the Bishop on c5
            ('4k3/8/8/2b5/8/4B3/5K2/8 w - - 0 1', E3, {'e3d4', 'e3c5'}),

            # Rook e8 pinned to the King on g8 by the Rook on b8
            ('1r2R1K1/8/8/8/8/8/8/4k3 w - - 0 1', E8, {'e8b8', 'e8c8', 'e8d8', 'e8f8'}),
        ):
            _board = ChessBoard(fen=fen)
            self.assertEqual({m.uci for m in _board.legal_moves if m.from_square == from_square}, match)

    def test_a_slider_stacked_behind_the_pinner_pins_nothing(self):
        """
        Only the nearest enemy slider on a ray pins the piece in front of the King. One stacked
        behind it is attacking nothing, because its colleague blocks the path wherever the pinned
        piece goes, and counting it as a second pinner reads the position as a double check and
        freezes the piece completely.

        These are the positions above with a second slider added behind the pinner. The moves are
        the same ones: the extra piece changes nothing.
        """
        for fen, from_square, match in (
            # A Queen on b6 behind the pinning Bishop on c5
            ('4k3/8/1q6/2b5/8/4B3/5K2/8 w - - 0 1', E3, {'e3d4', 'e3c5'}),

            # A Rook on a8 behind the pinning Rook on b8
            ('rr2R1K1/8/8/8/8/8/8/4k3 w - - 0 1', E8, {'e8b8', 'e8c8', 'e8d8', 'e8f8'}),
        ):
            _board = ChessBoard(fen=fen)
            self.assertEqual({m.uci for m in _board.legal_moves if m.from_square == from_square}, match)

    def test_make_basic_moves(self):
        bb = ChessBoard()
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
            _board = ChessBoard(fen=fen)
            _moves = {m.uci for m in _board.legal_moves if m.from_square == E1}
            self.assertEqual(_moves, match)

    def test_castling_moves(self):
        bb = ChessBoard()
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

    def test_fen_castling_field_round_trip(self):
        """Every castling availability field survives a load and a re-render of the FEN."""
        for flags in ('KQkq', 'KQk', 'KQ', 'Kkq', 'Kq', 'K', 'Qkq', 'Q', 'kq', 'k', 'q', '-'):
            fen = f'r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w {flags} - 0 1'
            _board = ChessBoard(fen=fen)
            self.assertEqual(_board.castle_flags, flags)
            self.assertEqual(_board.fen, fen)

    def test_fen_castling_field_is_honoured(self):
        """
        Castling rights come from the FEN, not from where the pieces happen to stand. Every
        King and Rook here is on its original square, but the FEN says the rights are gone.
        """
        _board = ChessBoard(fen='r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w - - 0 1')
        self.assertEqual(_board.castle_flags, '-')
        self.assertEqual({m.uci for m in _board.legal_moves if m.is_castling}, set())

    def test_fen_castling_field_narrowed_to_the_board(self):
        """A FEN claiming rights that the pieces cannot support is narrowed to reality."""
        _board = ChessBoard(fen='4k3/8/8/8/8/8/8/4K3 w KQkq - 0 1')  # No Rooks anywhere
        self.assertEqual(_board.castle_flags, '-')

    def test_fen_without_castling_field_infers_from_position(self):
        """Short FENs carry no castling field, so rights fall back to piece placement."""
        self.assertEqual(ChessBoard(fen='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w').castle_flags, 'KQkq')
        self.assertEqual(ChessBoard(fen='4k3/8/8/8/8/8/8/4K3 w').castle_flags, '-')

    def test_fen_does_not_resurrect_lost_castling_rights(self):
        """
        A round trip through FEN must not hand back rights the game has already forfeited.
        ChessBoard.copy() rebuilds from the FEN, and ai.search.alpha_beta copies the board once per
        root move, so the search would otherwise plan castles the real game can no longer play.
        """
        bb = ChessBoard('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1')
        for uci in ('e1e2', 'e8e7', 'e2e1', 'e7e8'):  # Both Kings step out and back
            bb.make_move(Move.from_uci(uci))

        self.assertEqual(bb.castle_flags, '-')
        self.assertEqual(ChessBoard(bb.fen).castle_flags, '-')
        self.assertEqual({m.uci for m in ChessBoard(bb.fen).legal_moves if m.is_castling}, set())

    def test_castling_rights_lost_when_rook_captured(self):
        """
        A Rook captured on its original square takes its castling right with it, even though
        the capturing piece is neither a King nor a Rook.
        """
        bb = ChessBoard('r3k2r/pppppppp/8/8/8/6n1/PPPPPPPP/R3K2R b KQkq - 0 1')
        self.assertEqual(bb.castle_flags, 'KQkq')

        bb.make_move(Move.from_uci('g3h1'))  # Black Knight takes the Rook on h1

        piece = bb.piece_at(H1)
        self.assertEqual((piece.type, piece.colour), (KNIGHT, BLACK))
        self.assertEqual(bb.castle_flags, 'Qkq')

        # White may still castle Queenside, but Kingside is gone with the Rook. Offering it
        # would let White capture the Knight and conjure a Rook onto f1.
        self.assertEqual({m.uci for m in bb.legal_moves if m.is_castling}, {'e1c1'})

        # Undoing the capture restores the Rook and the right along with it
        bb.unmake_move()
        self.assertEqual(bb.castle_flags, 'KQkq')
        self.assertEqual({m.uci for m in bb.legal_moves if m.is_castling}, {'e8g8', 'e8c8'})

    def test_en_passant(self):
        bb = ChessBoard()
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

    def test_en_passant_discovered_check(self):
        """
        En passant clears two squares on the same rank at once: the capturing pawn leaves,
        and the captured pawn is removed from beside it. If the King sits on that rank the
        capture can expose it, which makes the capture illegal.
        """
        # White Ka5 and Black Rh5 share rank 5, with the White b5 pawn and the Black c5 pawn
        # between them. Taking c5 en passant would remove both and open the rank.
        bb = ChessBoard('8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1')
        bb.make_move(Move.from_uci('b4b3'))
        bb.make_move(Move.from_uci('c7c5'))

        self.assertEqual(bb.en_passant_sq, C6)
        self.assertEqual({m.uci for m in bb.legal_moves if m.from_square == B5}, {'b5b6'})

        # The same applies to Black, whose King on h4 shares rank 4 with the White b4 Rook
        bb = ChessBoard('8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1')
        bb.make_move(Move.from_uci('g2g4'))

        self.assertEqual(bb.en_passant_sq, G3)
        self.assertEqual({m.uci for m in bb.legal_moves if m.from_square == F4}, {'f4f3'})

    def test_en_passant_answers_a_checking_pawn(self):
        """
        A pawn that double-pushes into check can be taken en passant, which answers the check by
        removing the checker. The capture lands past the checking pawn rather than on it, so it
        looks like neither a capture of the checker nor a block of the ray between it and the King.

        Here it is the only legal reply, so refusing it scores the position as checkmate.
        """
        bb = ChessBoard('8/1B6/R7/5k2/5p1P/3N3P/6P1/1K6 w - - 0 1')
        bb.make_move(Move.from_uci('g2g4'))  # Double-pushing to g4 checks the King on f5

        self.assertEqual(bb.en_passant_sq, G3)
        self.assertTrue(bb.is_in_check)
        self.assertEqual({m.uci for m in bb.legal_moves}, {'f4g3'})
        self.assertIsNone(bb.outcome)

        bb.make_move(Move.from_uci('f4g3'))
        self.assertEqual(bb.fen, '8/1B6/R7/5k2/7P/3N2pP/8/1K6 w - - 0 2')

    def test_promotion(self):
        b = ChessBoard('rnbqr3/pppp2P1/3k1n1p/2p1p3/3b4/8/PPPPPP1P/RNBQKBNR w KQ - 0 1')
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

    def test_promotion_piece_is_part_of_the_move(self):
        """
        The four promotions of a pawn are four different moves reaching four different
        positions, so they must not compare or hash as one.
        """
        self.assertNotEqual(Move.from_uci('g7g8q'), Move.from_uci('g7g8r'))
        self.assertEqual(len({Move.from_uci(f'g7g8{p}') for p in 'qrbn'}), 4)

        # Moves that are not promotions are unaffected
        self.assertEqual(Move.from_uci('e2e4'), Move.from_uci('e2e4'))
        self.assertEqual(len({Move.from_uci('e2e4'), Move.from_uci('e2e4')}), 1)

        # And each promotion really does produce a different board
        fen = 'rnbqr3/pppp2P1/3k1n1p/2p1p3/3b4/8/PPPPPP1P/RNBQKBNR w KQ - 0 1'
        placements = set()
        for piece in 'qrbn':
            _board = ChessBoard(fen)
            _board.make_safe_move(f'g7g8{piece}')
            placements.add(_board.fen)
        self.assertEqual(len(placements), 4)

    def test_promotion_without_a_piece_is_rejected(self):
        """
        A promotion with no piece named used to pass the legality check, because equality
        ignored the promotion, and then died inside make_move on `None.lower()`.
        """
        fen = 'rnbqr3/pppp2P1/3k1n1p/2p1p3/3b4/8/PPPPPP1P/RNBQKBNR w KQ - 0 1'

        _board = ChessBoard(fen)
        self.assertNotIn(Move.from_uci('g7g8'), list(_board.legal_moves))
        with self.assertRaises(IllegalMove):
            _board.make_safe_move('g7g8')

        # make_move is the unchecked path, and must still fail meaningfully rather than
        # with an AttributeError
        _board = ChessBoard(fen)
        with self.assertRaises(IllegalMove):
            _board.make_move('g7g8')

        # A rejected move must leave nothing behind for unmake_move to pop
        self.assertEqual(len(_board._history), 0)
        self.assertEqual(_board.fen, fen)

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
            _board = ChessBoard(fen=fen)
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
            _board = ChessBoard(fen=fen)
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
            _board = ChessBoard(fen)
            self.assertEqual(_board.has_insufficient_material, result)

    def test_fifty_move_draw_counts_plies(self):
        """
        The fifty move rule is fifty moves by *each* player. The halfmove clock counts plies,
        so the draw is only claimable once it reaches 100, not 50.
        """
        _board = ChessBoard('8/8/4k3/8/8/4K3/8/R7 w - - 49 80')
        _board.make_move(Move.from_uci('a1a2'))
        self.assertEqual(_board.halfmove_clock, 50)  # Only 25 moves each
        self.assertFalse(_board.is_game_over)
        _board.raise_if_game_over()  # Must not raise

        _board = ChessBoard('8/8/4k3/8/8/4K3/8/R7 w - - 98 80')
        _board.make_move(Move.from_uci('a1a2'))
        self.assertEqual(_board.halfmove_clock, 99)
        self.assertFalse(_board.is_game_over)

        _board.make_move(Move.from_uci('e6e7'))
        self.assertEqual(_board.halfmove_clock, 100)
        self.assertTrue(_board.is_game_over)
        with self.assertRaises(FiftyMoveDraw):
            _board.raise_if_game_over()

    def test_threefold_repetition(self):
        _board = ChessBoard(track_repetitions=True)
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

        # is_game_over must agree with raise_if_game_over about what ends a game
        self.assertTrue(_board.is_game_over)
        with self.assertRaises(ThreefoldRepetition):
            _board.raise_if_game_over()

        _board.unmake_move()
        _board.make_move(Move.from_uci('h2h3'))
        self.assertFalse(_board.has_threefold_repetition)
        self.assertFalse(_board.is_game_over)

    def test_a_copy_keeps_the_repetition_history(self):
        """
        A regression. `copy` rebuilt the board from its FEN, and a FEN carries castling rights,
        the en passant square and both clocks - but not how many times a position has occurred.
        So a copy of a game drawn by repetition had the *same signature* as the original and was
        still running, which is the one thing `GameState.signature` promises cannot happen.

        It mattered because copies are not incidental: `alpha_beta` takes one per root move and
        `ai.match` plays every game on one. Nothing enabled `track_repetitions` outside this
        file, so it never bit - but "no caller currently passes True" is not the same as correct,
        and the shared suite could not see it because a copy compared equal on every field it
        knew to look at.
        """
        _board = ChessBoard(track_repetitions=True)
        for move in ('B1C3', 'B8C6', 'C3B1', 'C6B8') * 3:
            _board.make_move(Move.from_uci(move))

        self.assertTrue(_board.has_threefold_repetition, 'the fixture should be a drawn game')

        clone = _board.copy()
        self.assertEqual(_board.signature, clone.signature, 'the same position by any measure')
        self.assertTrue(clone.track_repetitions, 'the copy stopped tracking')
        self.assertTrue(clone.has_threefold_repetition)
        self.assertEqual(_board.result, clone.result)
        self.assertEqual(_board.is_game_over, clone.is_game_over)

    def test_a_copy_leaves_the_move_history_behind(self):
        """
        The other half of the rule, so the fix above is not read as "copy everything". A copy may
        drop anything a caller cannot observe, and `move_history` is only read by `pgn_uci` - it
        is a record of where the board has been, where the repetition history is part of where
        the board may still go.
        """
        _board = ChessBoard()
        _board.make_move(Move.from_uci('e2e4'))

        clone = _board.copy()
        self.assertEqual([], clone.move_history)
        self.assertEqual(_board.signature, clone.signature)
        self.assertEqual(_board.result, clone.result)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
