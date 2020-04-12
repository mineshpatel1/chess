import unittest

from engine import position
from engine.exceptions import FiftyMoveDraw, ThreefoldRepetition
from engine.position import Position

from engine import game
from engine.game import Game
from engine.constants import WHITE, BLACK, STARTING_STATE


class TestBoard(unittest.TestCase):
    def _verify_pos(self, index, file, rank, coord):
        f, r = position.index_to_rankfile(index)
        self.assertEqual((f, r), (file, rank))

        i = position.rankfile_to_index(f, r)
        self.assertEqual(i, index)

        p = position.from_index(index)
        self.assertEqual((p.file, p.rank, str(p)), (file, rank, coord))

    def test_positions(self):
        for index, file, rank, coord in (
            (0, 0, 0, 'A1'),   # Bottom-Left
            (7, 7, 0, 'H1'),   # Bottom-Right
            (56, 0, 7, 'A8'),  # Top-Left
            (63, 7, 7, 'H8'),  # Top-Right

            (12, 4, 1, 'E2'),  # Misc positions
            (38, 6, 4, 'G5'),
            (53, 5, 6, 'F7'),
        ):
            self._verify_pos(index, file, rank, coord)

    def test_coord_to_position(self):
        for coord, pos in (
            ('A1', Position(0, 0)),
            ('E7', Position(4, 6)),
            ('D5', Position(3, 4)),
            ('H8', Position(7, 7)),
        ):
            self.assertEqual(position.from_coord(coord), pos)

    def test_start_layout(self):
        _board = Game(state=STARTING_STATE)
        _pieces = {
            0: 'R', 1: 'N', 2: 'B', 3: 'Q', 4: 'K', 5: 'B', 6: 'N', 7: 'R', 8: 'P', 9: 'P', 10: 'P', 11: 'P', 12: 'P',
            13: 'P', 14: 'P', 15: 'P', 48: 'p', 49: 'p', 50: 'p', 51: 'p', 52: 'p', 53: 'p', 54: 'p', 55: 'p', 56: 'r',
            57: 'n', 58: 'b', 59: 'q', 60: 'k', 61: 'b', 62: 'n', 63: 'r',
        }

        for piece in sorted(_board.pieces):
            self.assertEqual(_pieces[piece.pos.index], piece.code)

    def test_check(self):
        for test in (
            ('rnbqkb2/ppppp1p1/5p2/2n1r2p/3KP3/3P4/PPP2PPP/RNBQ1BNR', WHITE, False),
            ('rnb1kbnr/pppp1ppp/4p3/8/7q/5P2/PPPPP1PP/RNBQKBNR', WHITE, True),
            ('rnb1kbnr/pppp1ppp/4p3/7q/8/BP3P2/P1PPP1PP/RN1QKBNR', BLACK, False),
            ('rnb2bnr/ppppkppp/4p3/7q/8/BP3P2/P1PPP1PP/RN1QKBNR', BLACK, True),
        ):
            _board = Game(state=test[0])
            self.assertEqual(_board.is_in_check(test[1]), test[2])

    def test_checkmate(self):
        for test in (
            ('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR', WHITE, False),
            ('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR', BLACK, False),
            ('rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR', WHITE, True),
            ('rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR', BLACK, False),
            ('3q1bRk/5p2/5N1p/8/8/8/2r2PPP/6K1', WHITE, False),
            ('3q1bRk/5p2/5N1p/8/8/8/2r2PPP/6K1', BLACK, True),
        ):
            _board = Game(state=test[0])
            self.assertEqual(_board.is_checkmate(test[1]), test[2])

    def test_stalemate(self):
        for test in (
            ('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR', WHITE, False),
            ('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR', BLACK, False),
            ('rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR', WHITE, False),
            ('rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR', BLACK, False),
            ('5k2/5P2/5K2/8/8/8/8/8 w - - 0 1', WHITE, False),
            ('5k2/5P2/5K2/8/8/8/8/8 b - - 0 1', BLACK, True),
        ):
            _board = Game(state=test[0])
            self.assertEqual(_board.is_stalemate(test[1]), test[2])

    def test_castle_flags(self):
        for test in (
            (STARTING_STATE, 'KQkq'),
            ('rnbqkbnr/pppppppp/8/8/8/7P/PPPPPPPR/RNBQKBN1', 'Qkq'),
            ('rnbqkbnr/pppppppp/8/8/8/P3P3/RPPP1PPP/1NBQKBNR', 'Kkq'),
            ('rnbqkbnr/pppppppp/8/8/8/4P3/PPPPKPPP/RNBQ1BNR', 'kq'),
            ('rnbqkbn1/pppppppr/7p/8/8/8/PPPPPPPP/RNBQKBNR', 'KQq'),
            ('1nbqkbnr/rpppppp1/p6p/8/8/8/PPPPPPPP/RNBQKBNR', 'KQk'),
            ('rnbq1bnr/ppppkpp1/7p/4p3/8/8/PPPPPPPP/RNBQKBNR', 'KQ'),
            ('rnbq1bnr/ppppkpp1/7p/4p3/8/4P3/PPPPKPPP/RNBQ1BNR', '-'),
        ):
            _board = Game(state=test[0])
            self.assertEqual(_board.castle_flags, test[1])

    def test_fen(self):
        for fen in (
            'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR',
            'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR',
            'rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR',
            'rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R',
            'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
            'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1',
            'rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1',
            'rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 0 1',
            'rnbqkbnr/pppppp1p/8/8/5Pp1/8/PPPPP1PP/RNBQKBNR b KQkq f3 0 1',
        ):
            _board = game.Game(fen)
            self.assertEqual(_board.fen, fen)

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
            _board = Game(fen)
            self.assertEqual(_board.has_insufficient_material, result)

    def test_fifty_move_draw(self):
        _board = Game('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 49 1')
        self.assertEqual(_board.halfmove_clock, 49)
        _board._move(Position(0, 1), Position(0, 2))  # Move pawn
        _board.raise_if_game_over()
        self.assertEqual(_board.halfmove_clock, 0)

        _board = Game('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 49 60')
        _board._move(Position(1, 0), Position(2, 2))  # Move other piece
        with self.assertRaises(FiftyMoveDraw):
            _board.raise_if_game_over()

        _board = Game('rn1qkbnr/ppp1pppp/3p4/8/6b1/5P2/PPPPP1PP/RNBQKBNR b KQkq - 49 60')
        _board._move(Position(6, 3), Position(5, 2))  # Take piece
        _board.raise_if_game_over()
        self.assertEqual(_board.halfmove_clock, 0)

    def test_threefold_repetition(self):
        _board = Game()
        for move in (
            ('B2', 'B3'),
            ('C7', 'C6'),
            ('B3', 'B4'),
            ('C6', 'C5'),
            ('B4', 'C5'),
            ('B8', 'C6'),
            ('C2', 'C4'),
            ('A8', 'B8'),
            ('D1', 'B3'),
            ('B8', 'A8'),
            ('B3', 'D3'),
            ('A8', 'B8'),
            ('D3', 'B3'),
            ('B8', 'A8'),
            ('B3', 'D3'),
            ('A8', 'B8'),
            ('D3', 'B3'),
        ):
            _board.player_move(position.from_coord(move[0]), position.from_coord(move[1]))
        with self.assertRaises(ThreefoldRepetition):
            _board.raise_if_game_over()

    def test_en_passant(self):
        _board = Game(state=STARTING_STATE)
        self.assertEqual(_board.en_passant, None)

        _board._move(position.from_coord('B2'), position.from_coord('B4'))
        self.assertEqual(_board.en_passant, position.from_coord('B3'))
        self.assertEqual(_board.fen, 'rnbqkbnr/pppppppp/8/8/1P6/8/P1PPPPPP/RNBQKBNR w KQkq b3 0 1')

        _board._move(position.from_coord('G8'), position.from_coord('F6'))
        self.assertEqual(_board.en_passant, None)
        self.assertEqual(_board.fen, 'rnbqkb1r/pppppppp/5n2/8/1P6/8/P1PPPPPP/RNBQKBNR w KQkq - 1 1')

    def test_value(self):
        for test in (
            (STARTING_STATE, 0),
            ('rn1qk3/p1p1p3/8/3Q4/8/8/PPPPPP1P/RNBQKBNR b - - 0 1', 27),
            ('rnbqkbnr/pppp1ppp/8/8/3q4/8/P2P1PPP/4KBNR w - - 0 1', -31),
        ):
            _board = Game(state=test[0])
            self.assertEqual(_board.board_value, test[1])


    def test_print(self):
        _board = Game(state=STARTING_STATE)
        match = ("""
8 [♜][♞][♝][♛][♚][♝][♞][♜]
7 [♟][♟][♟][♟][♟][♟][♟][♟]
6 [ ][ ][ ][ ][ ][ ][ ][ ]
5 [ ][ ][ ][ ][ ][ ][ ][ ]
4 [ ][ ][ ][ ][ ][ ][ ][ ]
3 [ ][ ][ ][ ][ ][ ][ ][ ]
2 [♙][♙][♙][♙][♙][♙][♙][♙]
1 [♖][♘][♗][♕][♔][♗][♘][♖]
   A  B  C  D  E  F  G  H """)
        self.assertEqual(str(_board), match)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
