from __future__ import annotations
from typing import List, Tuple, Optional, Iterable

import log
from engine.position import char_to_file, file_rank_to_index
from engine.exceptions import IllegalMove
from engine.constants import (
    Colour,
    WHITE,
    BLACK,

    FILE_NAMES,
    RANK_NAMES,
    STARTING_STATE,

    PieceType,
    PAWN,
    ROOK,
    KNIGHT,
    BISHOP,
    QUEEN,
    KING,
    PIECE_ICONS,
    PIECE_NAMES,
    PIECE_TYPES,
    PIECE_POINTS,

    Direction,
    NORTH,
    NORTHEAST,
    EAST,
    SOUTHEAST,
    SOUTH,
    SOUTHWEST,
    WEST,
    NORTHWEST,
)

Bitboard = int


def msb(x):
    """Returns most significant bit."""
    return x.bit_length() - 1


def lsb(x):
    """Returns least significant bit."""
    return msb(x & -x)


def binary_str(i: int):
    """Returns a base 2 integer as a binary string of length 64"""
    s = '{0:b}'.format(i)
    s = ('0' * (64 - len(s))) + s  # Pad with zeroes
    assert len(s) == 64
    return s


def bit_count(i):
    count = 0
    while i:
        i &= i - 1
        count += 1
    return count


def bitboard_to_squares(bb):
    while bb:
        r = bb.bit_length() - 1
        yield Square(r)
        bb ^= BB_SQUARES[r]


def bitboard_to_str(bb):
    """Prints a visual representation of the occupation represented by the input bitboard integer."""
    board_str = ''
    rank = 8
    for sq in SQUARES_VFLIP:
        if rank > sq.rank:
            board_str += f'\n{sq.rank + 1} '
            rank = sq.rank

        if binary_str(bb)[63 - sq] == '1':
            board_str += '[•]'
        else:
            board_str += '[ ]'
    board_str += '\n   A  B  C  D  E  F  G  H '
    return board_str


class Square(int):
    @staticmethod
    def from_file_rank(file: int, rank: int):
        return Square(file_rank_to_index(file, rank))

    @staticmethod
    def from_coord(coord: str):
        file = char_to_file(coord[0].upper())
        rank = int(coord[1]) - 1
        return Square(file_rank_to_index(file, rank))

    @property
    def file(self):
        return self % 8

    @property
    def rank(self):
        return int(self / 8)

    @property
    def colour(self):
        return WHITE if self % 2 == 0 else BLACK

    @property
    def name(self):
        return FILE_NAMES[self.file] + RANK_NAMES[self.rank]

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)


def square_distance(a: Square, b: Square) -> int:
    """
    Gets the distance (i.e., the number of king steps) from square *a* to *b*.
    """
    a, b = Square(a), Square(b)
    return max(abs(a.file - b.file), abs(a.rank - b.rank))


def mirror_square(square: Square, vertical: bool = True) -> Square:
    """Returns a square position as if the board was mirrored vertically."""
    if vertical:
        return Square(square ^ 56)
    else:
        return Square(square ^ 7)


SQUARES = [
    A1, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8,
] = [Square(i) for i in range(64)]

SQUARES_VFLIP = [mirror_square(sq, True) for sq in SQUARES]

BB_DIRECTIONS = {
    NORTH: 8,
    NORTHEAST: 9,
    EAST: 1,
    SOUTHEAST: -7,
    SOUTH: -8,
    SOUTHWEST: -9,
    WEST: -1,
    NORTHWEST: 7,
}

# 64-bit representations of board positions
BB_EMPTY = 0
BB_BOARD = (2 ** 64) - 1
BB_WHITE_SQUARES = 6172840429334713770
BB_BLACK_SQUARES = 12273903644374837845

BB_SQUARES = [
    BB_A1, BB_B1, BB_C1, BB_D1, BB_E1, BB_F1, BB_G1, BB_H1,
    BB_A2, BB_B2, BB_C2, BB_D2, BB_E2, BB_F2, BB_G2, BB_H2,
    BB_A3, BB_B3, BB_C3, BB_D3, BB_E3, BB_F3, BB_G3, BB_H3,
    BB_A4, BB_B4, BB_C4, BB_D4, BB_E4, BB_F4, BB_G4, BB_H4,
    BB_A5, BB_B5, BB_C5, BB_D5, BB_E5, BB_F5, BB_G5, BB_H5,
    BB_A6, BB_B6, BB_C6, BB_D6, BB_E6, BB_F6, BB_G6, BB_H6,
    BB_A7, BB_B7, BB_C7, BB_D7, BB_E7, BB_F7, BB_G7, BB_H7,
    BB_A8, BB_B8, BB_C8, BB_D8, BB_E8, BB_F8, BB_G8, BB_H8,
] = [1 << sq for sq in SQUARES]

BB_FILES = [
    BB_FILE_A,
    BB_FILE_B,
    BB_FILE_C,
    BB_FILE_D,
    BB_FILE_E,
    BB_FILE_F,
    BB_FILE_G,
    BB_FILE_H,
] = [
    # Create a set of the first file (A) and then iteratively shift all positions eastward 1 square
    (BB_A1 | BB_A2 | BB_A3 | BB_A4 | BB_A5 | BB_A6 | BB_A7 | BB_A8) << (i * BB_DIRECTIONS['e'])
    for i in range(8)
]

BB_RANKS = [
    BB_RANK_1,
    BB_RANK_2,
    BB_RANK_3,
    BB_RANK_4,
    BB_RANK_5,
    BB_RANK_6,
    BB_RANK_7,
    BB_RANK_8,
] = [
    # Create a set of the first rank (1) and then iteratively shift all positions northward 1 square
    (BB_A1 | BB_B1 | BB_C1 | BB_D1 | BB_E1 | BB_F1 | BB_G1 | BB_H1) << (i * BB_DIRECTIONS['n'])
    for i in range(8)
]


def _gen_moves(intervals: Iterable[Tuple[int, int]]) -> List[Bitboard]:
    bbs = []
    for sq in SQUARES:
        moves = BB_EMPTY
        for i, j in intervals:
            _file = sq.file + i
            _rank = sq.rank + j
            if 0 <= _file < 8 and 0 <= _rank < 8:  # Checks within the board
                _sq = file_rank_to_index(sq.file + i, sq.rank + j)
                moves |= BB_SQUARES[_sq]
        bbs.append(moves)
    return bbs


def _gen_rays(file_adjust, rank_adjust) -> List[Bitboard]:
    def _in_board(_file, _rank):
        return 0 <= _file < 8 and 0 <= _rank < 8

    bbs = []
    for sq in SQUARES:
        moves = BB_EMPTY
        _file = sq.file
        _rank = sq.rank
        while _in_board(_file, _rank):  # Still in board
            _file = _file + file_adjust
            _rank = _rank + rank_adjust
            if _in_board(_file, _rank):
                _sq = file_rank_to_index(_file, _rank)
                moves |= BB_SQUARES[_sq]
        bbs.append(moves)
    return bbs


BB_ORIGINAL_ROOKS = {
    WHITE: (BB_A1 | BB_H1),
    BLACK: (BB_A8 | BB_H8),
}
BB_KNIGHT_MOVES = _gen_moves(((1, 2), (1, -2), (-1, 2), (-1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1)))
BB_KING_MOVES = _gen_moves(((1, 1), (0, 1), (1, 0), (-1, -1), (-1, 0), (0, -1), (1, -1), (-1, 1)))
BB_PAWN_ATTACKS = {
    WHITE: _gen_moves(((1, 1), (-1, 1))),
    BLACK: _gen_moves(((1, -1), (-1, -1))),
}
BB_PAWN_MOVES = {
    WHITE: _gen_moves(((0, 1),)),  # Single advance
    BLACK: _gen_moves(((0, -1),)),
}

for _rank in (1, 6):
    for _file in range(8):
        i = file_rank_to_index(_file, _rank)
        if _rank == 1:  # White pawns
            single_move = BB_PAWN_MOVES[WHITE][i]
            BB_PAWN_MOVES[WHITE][i] = single_move << 8 | single_move
        else:
            single_move = BB_PAWN_MOVES[BLACK][i]
            BB_PAWN_MOVES[BLACK][i] = single_move >> 8 | single_move

BB_RAYS = {
    NORTH: _gen_rays(0, 1),
    NORTHEAST: _gen_rays(1, 1),
    EAST: _gen_rays(1, 0),
    SOUTHEAST: _gen_rays(1, -1),
    SOUTH: _gen_rays(0, -1),
    SOUTHWEST: _gen_rays(-1, -1),
    WEST: _gen_rays(-1, 0),
    NORTHWEST: _gen_rays(-1, 1),
}

BB_CARDINALS = [
    BB_RAYS[NORTH][i] | BB_RAYS[EAST][i] | BB_RAYS[SOUTH][i] | BB_RAYS[WEST][i]
    for i in range(64)
]

BB_DIAGONALS = [
    BB_RAYS[NORTHEAST][i] | BB_RAYS[SOUTHEAST][i] | BB_RAYS[SOUTHWEST][i] | BB_RAYS[NORTHWEST][i]
    for i in range(64)
]

BB_BETWEEN = []  # type: List[List[int]]


def _calc_between(rays, _from_sq, _to_sq):
    possible = rays[_from_sq]
    blockers = possible & BB_SQUARES[_to_sq]
    blocked_paths = rays[lsb(blockers)] | rays[msb(blockers)]
    return (possible & ~blocked_paths) ^ BB_SQUARES[_to_sq]


for _from_sq in SQUARES:
    BB_BETWEEN.append([])
    for _to_sq in SQUARES:
        bb_to_sq = BB_SQUARES[_to_sq]

        between = None
        for _direction in BB_RAYS:
            if BB_RAYS[_direction][_from_sq] & bb_to_sq:
                between = _calc_between(BB_RAYS[_direction], _from_sq, _to_sq)
                continue

        if between:
            BB_BETWEEN[_from_sq].append(between)
        else:
            BB_BETWEEN[_from_sq].append(BB_EMPTY)


class Piece:
    TYPE = None
    BASE_VALUE = 0
    CASTLE_POSITIONS = {
        WHITE: {},
        BLACK: {},
    }

    def __init__(self, piece_type: PieceType, colour: bool = WHITE):
        assert colour in [WHITE, BLACK], f"Invalid colour: {colour} chosen."
        assert piece_type in PIECE_TYPES, f"Invalid piece type: {piece_type} chosen."
        self.colour = colour
        self.type = piece_type

    @property
    def name(self) -> str:
        colour_name = 'White' if self.colour else 'Black'
        return f'{colour_name} {PIECE_NAMES[self.type]}'

    @property
    def code(self) -> str:
        return self.type.upper() if self.colour == WHITE else self.type.lower()

    @property
    def icon(self) -> str:
        return PIECE_ICONS[self.code]

    @property
    def value(self) -> int:
        modifier = 1 if self.colour == WHITE else -1
        return modifier * PIECE_POINTS[self.type]

    def __str__(self) -> str:
        return f'{self.icon}'

    def __repr__(self) -> str:
        return str(self)


class Move:
    @staticmethod
    def from_uci(uci: str) -> Move:
        assert len(uci) == 4, "Invalid UCI"
        pos_1 = uci[:2]
        from_sq = file_rank_to_index(char_to_file(pos_1[0]), int(pos_1[1]) - 1)
        pos_2 = uci[2:4]
        to_sq = file_rank_to_index(char_to_file(pos_2[0]), int(pos_2[1]) - 1)
        return Move(Square(from_sq), Square(to_sq))

    def __init__(
            self,
            from_square: Square,
            to_square: Square,
            is_castling: bool = False,
    ):
        self.from_square = Square(from_square)
        self.to_square = Square(to_square)
        self.is_castling = is_castling

    @property
    def uci(self):
        return f'{str(self.from_square).lower()}{str(self.to_square).lower()}'

    def __str__(self):
        return f'{self.uci}'

    def __repr__(self):
        return f"'{str(self)}'"

    def __eq__(self, other):
        return (
            other.from_square == self.from_square and
            other.to_square == self.to_square
        )


class Board:
    def __init__(self, fen: str = STARTING_STATE):
        self.turn = WHITE
        self.en_passant_sq = None
        self.halfmove_clock = 0
        self.fullmoves = 0

        self._history = []
        self._clear()
        self._set_from_fen(fen)

    @property
    def _bb_en_passant(self):
        return BB_SQUARES[self.en_passant_sq] if self.en_passant_sq else BB_EMPTY

    def _save(self):
        self._history.append(_BoardState(self))

    def _clear(self):
        """Defines an empty bitbaord."""
        self.pawns = {
            WHITE: BB_EMPTY,
            BLACK: BB_EMPTY,
        }
        self.knights = {
            WHITE: BB_EMPTY,
            BLACK: BB_EMPTY,
        }
        self.bishops = {
            WHITE: BB_EMPTY,
            BLACK: BB_EMPTY,
        }
        self.rooks = {
            WHITE: BB_EMPTY,
            BLACK: BB_EMPTY,
        }
        self.queens = {
            WHITE: BB_EMPTY,
            BLACK: BB_EMPTY,
        }
        self.kings = {
            WHITE: BB_EMPTY,
            BLACK: BB_EMPTY,
        }

        self.occupied = BB_EMPTY
        self.occupied_colour = {
            WHITE: BB_EMPTY,
            BLACK: BB_EMPTY,
        }

        self.castling_rights = {  # Should call self._update_castling_rights
            WHITE: BB_ORIGINAL_ROOKS[WHITE],
            BLACK: BB_ORIGINAL_ROOKS[BLACK],
        }

        self.turn = WHITE
        self.en_passant_sq = None
        self.halfmove_clock = 0
        self.fullmoves = 0

    def _set_from_fen(self, fen: str):
        rank = 7
        file = 0
        components = fen.split(' ')
        for char in components[0]:
            if char.isdigit():
                file += int(char)
            elif char == '/':
                rank -= 1
                file = 0
            else:
                assert char.lower() in PIECE_TYPES, f'{char} is not a valid piece in FEN notation.'
                self.place_piece(
                    Square.from_file_rank(file, rank),
                    char.lower(),
                    WHITE if char.isupper() else BLACK,
                )
                file += 1

        if len(components) > 1:
            turn = components[1].lower()
            assert turn in ['w', 'b'], "Invalid FEN."
            self.turn = BLACK if turn == 'b' else WHITE

        if len(components) > 3:
            _en_passant_coord = components[3].upper()
            self.en_passant_sq = None if _en_passant_coord == '-' else Square.from_coord(_en_passant_coord)

        if len(components) > 4:
            self.halfmove_clock = int(components[4])

        if len(components) > 5:
            self.fullmoves = int(components[5])

        self._update_castling_rights()  # Cache castling rights

    def _attack_rays_from_square(
            self, square: Square, directions: Iterable[Direction], ignore: Bitboard = BB_EMPTY,
    ) -> Bitboard:
        moves = BB_EMPTY
        for direction in directions:
            possible = BB_RAYS[direction][square]
            blockers = (possible & self.occupied) ^ ignore
            blocked_paths = BB_RAYS[direction][lsb(blockers)] | BB_RAYS[direction][msb(blockers)]
            moves |= possible & ~blocked_paths
        return moves

    def _moves_from_square(
            self, square: Square, colour: Colour, attacks_only: bool = False, ignore: Bitboard = BB_EMPTY,
    ) -> Optional[Bitboard]:
        bb_sq = BB_SQUARES[square]

        if self.pawns[colour] & bb_sq:
            moves = BB_PAWN_ATTACKS[colour][square]

            # If actually moving the piece, need to restrict pawn diagonal moves to captures
            if not attacks_only:
                moves &= (self.occupied_colour[not colour] | self._bb_en_passant)
                moves |= BB_PAWN_MOVES[colour][square]
            return moves
        elif self.rooks[colour] & bb_sq:
            return self._attack_rays_from_square(square, (NORTH, EAST, WEST, SOUTH), ignore=ignore)
        elif self.knights[colour] & bb_sq:
            return BB_KNIGHT_MOVES[square]
        elif self.bishops[colour] & bb_sq:
            return self._attack_rays_from_square(
                square,
                (NORTHWEST, NORTHEAST, SOUTHWEST, SOUTHEAST),
                ignore=ignore,
            )
        elif self.queens[colour] & bb_sq:
            return self._attack_rays_from_square(
                square,
                (NORTH, EAST, WEST, SOUTH, NORTHWEST, NORTHEAST, SOUTHWEST, SOUTHEAST),
                ignore=ignore,
            )
        elif self.kings[colour] & bb_sq:
            moves = BB_KING_MOVES[square]
            return moves

    def _attack_bitboard(self, colour: Colour, ignore: Bitboard = BB_EMPTY) -> Bitboard:
        """
        Returns a bitboard of all possible squares that a player can currently attack.

        Args:
            colour: Colour of the player attacking
            mask: Filters out any pieces included in the mask: calculates the attack board as if they weren't there.
        """
        attack_moves = BB_EMPTY
        for from_square in bitboard_to_squares(self.occupied_colour[colour]):
            moves = self._moves_from_square(from_square, colour, ignore=ignore, attacks_only=True)
            if moves:
                attack_moves |= moves
        return attack_moves & ~self.occupied_colour[colour]

    def _pseudo_legal_moves_from_square(self, from_square: Square, colour: Colour) -> Iterable[Move]:
        moves = self._moves_from_square(from_square, colour)
        if moves:
            attack_moves = moves & ~self.occupied_colour[colour]  # Cannot take our own pieces
            for to_square in bitboard_to_squares(attack_moves):
                yield Move(from_square, to_square)

    def _attackers(self, target: Square, colour) -> Bitboard:
        cardinal_movers = self.rooks[colour] | self.queens[colour]
        diagonal_movers = self.bishops[colour] | self.queens[colour]

        return (
            (BB_CARDINALS[target] & cardinal_movers) |
            (BB_DIAGONALS[target] & diagonal_movers)
        )

    def _protectors(self, target: Square, colour: Colour) -> Bitboard:
        """
        Returns positions of pieces of the given colour that are protecting the given square from queens, rooks and
        bishops.
        """
        attackers = self._attackers(target, not colour)
        protectors = BB_EMPTY
        for attacker_sq in bitboard_to_squares(attackers):
            _blocker = BB_BETWEEN[attacker_sq][target] & self.occupied
            if _blocker and BB_SQUARES[msb(_blocker)] == _blocker:  # Check there's exactly one blocker
                protectors |= _blocker
        return protectors

    def _pseudo_legal_moves(self, colour: Colour) -> Iterable[Move]:
        # Generic moves
        for from_square in bitboard_to_squares(self.occupied_colour[colour]):
            moves = self._moves_from_square(from_square, colour)
            if moves:
                moves &= ~self.occupied_colour[colour]  # Cannot take our own pieces
                for to_square in bitboard_to_squares(moves):
                    yield Move(from_square, to_square)

        # Castling moves
        if self.castling_rights:
            from_square = msb(self.kings[colour])  # Is a move for the King
            for rook_sq in bitboard_to_squares(self.castling_rights[colour]):
                if not (BB_BETWEEN[from_square][rook_sq] & self.occupied):  # Check no pieces in-between
                    castle_sq = rook_sq + 2 if rook_sq.file == 0 else rook_sq - 1
                    yield Move(from_square, castle_sq, is_castling=True)

    def _update_castling_rights(self):
        # Can never get castling rights back, so if we've removed them all, return quickly
        if not (self.castling_rights[WHITE] | self.castling_rights[BLACK]):
            return self.castling_rights

        white_castling = BB_ORIGINAL_ROOKS[WHITE] & self.rooks[WHITE] & self.castling_rights[WHITE]
        black_castling = BB_ORIGINAL_ROOKS[BLACK] & self.rooks[BLACK] & self.castling_rights[BLACK]

        # Kings can't have moved
        if not self.kings[WHITE] & BB_E1:
            white_castling = BB_EMPTY
        if not self.kings[BLACK] & BB_E8:
            black_castling = BB_EMPTY

        self.castling_rights = {
            WHITE: white_castling,
            BLACK: black_castling,
        }
        return self.castling_rights

    @property
    def castle_flags(self) -> str:
        flags = ''

        def _castle_flag(_q, _colour):
            flag = 'q' if _q == 1 else 'k'
            return flag.upper() if _colour == WHITE else flag.lower()

        for colour in [WHITE, BLACK]:
            for i, file in enumerate([BB_FILE_H, BB_FILE_A]):  # Kingside, Queenside
                if self.castling_rights[colour] & file:
                    flags += _castle_flag(i, colour)

        return flags if len(flags) > 0 else '-'

    @property
    def fen(self) -> str:
        """
        Returns the board's current state in FE Notation.
        (https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation)
        """
        fen_str = ''
        rank = 7
        blank_counter = 0
        for sq in SQUARES_VFLIP:
            if rank > sq.rank:
                if blank_counter > 0:
                    fen_str += str(blank_counter)
                    blank_counter = 0
                fen_str += '/'
                rank = sq.rank

            piece = self.piece_at(sq)
            if piece:
                if blank_counter > 0:
                    fen_str += str(blank_counter)
                    blank_counter = 0
                fen_str += piece.code
            else:
                blank_counter += 1

        _turn = 'w' if self.turn == WHITE else 'b'
        _en_passant = '-' if not self.en_passant_sq else str(self.en_passant_sq).lower()
        fen_str += f' {_turn} {self.castle_flags} {_en_passant} {self.halfmove_clock} {self.fullmoves}'

        return fen_str

    @property
    def is_in_check(self):
        return bool(self.kings[self.turn] & self._attack_bitboard(not self.turn))

    @property
    def legal_moves(self) -> Iterable[Move]:
        """Yields legal moves for the turn player."""
        king = self.kings[self.turn]
        king_pos = msb(king)
        protectors = self._protectors(king_pos, self.turn)
        attacks = self._attack_bitboard(not self.turn, ignore=king)  # Pretend the King isn't there
        in_check = king & attacks
        for move in self._pseudo_legal_moves(self.turn):
            # If we are moving the king we should be careful
            if move.from_square == king_pos:
                if attacks & BB_SQUARES[move.to_square]:  # New position is under attack
                    continue

                if move.is_castling:
                    if in_check:  # Cannot castle whilst in check
                        continue
                    elif (attacks & BB_BETWEEN[move.from_square][move.to_square]) > BB_EMPTY:
                        continue  # Cannot castle if intermediate squares are under attack

            # Cannot move this piece, it's protecting the King
            if protectors & BB_SQUARES[move.from_square]:
                continue

            # If in check and we are not moving the king, we must protect it
            if in_check:
                if move.from_square != king_pos:
                    attackers = self._attackers(king_pos, not self.turn)

                    # If there is more than one attacking piece, we can't protect
                    if bit_count(attackers) > 1:
                        continue

                    attacker_sq = list(bitboard_to_squares(attackers))[0]
                    if not (  # Deem illegal unless the move is one of these two caveats:
                        BB_BETWEEN[attacker_sq][king_pos] & BB_SQUARES[move.to_square] or  # Piece blocks danger
                        move.to_square == attacker_sq  # Piece takes attacker
                    ):
                        continue

            yield move

    def make_move(self, move: Move):
        """
        Moves a piece on the board. Warning: moves are not checked for legality in this function, this is for speed.
        The consumer of this API should enforce legality by checking Bitboard.legal_moves.
        """
        self._save()
        piece = self.piece_at(move.from_square)
        captured_piece = self.piece_at(move.to_square)

        if not piece:
            raise IllegalMove(f"No piece at {move.from_square}")

        if piece.colour != self.turn:
            raise IllegalMove(f"Can't move that piece, it's not your turn.")

        # Castling if a king is moving more than 1 square
        if piece.type == KING and abs(move.from_square.file - move.to_square.file) > 1:
            # Move King
            self.remove_piece(move.from_square)
            self.place_piece(move.to_square, piece.type, piece.colour)

            # Move Rook
            rook_shift = 1 if move.to_square.file < move.from_square.file else -1  # For Queen/Kingside
            if rook_shift > 0:  # Queenside
                self.remove_piece(Square.from_file_rank(0, move.to_square.rank))
            else:
                self.remove_piece(Square.from_file_rank(7, move.to_square.rank))

            self.place_piece(
                Square.from_file_rank(move.to_square.file + rook_shift, move.from_square.rank),
                ROOK,
                piece.colour,
            )
        elif piece.type == PAWN and move.to_square == self.en_passant_sq:  # Take piece by en_passant
            shift = -8 if piece.colour == WHITE else 8
            capture_sq = self.en_passant_sq + shift
            captured_piece = self.remove_piece(capture_sq)
            self.remove_piece(move.from_square)
            self.place_piece(move.to_square, piece.type, piece.colour)
        else:
            # Regular piece move
            self.remove_piece(move.from_square)
            self.place_piece(move.to_square, piece.type, piece.colour)

        # Set En Passant square
        if piece.type == PAWN:
            distance = move.to_square.rank - move.from_square.rank
            if abs(distance) == 2:
                if distance > 0:  # White pawn goes from low rank to higher
                    self.en_passant_sq = Square(move.from_square + 8)  # En Passant square is 1 rank behind
                else:  # Black pawn
                    self.en_passant_sq = Square(move.from_square - 8)
            else:
                self.en_passant_sq = None
        else:
            self.en_passant_sq = None

        # Update castling rights if the king or rook move
        if piece.type in (KING, ROOK):
            self._update_castling_rights()

        # Reset halfmove clock if a pawn moved or a piece was captured
        if piece.type == PAWN or captured_piece:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1

        if self.turn == BLACK:  # Increment full moves after Black's turn
            self.fullmoves += 1

        self.turn = not self.turn

    def undo(self):
        """Reverses the previous move."""
        state = self._history.pop()
        state.load(self)

    def place_piece(self, square: Square, piece_type: PieceType, colour: Colour):
        """Place a piece of a given colour on a square of the board."""
        self.remove_piece(square)  # Remove the existing piece if it exists

        mask = BB_SQUARES[square]

        if piece_type.lower() == PAWN:
            self.pawns[colour] |= mask
        elif piece_type.lower() == ROOK:
            self.rooks[colour] |= mask
        elif piece_type.lower() == KNIGHT:
            self.knights[colour] |= mask
        elif piece_type.lower() == BISHOP:
            self.bishops[colour] |= mask
        elif piece_type.lower() == QUEEN:
            self.queens[colour] |= mask
        elif piece_type.lower() == KING:
            self.kings[colour] |= mask

        self.occupied |= mask
        self.occupied_colour[colour] |= mask

    def remove_piece(self, square: Square) -> Optional[Piece]:
        """
        Removes a piece, if possible, from a square on the board.

        Returns:
            Returns the piece that existed at the square, if applicable.
        """
        piece = self.piece_at(square)
        if not piece:
            return None

        mask = BB_SQUARES[square]

        if piece.type == PAWN:
            self.pawns[piece.colour] ^= mask
        elif piece.type == ROOK:
            self.rooks[piece.colour] ^= mask
        elif piece.type == KNIGHT:
            self.knights[piece.colour] ^= mask
        elif piece.type == BISHOP:
            self.bishops[piece.colour] ^= mask
        elif piece.type == QUEEN:
            self.queens[piece.colour] ^= mask
        elif piece.type == KING:
            self.kings[piece.colour] ^= mask

        self.occupied ^= mask
        self.occupied_colour[piece.colour] ^= mask

        return piece

    def piece_at(self, square: Square) -> Optional[Piece]:
        """Optionally returns the piece occupying the given square."""
        mask = BB_SQUARES[square]

        if not self.occupied & mask:
            return None

        for colour in [WHITE, BLACK]:
            if self.pawns[colour] & mask:
                return Piece(PAWN, colour)
            elif self.rooks[colour] & mask:
                return Piece(ROOK, colour)
            elif self.knights[colour] & mask:
                return Piece(KNIGHT, colour)
            elif self.bishops[colour] & mask:
                return Piece(BISHOP, colour)
            elif self.queens[colour] & mask:
                return Piece(QUEEN, colour)
            elif self.kings[colour] & mask:
                return Piece(KING, colour)

    def __str__(self):
        board_str = ''
        rank = 8
        for sq in SQUARES_VFLIP:
            if rank > sq.rank:
                board_str += f'\n{sq.rank + 1} '
                rank = sq.rank

            piece = self.piece_at(sq)
            if piece:
                board_str += f'[{piece.icon}]'
            else:
                board_str += '[ ]'
        board_str += '\n   A  B  C  D  E  F  G  H '
        return board_str


class _BoardState:
    """Storage of bitboard integers representing state. Very cheap to copy, even if a bit ugly."""
    def __init__(self, board: Board):
        self.turn = board.turn
        self.en_passant_sq = board.en_passant_sq
        self.halfmove_clock = board.halfmove_clock
        self.fullmoves = board.fullmoves

        self.b_pawns = board.pawns[BLACK]
        self.w_pawns = board.pawns[WHITE]
        self.b_rooks = board.rooks[BLACK]
        self.w_rooks = board.rooks[WHITE]
        self.b_knights = board.knights[BLACK]
        self.w_knights = board.knights[WHITE]
        self.b_bishops = board.bishops[BLACK]
        self.w_bishops = board.bishops[WHITE]
        self.b_queens = board.queens[BLACK]
        self.w_queens = board.queens[WHITE]
        self.b_kings = board.kings[BLACK]
        self.w_kings = board.kings[WHITE]

        self.occupied = board.occupied
        self.occupied_colour_w = board.occupied_colour[WHITE]
        self.occupied_colour_b = board.occupied_colour[BLACK]
        self.castling_rights = board.castling_rights

    def load(self, board: Board):
        board.turn = self.turn
        board.en_passant_sq = self.en_passant_sq
        board.halfmove_clock = self.halfmove_clock
        board.fullmoves = self.fullmoves

        board.pawns[BLACK] = self.b_pawns
        board.pawns[WHITE] = self.w_pawns
        board.rooks[BLACK] = self.b_rooks
        board.rooks[WHITE] = self.w_rooks
        board.knights[BLACK] = self.b_knights
        board.knights[WHITE] = self.w_knights
        board.bishops[BLACK] = self.b_bishops
        board.bishops[WHITE] = self.w_bishops
        board.queens[BLACK] = self.b_queens
        board.queens[WHITE] = self.w_queens
        board.kings[BLACK] = self.b_kings
        board.kings[WHITE] = self.w_kings

        board.occupied = self.occupied
        board.occupied_colour[WHITE] = self.occupied_colour_w
        board.occupied_colour[BLACK] = self.occupied_colour_b
        board.castling_rights = self.castling_rights
