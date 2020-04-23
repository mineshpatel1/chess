from typing import Dict, Optional, Union

import log
from game.bitboard import *
from game.move import Move
from game.piece import Piece
from game.exceptions import (
    Checkmate,
    Stalemate,
    IllegalMove,
    FiftyMoveDraw,
    ThreefoldRepetition,
    InsufficientMaterial,
)
from game.constants import (
    Colour,
    WHITE,
    BLACK,

    STARTING_STATE,

    PieceType,
    PAWN,
    ROOK,
    KNIGHT,
    BISHOP,
    QUEEN,
    KING,
    PIECE_TYPES,
    PIECE_VALUES,

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
from game.square import (
    PAWN_POSITION_VALUES,
    ROOK_POSITION_VALUES,
    KNIGHT_POSITION_VALUES,
    BISHOP_POSITION_VALUES,
    QUEEN_POSITION_VALUES,
    KING_POSITION_VALUES,
    KING_LATE_GAME_POSITION_VALUES,
)


class Board:
    def __init__(self, fen: str = STARTING_STATE, track_repetitions: bool = False):
        """
        Represents the chess game and game state as a bitboard.

        Args:
            fen: Forsyth–Edwards Notation string to load the game state from
                (https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation).
            track_repetitions: If specified as True, will store game states every move to verify if the same game
                position has been reached 3 times for a threefold repetition draw
                (https://en.wikipedia.org/wiki/Threefold_repetition). Note that this will impact performance but
                is required for a strictly true game state.
        """
        self.turn = WHITE
        self.en_passant_sq = None
        self.halfmove_clock = 0
        self.fullmoves = 0
        self.track_repetitions = track_repetitions
        self.repetitions = []
        self.move_history = []

        self._history = []
        self._clear()
        self._set_from_fen(fen)

    @property
    def _bb_en_passant(self):
        return BB_SQUARES[self.en_passant_sq] if self.en_passant_sq else BB_EMPTY

    @property
    def _short_fen(self):
        return ' '.join(self.fen.split(' ')[:4])

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
        self.repetitions = []

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
            blockers = (possible & self.occupied) & ~ignore

            if blockers:
                blocked_paths = BB_RAYS[direction][lsb(blockers)] | BB_RAYS[direction][msb(blockers)]
            else:
                blocked_paths = BB_EMPTY
            moves |= possible & ~blocked_paths
        return moves

    def _moves_from_square(
            self, square: Square, colour: Colour, attacks_only: bool = False, ignore: Bitboard = BB_EMPTY,
    ) -> Optional[Bitboard]:
        def _filter_occupied(_moves):
            if attacks_only:
                return _moves
            else:
                return _moves & ~self.occupied_colour[colour]  # Cannot take our own pieces

        bb_sq = BB_SQUARES[square]

        if self.pawns[colour] & bb_sq:
            moves = BB_PAWN_ATTACKS[colour][square]

            # If actually moving the piece, need to restrict pawn diagonal moves to captures
            if not attacks_only:
                moves &= (self.occupied_colour[not colour] | self._bb_en_passant)
                advances = BB_PAWN_MOVES[colour]['single'][square] & ~self.occupied  # Single move (unless occupied)
                if advances:  # Conditionally allow a double advance
                    advances |= BB_PAWN_MOVES[colour]['double'][square] & ~self.occupied
                moves |= advances
            return moves
        elif self.rooks[colour] & bb_sq:
            moves = self._attack_rays_from_square(square, (NORTH, EAST, WEST, SOUTH), ignore=ignore)
            return _filter_occupied(moves)
        elif self.knights[colour] & bb_sq:
            return _filter_occupied(BB_KNIGHT_MOVES[square])
        elif self.bishops[colour] & bb_sq:
            moves = self._attack_rays_from_square(
                square,
                (NORTHWEST, NORTHEAST, SOUTHWEST, SOUTHEAST),
                ignore=ignore,
            )
            return _filter_occupied(moves)
        elif self.queens[colour] & bb_sq:
            moves = self._attack_rays_from_square(
                square,
                (NORTH, EAST, WEST, SOUTH, NORTHWEST, NORTHEAST, SOUTHWEST, SOUTHEAST),
                ignore=ignore,
            )
            return _filter_occupied(moves)
        elif self.kings[colour] & bb_sq:
            return _filter_occupied(BB_KING_MOVES[square])

    def _attack_bitboard(self, colour: Colour, ignore: Bitboard = BB_EMPTY) -> Bitboard:
        """
        Returns a bitboard of all possible squares that a player can currently attack.

        Args:
            colour: Colour of the player attacking
            ignore: Filters out any pieces included in the mask: calculates the attack game as if they weren't there.
        """
        attack_moves = BB_EMPTY
        for from_square in bitboard_to_squares(self.occupied_colour[colour]):
            moves = self._moves_from_square(from_square, colour, ignore=ignore, attacks_only=True)
            if moves:
                attack_moves |= moves
        return attack_moves

    @staticmethod
    def _filter_blockers(attackers: Bitboard, target: Square, mask: Bitboard) -> Bitboard:
        for attacker_sq in bitboard_to_squares(attackers):
            if BB_BETWEEN[attacker_sq][target] & mask:
                attackers &= ~BB_SQUARES[attacker_sq]
        return attackers

    def _attackers(
        self, target: Square, colour,
        filter_blockers: bool = False,
    ) -> Bitboard:
        """Returns the slide attackers of a given square."""
        cardinal_movers = self.rooks[colour] | self.queens[colour]
        diagonal_movers = self.bishops[colour] | self.queens[colour]

        attackers = (
            (BB_CARDINALS[target] & cardinal_movers) |
            (BB_DIAGONALS[target] & diagonal_movers) |
            (BB_KNIGHT_MOVES[target] & self.knights[colour]) |

            # We actually want the reverse colour for the pawn, because the target square is the point of view
            (BB_PAWN_ATTACKS[not colour][target] & self.pawns[colour])
        )

        if filter_blockers:
            attackers = self._filter_blockers(attackers, target, filter_blockers)
        return attackers

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
        """Generates possible moves without taking into account check and the safety of the King."""

        # Generic piece moves
        non_pawns = self.occupied_colour[colour] & ~self.pawns[colour]
        for from_square in bitboard_to_squares(non_pawns):
            moves = self._moves_from_square(from_square, colour)
            if moves:
                moves &= ~self.occupied_colour[colour]  # Cannot take our own pieces
                for to_square in bitboard_to_squares(moves):
                    yield Move(from_square, to_square)

        # Handle pawns specifically so we can assign promotions to moves
        pawns = self.pawns[colour]
        for from_square in bitboard_to_squares(pawns):
            moves = self._moves_from_square(from_square, colour)
            if moves:
                moves &= ~self.occupied_colour[colour]
                bb_sq = BB_SQUARES[from_square]
                for to_square in bitboard_to_squares(moves):
                    if self.pawns[colour] & bb_sq and to_square.rank in (0, 7):  # Promotion
                        for piece_type in (QUEEN, ROOK, BISHOP, KNIGHT):
                            yield Move(from_square, to_square, promotion=piece_type)
                    else:
                        yield Move(from_square, to_square)

        # Castling moves
        if self.castling_rights:
            from_square = Square(msb(self.kings[colour]))  # Is a move for the King
            for rook_sq in bitboard_to_squares(self.castling_rights[colour]):
                if not (BB_BETWEEN[from_square][rook_sq] & self.occupied):  # Check no pieces in-between
                    castle_sq = rook_sq + 2 if rook_sq.file == 0 else rook_sq - 1
                    yield Move(from_square, castle_sq, is_castling=True)

    def _update_castling_rights(self):
        """Sets the castling rights of the game based on the positions of the kings and rooks."""

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
        Returns the game's current state in FE Notation.
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

            if sq == H1 and piece is None:
                fen_str += str(blank_counter)

        _turn = 'w' if self.turn == WHITE else 'b'
        _en_passant = '-' if not self.en_passant_sq else str(self.en_passant_sq).lower()
        fen_str += f' {_turn} {self.castle_flags} {_en_passant} {self.halfmove_clock} {self.fullmoves}'

        return fen_str

    @property
    def all_pawns(self):
        return self.pawns[WHITE] | self.pawns[BLACK]

    @property
    def all_rooks(self):
        return self.rooks[WHITE] | self.rooks[BLACK]

    @property
    def all_knights(self):
        return self.knights[WHITE] | self.knights[BLACK]

    @property
    def all_bishops(self):
        return self.bishops[WHITE] | self.bishops[BLACK]

    @property
    def all_queens(self):
        return self.queens[WHITE] | self.queens[BLACK]

    @property
    def all_kings(self):
        return self.kings[WHITE] | self.kings[BLACK]

    @property
    def is_in_check(self):
        return bool(self.kings[self.turn] & self._attack_bitboard(not self.turn))

    @property
    def is_checkmate(self):
        return self.is_in_check and not any(self.legal_moves)

    @property
    def is_stalemate(self):
        return not self.is_in_check and not any(self.legal_moves)

    @property
    def has_insufficient_material(self):
        # If the player has any pawns, rooks or queens the game can be won
        if self.occupied & (self.all_pawns | self.all_rooks | self.all_queens):
            return False

        # If it is king vs king
        if self.occupied == self.all_kings:
            return True

        total_pieces = bit_count(self.occupied)
        if total_pieces > 4:  # Checkmate can be achieved if there are more than 4 pieces
            return False

        # If it is king + knight vs king
        if self.occupied & self.all_knights:
            return total_pieces <= 3  # Insufficient if it is only King + Knight vs King
        elif self.occupied & self.all_bishops:  # Player has at least one bishop
            if total_pieces < 4:
                return True

            # All remaining bishops are on the same colour squares

            only_bishops_and_kings = self.occupied == (self.all_bishops | self.all_kings)
            bishops_are_same_colour = (
                    (not self.all_bishops & BB_WHITE_SQUARES) or  # No bishops on white squares
                    (not self.all_bishops & BB_BLACK_SQUARES)  # No bishops on black squares
            )
            return bishops_are_same_colour and only_bishops_and_kings

    @property
    def has_threefold_repetition(self):
        if not self.track_repetitions:  # Never true if we are not keeping track
            return False

        count_reps = {}
        for rep in self.repetitions:
            count_reps[rep] = count_reps.get(rep, 0)
            count_reps[rep] += 1
            if count_reps[rep] >= 3:
                return True
        return False

    @property
    def is_game_over(self):
        has_legal_move = any(self.legal_moves)

        if self.halfmove_clock >= 50:  # 50 move draw
            return True
        elif self.has_insufficient_material:
            return True
        elif not has_legal_move:  # Check/stalemate
            return True
        else:
            return False

    @property
    def value(self) -> int:
        """Simple evaluation of the game, positive for white, negative for black."""
        def _count_value(_piece_type, _pieces, _modifier):
            return PIECE_VALUES[_piece_type] * bit_count(_pieces) * _modifier

        total = 0
        for colour in (WHITE, BLACK):
            for piece_type in (PAWN, ROOK, BISHOP, KNIGHT, QUEEN, KING):
                modifier = 1 if colour == WHITE else -1
                if piece_type == PAWN:
                    total += _count_value(piece_type, self.pawns[colour], modifier)
                elif piece_type == ROOK:
                    total += _count_value(piece_type, self.rooks[colour], modifier)
                elif piece_type == KNIGHT:
                    total += _count_value(piece_type, self.knights[colour], modifier)
                elif piece_type == BISHOP:
                    total += _count_value(piece_type, self.bishops[colour], modifier)
                elif piece_type == QUEEN:
                    total += _count_value(piece_type, self.queens[colour], modifier)
                elif piece_type == KING:
                    total += _count_value(piece_type, self.kings[colour], modifier)
        return total

    @property
    def weighted_value(self) -> int:
        """
        Weighted evaluation of the game, positive for white, negative for black. Adjusts piece values depending on the
        positions on the game. More expensive to calcualte than Board.value.
        """
        total = 0
        for colour in (WHITE, BLACK):
            modifier = 1 if colour == WHITE else -1
            for piece_type in (PAWN, ROOK, BISHOP, KNIGHT, QUEEN, KING):
                if piece_type == PAWN:
                    for sq in bitboard_to_squares(self.pawns[colour]):
                        total += ((PIECE_VALUES[piece_type] + PAWN_POSITION_VALUES[colour][sq]) * modifier)
                elif piece_type == ROOK:
                    for sq in bitboard_to_squares(self.rooks[colour]):
                        total += ((PIECE_VALUES[piece_type] + ROOK_POSITION_VALUES[colour][sq]) * modifier)
                elif piece_type == KNIGHT:
                    for sq in bitboard_to_squares(self.knights[colour]):
                        total += ((PIECE_VALUES[piece_type] + KNIGHT_POSITION_VALUES[colour][sq]) * modifier)
                elif piece_type == BISHOP:
                    for sq in bitboard_to_squares(self.bishops[colour]):
                        total += ((PIECE_VALUES[piece_type] + BISHOP_POSITION_VALUES[colour][sq]) * modifier)
                elif piece_type == QUEEN:
                    for sq in bitboard_to_squares(self.queens[colour]):
                        total += ((PIECE_VALUES[piece_type] + QUEEN_POSITION_VALUES[colour][sq]) * modifier)
                elif piece_type == KING:
                    if (
                        not (self.queens[WHITE] | self.queens[BLACK]) or
                        bit_count(
                            self.queens[WHITE] | self.queens[BLACK] |
                            self.rooks[WHITE] | self.rooks[BLACK] |
                            self.bishops[WHITE] | self.bishops[BLACK] |
                            self.knights[WHITE] | self.knights[BLACK]
                        ) <= 4
                    ):
                        pos_values = KING_POSITION_VALUES
                    else:
                        pos_values = KING_LATE_GAME_POSITION_VALUES
                    for sq in bitboard_to_squares(self.kings[colour]):
                        total += ((PIECE_VALUES[piece_type] + pos_values[colour][sq]) * modifier)
        return total

    @property
    def relative_value(self):
        """Board value normalised for the given player. All players should look to maximise this value."""
        modifier = 1 if self.turn == WHITE else -1
        return modifier * self.value

    @property
    def legal_moves(self) -> Iterable[Move]:
        """Yields legal moves for the turn player."""

        def _is_safe(_attackers, _king_pos, _move):
            _num_attackers = bit_count(_attackers)
            if _num_attackers > 1:
                return False
            elif bit_count(_attackers) == 1:
                _attacker_sq = list(bitboard_to_squares(_attackers))[0]
                if not (  # Deem illegal unless the move is one of these two caveats:
                    BB_BETWEEN[_attacker_sq][_king_pos] & BB_SQUARES[_move.to_square] or  # Piece blocks danger
                    _move.to_square == _attacker_sq  # Piece takes attacker
                ):
                    return False
            return True

        king = self.kings[self.turn]
        king_pos = Square(msb(king))
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
                all_attackers = self._attackers(king_pos, not self.turn)
                protected_attacker = self._attackers(
                    king_pos, not self.turn, filter_blockers=BB_SQUARES[move.from_square],
                )
                if not _is_safe(protected_attacker ^ all_attackers, king_pos, move):
                    continue

            # If in check and we are not moving the king, we must protect it
            if in_check:
                if move.from_square != king_pos:
                    attackers = self._attackers(king_pos, not self.turn, filter_blockers=self.occupied)
                    if not _is_safe(attackers, king_pos, move):
                        continue

            yield move

    @property
    def turn_name(self) -> str:
        return 'white' if self.turn else 'black'

    @property
    def pieces(self) -> Dict[PieceType, Dict[str, int]]:
        return {
            PAWN: self.pawns,
            ROOK: self.rooks,
            KNIGHT: self.knights,
            BISHOP: self.bishops,
            QUEEN: self.queens,
            KING: self.kings,
        }

    @property
    def pgn_uci(self) -> str:
        """
        Prints a pseudo Portable Game Notation with moves in UCI format.
        Can be loaded by https://www.chess.com/analysis
        """
        pgn = ''
        for i, move in enumerate(self.move_history):
            if i % 2 == 0:
                pgn += f'{int(i / 2) + 1}. '
            pgn += f'{index_to_coord(move.from_square).lower()}{index_to_coord(move.to_square).lower()} '
        return pgn

    def make_safe_move(self, move: Union[Move, str]):
        """
        Same as Board.make_move, but validates if the move is legal first (slower).
        """
        if isinstance(move, str):
            move = Move.from_uci(move)

        if move in self.legal_moves:
            self.make_move(move)
        else:
            raise IllegalMove(f"Move {move} is illegal.")

    def make_move(self, move: Union[Move, str]):
        """
        Moves a piece on the game. Warning: moves are not checked for legality in this function, this is for speed.
        The consumer of this API should enforce legality by checking Bitboard.legal_moves.
        """

        if isinstance(move, str):
            move = Move.from_uci(move)

        self._save()
        piece = self.piece_at(move.from_square)
        captured_piece = self.piece_at(move.to_square)

        if not piece:
            raise IllegalMove(f"No piece at {move.from_square}")

        if piece.colour != self.turn:
            raise IllegalMove(f"Can't move that piece, it's not your turn.")

        backrank = 7 if self.turn == WHITE else 0

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

            self.repetitions = []  # Reset repetitions when castling
        elif piece.type == PAWN and move.to_square == self.en_passant_sq:  # Take piece by en_passant
            shift = -8 if piece.colour == WHITE else 8
            capture_sq = self.en_passant_sq + shift
            captured_piece = self.remove_piece(capture_sq)
            self.remove_piece(move.from_square)
            self.place_piece(move.to_square, piece.type, piece.colour)
        elif piece.type == PAWN and move.to_square.rank == backrank:  # Promotion
            self.remove_piece(move.from_square)
            self.place_piece(move.to_square, move.promotion, piece.colour)  # Assume queen for now
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
            self.repetitions = []
        else:
            self.halfmove_clock += 1
            if self.track_repetitions:
                self.repetitions.append(self._short_fen)  # Imperfect repetition tracking

        if self.turn == BLACK:  # Increment full moves after Black's turn
            self.fullmoves += 1

        self.move_history.append(move)
        self.turn = not self.turn

    def unmake_move(self):
        """Reverses the previous move."""
        state = self._history.pop()
        state.load(self)

    def place_piece(self, square: Square, piece_type: PieceType, colour: Colour):
        """Place a piece of a given colour on a square of the game."""
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
        Removes a piece, if possible, from a square on the game.

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

    def raise_if_game_over(self):
        """Raises an exception if the game is in an end state."""
        if self.halfmove_clock >= 50:
            raise FiftyMoveDraw
        elif self.has_insufficient_material:
            raise InsufficientMaterial
        elif self.has_threefold_repetition:
            raise ThreefoldRepetition
        elif self.is_checkmate:
            raise Checkmate
        elif self.is_stalemate:
            raise Stalemate

    def __str__(self):
        """Board representation using Unicode piece characters."""
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

    # Aliases for benchmarking
    push = make_move
    pop = unmake_move


class _BoardState:
    """Storage of bitboard integers representing state. Very cheap to copy, even if a bit ugly."""
    def __init__(self, board: Board):
        self.turn = board.turn
        self.en_passant_sq = board.en_passant_sq
        self.halfmove_clock = board.halfmove_clock
        self.fullmoves = board.fullmoves
        self.repetitions = tuple(board.repetitions)
        self.move_history = tuple(board.move_history)

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
        board.repetitions = list(self.repetitions)
        board.move_history = list(self.move_history)

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
