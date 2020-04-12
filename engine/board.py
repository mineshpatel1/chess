from typing import Iterable, Optional, Dict, List, Tuple, Set

from engine import position
from engine.exceptions import (
    InsufficientMaterial,
    ThreefoldRepetition,
    FiftyMoveDraw,
    IllegalMove,
    Stalemate,
    Checkmate,
)
from engine.pieces import Piece, Rook, Knight, Bishop, Queen, King, Pawn
from engine.position import Position

from engine.constants import (
    WHITE,
    BLACK,
    PAWN,
    KNIGHT,
    BISHOP,
    ROOK,
    QUEEN,
    KING,
    KINGSIDE,
    QUEENSIDE,
    STARTING_STATE,
)

FEN_PIECES = {
    'r': Rook,
    'n': Knight,
    'b': Bishop,
    'q': Queen,
    'k': King,
    'p': Pawn,
}


class Square:
    def __init__(self, pos: Position, piece: Optional[Piece] = None):
        self.pos = pos
        self.piece = piece

    @property
    def is_occupied(self):
        return self.piece is not None

    def __str__(self):
        if self.is_occupied:
            return str(self.piece)
        else:
            return str(self.pos)

class Board:
    def __init__(self, state: str = STARTING_STATE):
        (
            self.pieces,
            self.turn,
            self.en_passant,
            self.halfmove_clock,
            self.fullmoves,
        ) = from_fen(state)

        # FEN isn't sufficient to describe this, so assume unique
        self.repetitions = {}  # type: Dict[str, int]

    @property
    def has_insufficient_material(self) -> bool:
        if len(self.pieces) > 4:  # Any game with more than 4 pieces can be won
            return False
        elif len(self.pieces) < 3:  # Any game with only kings can't be won
            return True
        elif len(self.pieces) == 3:
            for piece in self.pieces:
                if piece.type in (PAWN, ROOK, QUEEN):  # Any of these can can checkmate
                    return False
            return True
        elif len(self.pieces) == 4:
            bishop_squares = set()
            for piece in self.pieces:
                if piece.type in (PAWN, KNIGHT, ROOK, QUEEN):  # Any of these can can checkmate
                    return False
                elif piece.type == BISHOP:
                    bishop_squares.add(piece.pos.colour)

            # Two bishops on the same colour can't checkmate
            if len(bishop_squares) > 1:
                return False
            return True

    @property
    def has_threefold_repetition(self):
        for fen, repetitions in self.repetitions.items():
            if repetitions >= 3:
                return True
        return False

    def _add_repetition(self):
        fen = self._short_fen
        self.repetitions[fen] = self.repetitions.get(fen, 0)
        self.repetitions[fen] += 1

    def _move(self, start_pos: Position, end_pos: Position, simulate: bool = False):
        """Moves a piece from the start position to the end position if legal."""

        def _reverse_move(
                _board, _start_piece, _start_pos, _taken_piece=None,
        ):
            _start_piece.pos = _start_pos
            if _taken_piece:
                _board.pieces.add(_taken_piece)

        if start_pos == end_pos or not start_pos.in_board or not end_pos.in_board:
            raise IllegalMove(f"{start_pos} to {end_pos} is an illegal move.")

        squares = self.squares
        start_square = squares[start_pos.index]
        start_piece = squares[start_pos.index].piece

        if not end_pos in start_square.piece.legal_moves(self):
            raise IllegalMove(f"{start_pos} to {end_pos} is an illegal move.")

        # Castling attempt, only way to move a King two spaces cardinally
        if start_piece.type == KING and abs(end_pos.file - start_pos.file) == 2:
            can_castle = True
            if end_pos.file > start_pos.file:
                direction = 1
            else:
                direction = -1

            for i in range(2):
                _castle_pos = Position(start_pos.file + (direction * (i + 1)), start_pos.rank)
                if self.attacked_by(_castle_pos, start_piece.colour):
                    can_castle = False
                    break

            # Special move if castling, moving two pieces
            if can_castle and not simulate:
                if direction > 0:   # Kingside castle
                    num_squares = 2
                    castle_type = KINGSIDE
                else:               # Queenside castle
                    num_squares = 3
                    castle_type = QUEENSIDE

                # Move King
                start_piece.pos = end_pos
                start_piece.move_history.append((start_pos, end_pos))

                # Move Rook
                rook = list(filter(
                    lambda p: p.type == ROOK and p.colour == start_piece.colour and p.rook_type == castle_type,
                    self.pieces
                ))[0]
                move_rook = Position(rook.pos.file + (-num_squares * direction), rook.pos.rank)
                rook.pos = move_rook
                rook.move_history.append((rook.pos, move_rook))
                self.halfmove_clock += 1  # Note that we don't add a repetition here
                return
            else:
                raise IllegalMove("Cannot castle, intermediate squares are being attacked.")

        # Regular move
        end_square = squares[end_pos.index]
        if (  # Take pawn by En Passant
            start_piece.type == PAWN and
            self.en_passant is not None and
            end_pos == self.en_passant
        ):
            direction = 1 if start_piece.colour == WHITE else -1
            taken_piece = squares[Position(end_pos.file, end_pos.rank - direction).index].piece
            self.pieces.remove(taken_piece)
        elif end_square.is_occupied:  # Take piece if it exists
            taken_piece = end_square.piece
            self.pieces.remove(taken_piece)
        else:
            taken_piece = None
        start_piece.pos = end_pos  # Move piece

        # If the move would put the player in check, reverse it and declare it illegal
        if self.is_in_check(start_piece.colour):
            _reverse_move(self, start_piece, start_pos, taken_piece)
            raise IllegalMove(f"{start_pos} to {end_pos} would put {start_piece.colour} in check.")

        if simulate:
            _reverse_move(self, start_piece, start_pos, taken_piece)
        else:
            # Mark en passant square if a pawn moves two squares, otherwise clear
            if start_piece.type == PAWN and abs(end_pos.rank - start_pos.rank) == 2:
                self.en_passant = end_pos
            else:
                self.en_passant = None
            start_piece.move_history.append((start_pos, end_pos))

            if start_piece.type == PAWN:  # Promote pawn if it reaches the end of the board
                end_rank = 7 if start_piece.colour == WHITE else 0
                if end_pos.rank == end_rank:
                    self.pieces.remove(start_piece)
                    new_piece = Queen(pos=end_pos, colour=start_piece.colour)  # Automatically choose Queen for now
                    new_piece.move_history = start_piece.move_history
                    self.pieces.add(new_piece)

            # Reset or increment the half move clock
            if start_piece.type == PAWN or taken_piece is not None:
                self.halfmove_clock = 0
                self.repetitions = {}
            else:
                self.halfmove_clock += 1
                self._add_repetition()

    def _can_castle(self, colour: str, castle_type: str) -> bool:
        assert colour in (BLACK, WHITE), f"Invalid colour, {colour}, specified."
        assert castle_type in (QUEENSIDE, KINGSIDE), f"Invalid castle type, {castle_type}, specified."

        king = list(filter(lambda p: p.colour == colour and p.type == KING and p.can_castle, self.pieces))
        if len(king) == 0:
            return False

        rook = list(filter(
            lambda p: p.colour == colour and p.type in ROOK and p.can_castle and p.rook_type == castle_type,
            self.pieces)
        )
        if len(rook) == 0:
            return False
        return True

    def raise_if_game_over(self):
        """Raises an exception if the board is in an end state."""
        if self.halfmove_clock >= 50:
            raise FiftyMoveDraw
        elif self.has_insufficient_material:
            raise InsufficientMaterial
        elif self.has_threefold_repetition:
            raise ThreefoldRepetition
        elif self.is_checkmate(self.turn):
            raise Checkmate
        elif self.is_stalemate(self.turn):
            raise Stalemate

    def player_move(self, start_pos: Position, end_pos: Position):
        squares = self.squares
        start_square = squares[start_pos.index]
        assert start_square.is_occupied, f"No piece at {start_pos}"

        start_piece = squares[start_pos.index].piece
        assert start_piece.colour == self.turn, f"It is {self.turn}'s turn, cannot move enemy piece."

        self._move(start_pos, end_pos, simulate=False)

        if self.turn == BLACK:
            self.fullmoves += 1
            self.turn = WHITE
        else:
            self.turn = BLACK

    def possible_moves(self, colour: str = WHITE) -> Iterable[Tuple[Position, Position]]:
        for piece in filter(lambda p: p.colour == colour, self.pieces):
            for move in piece.legal_moves(self):
                try:
                    self._move(piece.pos, move, simulate=True)
                    yield (piece.pos, move)
                except IllegalMove:
                    pass

    def attacked_by(self, pos: Position, player_colour: str = WHITE) -> bool:
        """Returns whether or not a given square is being attacked by the opposite side."""
        for piece in filter(
            lambda p: p.colour != player_colour,
            self.pieces
        ):
            if pos in piece.legal_moves(self):
                return True
        return False

    def is_in_check(self, colour: str) -> bool:
        """Checks if the associated King is in check on the board."""
        assert colour in [WHITE, BLACK]

        king = list(filter(
            lambda p: p.type == 'k' and p.colour == colour,
            self.pieces
        ))[0]

        # Loop through legal moves of opposing side and return true if they contain the King
        return self.attacked_by(king.pos, colour)

    def is_checkmate(self, colour: str) -> bool:
        """Checks if a given player has been checkmated."""

        if self.is_in_check(colour):
            i = 0
            for _ in self.possible_moves(colour):
                i += 1
                if i > 0:
                    return False
            return True
        else:
            return False

    def is_stalemate(self, colour: str) -> bool:
        """Checks if a game is in stalemate."""

        if not self.is_in_check(colour):
            i = 0
            for _ in self.possible_moves(colour):
                i += 1
                if i > 0:
                    return False
            return True
        else:
            return False

    @property
    def castle_flags(self) -> str:
        flags = ''

        for opt in (
            (WHITE, KINGSIDE, 'K'),
            (WHITE, QUEENSIDE, 'Q'),
            (BLACK, KINGSIDE, 'k'),
            (BLACK, QUEENSIDE, 'q'),
        ):
            if self._can_castle(opt[0], opt[1]):
                flags += opt[2]

        return flags if flags != '' else '-'


    @property
    def _squares_by_rank(self) -> Dict[int, List[Square]]:
        """All squares of the board, organised by rank."""
        by_rank = {}
        for square in self.squares:
            rank = square.pos.rank
            by_rank[rank] = by_rank.get(rank, [])
            by_rank[rank].append(square)
        return by_rank

    @property
    def squares(self) -> List[Square]:
        """All squares of the board, in order from A1, B1... G8, H8."""
        all_squares = []
        i = 0
        for piece in sorted(self.pieces):
            while i < piece.pos.index:
                all_squares.append(
                    Square(pos=position.from_index(i))
                )
                i += 1
            all_squares.append(Square(pos=position.from_index(i), piece=piece))
            i += 1

        # If the last piece on the board does not complete the board, fill it in.
        for j in range(i, 64):
            all_squares.append(Square(pos=position.from_index(j)))

        return all_squares

    @property
    def _short_fen(self) -> str:
        """Short version of the FEN without move numbers for repitition."""
        return ' '.join(self.fen.split(' ')[:4])

    @property
    def fen(self) -> str:
        """
        Returns the board's current state in FE Notation.
        (https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation)
        """
        fen_str = ''
        by_rank = self._squares_by_rank
        for rank in sorted(by_rank.keys(), reverse=True):
            blank_counter = 0
            for square in by_rank[rank]:
                if square.is_occupied:
                    if blank_counter > 0:
                        fen_str += str(blank_counter)
                        blank_counter = 0
                    fen_str += square.piece.code
                else:
                    blank_counter += 1

            if blank_counter > 0:
                fen_str += str(blank_counter)

            if rank > 0:
                fen_str += '/'

        if self.turn:
            _turn = 'w' if self.turn == WHITE else 'b'
            _en_passant = '-' if self.en_passant is None else str(self.en_passant).lower()
            fen_str += f' {_turn} {self.castle_flags} {_en_passant} {self.halfmove_clock} {self.fullmoves}'
        return fen_str

    def __str__(self) -> str:
        by_rank = self._squares_by_rank
        board_str = ''
        for rank in sorted(by_rank.keys(), reverse=True):
            board_str += f'\n{rank + 1} '
            for square in by_rank[rank]:
                if square.is_occupied:
                    board_str += f'[{square.piece.icon}]'
                else:
                    board_str += '[ ]'
        board_str += '\n   A  B  C  D  E  F  G  H '
        return board_str


def from_fen(fen: str = STARTING_STATE) -> Tuple[Set[Piece], str, Position, int, int]:
    rank = 7
    file = 0
    pieces = set()
    turn = None
    en_passant = None
    halfmove_clock = 0
    fullmoves = 1

    components = fen.split(' ')
    for char in components[0]:
        if char.isdigit():
            file += int(char)
        elif char == '/':
            rank -= 1
            file = 0
        else:
            assert char.lower() in list(FEN_PIECES.keys()), f'{char} is not a valid piece in FEN notation.'
            piece = FEN_PIECES[char.lower()](
                pos=Position(file, rank),
                colour=WHITE if char.isupper() else BLACK,
            )
            pieces.add(piece)
            file += 1

    if len(components) > 1:
        turn = WHITE if components[1].lower() == 'w' else BLACK

    if len(components) > 3:
        _en_passant_coord = components[3].upper()
        en_passant = None if _en_passant_coord == '-' else position.from_coord(_en_passant_coord)

    if len(components) > 4:
        halfmove_clock = int(components[4])

    if len(components) > 5:
        fullmoves = int(components[5])

    if len(pieces) < 2:
        raise ValueError(f"Invalid FEN: {fen}")

    return pieces, turn, en_passant, halfmove_clock, fullmoves
