Colour = bool
WHITE = True
BLACK = False

QUEENSIDE = 'queenside'
KINGSIDE = 'kingside'

FILE_NAMES = ["A", "B", "C", "D", "E", "F", "G", "H"]
RANK_NAMES = ["1", "2", "3", "4", "5", "6", "7", "8"]

PieceType = str
PAWN = 'p'
ROOK = 'r'
KNIGHT = 'n'
BISHOP = 'b'
QUEEN = 'q'
KING = 'k'
PIECE_TYPES = (PAWN, ROOK, KNIGHT, BISHOP, QUEEN, KING)

PIECE_NAMES = {
    PAWN: 'Pawn',
    ROOK: 'Rook',
    KNIGHT: 'Knight',
    BISHOP: 'Bishop',
    QUEEN: 'Queen',
    KING: 'King',
}

PIECE_ICONS = {
    PAWN.upper(): '♙',
    ROOK.upper(): '♖',
    KNIGHT.upper(): '♘',
    BISHOP.upper(): '♗',
    QUEEN.upper(): '♕',
    KING.upper(): '♔',
    PAWN: '♟',
    ROOK: '♜',
    KNIGHT: '♞',
    BISHOP: '♝',
    QUEEN: '♛',
    KING: '♚',
}

PIECE_VALUES = {
    PAWN: 100,
    KNIGHT: 320,
    BISHOP: 330,
    ROOK: 500,
    QUEEN: 900,
    KING: 20000,
}

PAWN_POSITION_BASE_VALUES = [
     0,  0,   0,   0,   0,   0,  0,  0,
    50, 50,  50,  50,  50,  50, 50, 50,
    10, 10,  20,  30,  30,  20, 10, 10,
     5,  5,  10,  25,  25,  10,  5,  5,
     0,  0,   0,  20,  20,   0,  0,  0,
     5, -5, -10,   0,   0, -10, -5,  5,
     5, 10,  10, -20, -20,  10, 10,  5,
     0,  0,   0,   0,   0,   0,  0,  0,
]

KNIGHT_POSITION_BASE_VALUES = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  10,  15,  15,  10,   5, -30,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]

BISHOP_POSITION_BASE_VALUES = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]

ROOK_POSITION_BASE_VALUES = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
]

QUEEN_POSITION_BASE_VALUES = [
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10,   0,   0,  0,  0,   0,   0, -10,
    -10,   0,   5,  5,  5,   5,   0, -10,
     -5,   0,   5,  5,  5,   5,   0,  -5,
      0,   0,   5,  5,  5,   5,   0,  -5,
    -10,   5,   5,  5,  5,   5,   0, -10,
    -10,   0,   5,  0,  0,   0,   0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20,
]

KING_POSITION_BASE_VALUES = [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
     20,  20,   0,   0,   0,   0,  20,  20,
     20,  30,  10,   0,   0,  10,  30,  20,
]

Direction = str
NORTH = 'n'
NORTHEAST = 'ne'
EAST = 'e'
SOUTHEAST = 'se'
SOUTH = 's'
SOUTHWEST = 'sw'
WEST = 'w'
NORTHWEST = 'nw'


CARDINALS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
DIAGONALS = [(1, 1), (-1, -1), (-1, 1), (1, -1)]

STARTING_STATE = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
