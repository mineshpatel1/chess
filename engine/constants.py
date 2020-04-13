Colour = str
WHITE = 'white'
BLACK = 'black'

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

PIECE_POINTS = {
    PAWN: 1,
    KNIGHT: 3,
    BISHOP: 3,
    ROOK: 5,
    QUEEN: 9,
    KING: 90,
}

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
