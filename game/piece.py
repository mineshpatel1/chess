from game.constants import (
    WHITE,
    BLACK,
    PieceType,
    PIECE_TYPES,
    PIECE_NAMES,
    PIECE_VALUES,
    PIECE_ICONS,
)


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
        return modifier * PIECE_VALUES[self.type]

    def __str__(self) -> str:
        return f'{self.icon}'

    def __repr__(self) -> str:
        return str(self)