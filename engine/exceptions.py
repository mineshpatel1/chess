class IllegalMove(Exception):
    pass

class Checkmate(Exception):
    pass

class Draw(Exception):
    MESSAGE = "Match drawn."

class Stalemate(Draw):
    MESSAGE = "Stalemate: player has no legal moves but is not in check."

class FiftyMoveDraw(Draw):
    MESSAGE = "50 moves have occured without moving a pawn or taking a piece."

class InsufficientMaterial(Draw):
    MESSAGE = "Player has insufficient material to checkmate."

class ThreefoldRepetition(Draw):
    MESSAGE = "Same board position has occurred 3 times."