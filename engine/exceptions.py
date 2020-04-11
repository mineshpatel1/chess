class IllegalMove(Exception):
    pass

class Checkmate(Exception):
    pass

class Draw(Exception):
    pass

class Stalemate(Draw):
    pass

class FiftyMoveDraw(Draw):
    pass

class InsufficientMaterial(Draw):
    pass

class ThreefoldRepetition(Draw):
    pass