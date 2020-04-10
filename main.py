import log
import board
import position

from pieces import Pawn
from position import Position


def main():
    b = board.Board()
    log.info(b)

    b.move(Position(0, 1), Position(0, 3))
    b.move(Position(0, 3), Position(0, 4))
    b.move(Position(0, 4), Position(0, 5))
    log.info(b.squares[Position(0, 5).index].piece.legal_moves(b))
    #
    # log.info(b.fen)

if __name__ == '__main__':
    main()
