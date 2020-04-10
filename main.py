import log
import board
import position

from pieces import Pawn
from position import Position


def main():
    b = board.Board()
    log.info(b)

    b.move(Position(0, 1), Position(0, 4))
    log.info(b)

if __name__ == '__main__':
    main()
