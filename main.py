import log
import board
import position

from board import Board
from pieces import Pawn
from position import Position


def main():
    b = Board(state="rnbqkbnr/pppppp1p/8/8/P1R1p3/8/1PPP1PPP/1NBQKBNR")

    rook = b.squares[Position(2, 3).index].piece
    for x in sorted(rook._moves(b)):
        log.info(x)

if __name__ == '__main__':
    main()
