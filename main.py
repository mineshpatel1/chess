import log
import board
import position

from board import Board
from pieces import Pawn
from position import Position


def main():
    b = Board('rnbqkb2/ppppp1p1/5p2/2n1r2p/3KP3/3P4/PPP2PPP/RNBQ1BNR')
    x = b.is_in_check(True)
    log.info(x)
    # piece = b.squares[Position(3, 3).index].piece
    # log.info(piece.legal_moves(b))

    b = Board('rnb1kbnr/pppp1ppp/4p3/8/7q/5P2/PPPPP1PP/RNBQKBNR')
    log.info(b.is_in_check(True))

if __name__ == '__main__':
    main()
